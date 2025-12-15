# app.py
from flask import Flask, request, send_file, jsonify, url_for
import os, tempfile, contextlib, uuid, time
from pathlib import Path

from dotenv import load_dotenv

# import the refactored helper
from combined_diff import diff_documents

load_dotenv()

app = Flask(__name__)

# where finished HTML reports live
REPORT_DIR = Path("reports").resolve()
REPORT_DIR.mkdir(exist_ok=True)


@contextlib.contextmanager
def tmp_workdir():
    """Create a throw-away working directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)            # return as Path object for convenience


@app.route("/diff", methods=["POST"])
def diff_route():
    # sanity-check the upload
    if "old" not in request.files or "new" not in request.files:
        return jsonify({"error": "Bitte sowohl 'old' als auch 'new' hochladen."}), 400

    with tmp_workdir() as workdir:
        # save uploads to temp files
        old_path = workdir / "old.json"
        new_path = workdir / "new.json"
        request.files["old"].save(old_path)
        request.files["new"].save(new_path)

        # run the diff – let the helper create the HTML inside workdir
        report_local = diff_documents(
            old=old_path,
            new=new_path,
            out=workdir / "combined_report.html",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # move the HTML into our public report folder under a unique name
        final_name = f"diff_report_{uuid.uuid4().hex}.html"
        final_path = REPORT_DIR / final_name
        os.replace(report_local, final_path)

    # hand back the absolute download URL
    return jsonify(
        {"download_url": url_for("download_report", filename=final_name, _external=True)}
    )


@app.route("/download/<filename>")
def download_report(filename):
    path = REPORT_DIR / filename
    if not path.is_file():
        return jsonify({"error": "Datei nicht gefunden"}), 404
    return send_file(path, mimetype="text/html", as_attachment=True, download_name=filename)


# housekeeping helper (call from a cron job or similar)
def delete_old_reports(days: int = 30):
    cutoff = time.time() - days * 86400
    for p in REPORT_DIR.iterdir():
        if p.is_file() and p.stat().st_mtime < cutoff:
            p.unlink()


if __name__ == "__main__":
    # Flask’s auto-reloader can interfere with tmp dirs; disable if you like
    app.run(host="0.0.0.0", port=8000, debug=True)
