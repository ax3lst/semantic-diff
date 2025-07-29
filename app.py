from flask import Flask, request, send_file, jsonify, url_for
import os, tempfile, contextlib, uuid, time
from semantic_diff import main as generate_report   # Your original script
from datetime import datetime

app = Flask(__name__)

# Folder to store reports
REPORT_DIR = os.path.abspath("reports")
os.makedirs(REPORT_DIR, exist_ok=True)

@contextlib.contextmanager
def tmp_workdir():
    with tempfile.TemporaryDirectory() as d:
        cwd_before = os.getcwd()
        os.chdir(d)
        try:
            yield d
        finally:
            os.chdir(cwd_before)

@app.route("/diff", methods=["POST"])
def diff():
    if "old" not in request.files or "new" not in request.files:
        return jsonify({"error": "Bitte 'old' und 'new' JSON-Dateien hochladen."}), 400

    with tmp_workdir() as work:
        old_path = os.path.join(work, "old.json")
        new_path = os.path.join(work, "new.json")
        request.files["old"].save(old_path)
        request.files["new"].save(new_path)

        generate_report(path_old=old_path, path_new=new_path)

        # Generate a unique file name
        filename = f"diff_report_{uuid.uuid4().hex}.html"
        save_path = os.path.join(REPORT_DIR, filename)
        os.rename(os.path.join(work, "diff_report.html"), save_path)

        # Return a link to the file
        download_url = url_for('download_report', filename=filename, _external=True)
        return jsonify({"download_url": download_url})

@app.route("/download/<filename>", methods=["GET"])
def download_report(filename):
    file_path = os.path.join(REPORT_DIR, filename)
    if not os.path.isfile(file_path):
        return jsonify({"error": "Datei nicht gefunden"}), 404
    return send_file(file_path, mimetype="text/html", as_attachment=True, download_name=filename)

# Optional: Cleanup function (run separately, e.g. daily)
def delete_old_reports(days=30):
    cutoff = time.time() - days * 86400  # 30 days in seconds
    for file in os.listdir(REPORT_DIR):
        path = os.path.join(REPORT_DIR, file)
        if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
            os.remove(path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)