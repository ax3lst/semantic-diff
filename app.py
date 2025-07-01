# app.py
from flask import Flask, request, send_file, jsonify
import os, tempfile, contextlib
from semantic_diff import main as generate_report   # nutzt dein Originalskript

app = Flask(__name__)

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
    # Erwartet multipart/form-data mit Feldern 'old' und 'new'
    if "old" not in request.files or "new" not in request.files:
        return jsonify({"error": "Bitte 'old' und 'new' JSON-Dateien hochladen."}), 400

    with tmp_workdir() as work:
        old_path = os.path.join(work, "old.json")
        new_path = os.path.join(work, "new.json")
        request.files["old"].save(old_path)
        request.files["new"].save(new_path)

        # Report generieren; schreibt diff_report.html in dasselbe Verzeichnis
        generate_report(path_old=old_path, path_new=new_path)

        return send_file(
            os.path.join(work, "diff_report.html"),
            mimetype="text/html",
            as_attachment=True,
            download_name="diff_report.html",
        )

# Lokaler Start für Tests
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
