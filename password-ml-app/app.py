import os
from flask import Flask, render_template, request, jsonify
from engine.ml_engine import password_assistant_with_reuse, initialize_models

app = Flask(__name__, template_folder="templates_", static_folder="static")

# initialize/load models once
initialize_models()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        password = request.form.get("password", "")
        result = password_assistant_with_reuse(password)
    return render_template("index.html", result=result)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json() or {}
    password = data.get("password", "")
    res = password_assistant_with_reuse(password)
    return jsonify(res)


@app.route("/result", methods=["POST"])
def result_page():
    # Accepts form POSTs (from the main page) and renders a standalone result page
    password = request.form.get("password", "")
    res = password_assistant_with_reuse(password)
    return render_template("result.html", result=res)


@app.route("/health")
def health():
    return "OK", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

