from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def gen():
    body = request.get_json() or {}
    prompt = body.get("prompt", "")
    return jsonify({"text": f"MOCK LLM RESPONSE for prompt length {len(prompt)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)