from flask import Flask, jsonify, request
from src.utils import base64_decode
from src.Model import LMModel
from pathlib import Path
import os
from json_result import get_json
from flask_cors import CORS
from flask_sslify import SSLify

app = Flask(__name__)
CORS(app)
sslify = SSLify(app)

# Model configuration
model_check_point = Path("./checkpoints/checkpoints_path_baseepoch_.pt")
labels_count = 24
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model = LMModel(model_check_point, labels_count)


@app.route("/", methods=["POST"])
def main():
    # some pillow image contain 4 channels, to remove the fourth channel we need to convert image model to RGB
    image = base64_decode(request.json["image"]).convert("RGB")
    pred = model(image=image)
    res = get_json(pred, image)
    return jsonify(res)


if __name__ == '__main__':
    context = ('./ssl/server.crt', './ssl/server.key')
    app.run(host="0.0.0.0", port=8000, debug=False, ssl_context=context)
