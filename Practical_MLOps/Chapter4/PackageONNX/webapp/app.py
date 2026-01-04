from flask import Flask, request, jsonify
import torch
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime
import os

app = Flask(__name__)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "roberta-sequence-classification-9.onnx")
session = onnxruntime.InferenceSession(model_path)


@app.route("/predict", methods=["POST"])
def predict():
    input_ids = torch.tensor(
    tokenizer.encode(request.json[0], add_special_tokens=True)
    ).unsqueeze(0)
    if input_ids.requires_grad:
        numpy_array = input_ids.detach().cpu().numpy()
    else:
        numpy_array = input_ids.cpu().numpy()

    inputs = {session.get_inputs()[0].name: numpy_array}
    out = session.run(None, inputs)
    result = np.argmax(out)
    return jsonify({"positive": bool(result)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)