import os
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# Load the ONNX model
session = ort.InferenceSession("denoising_autoencoder.onnx", providers=['CPUExecutionProvider'])

@app.route("/", methods=["GET", "POST"])
def index():
    input_path = None
    denoised_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            # Save uploaded image
            input_path = os.path.join("static", "input.png")
            file.save(input_path)

            # Preprocess image
            image = Image.open(input_path).convert("L")  # grayscale
            image = image.resize((28, 28))
            input_array = np.array(image).astype(np.float32) / 255.0
            input_array = input_array.reshape(1, 28, 28, 1)  # (1, 28, 28, 1)

            # Run inference
            inputs = {session.get_inputs()[0].name: input_array}
            outputs = session.run(None, inputs)
            denoised = outputs[0].reshape(28, 28)

            # Save denoised image
            denoised_img = Image.fromarray((denoised * 255).astype(np.uint8))
            denoised_path = os.path.join("static", "denoised.png")
            denoised_img.save(denoised_path)

    return render_template("index.html", input_image=input_path, denoised_image=denoised_path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
