from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load your pre-trained model
MODEL_PATH = "cnn_image_classifier.keras"
model = load_model(MODEL_PATH)

# Define class labels (modify according to your dataset)
class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']  

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    # Save the uploaded file
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Preprocess the image
    img = load_img(filepath, target_size=(32, 32))  # Resize according to your model's input
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    return render_template("index.html", label=predicted_class, confidence=confidence, image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
