Overview
This project is a web-based image classifier built using Python and Flask. It allows users to upload images and get predictions about the category of the image based on a trained neural network model. The app uses a pre-trained Keras model for image classification and provides a simple, intuitive interface for interacting with it.

Features
-Upload an image for classification.
-Display the predicted class of the uploaded image.
-Web interface powered by Flask.
-Model trained using Keras.

Technologies Used
-Python: Main programming language.
-Flask: Web framework for building the app.
-Keras: Deep learning framework used for training the image classifier.
-HTML/CSS: Frontend for the web interface.
-JavaScript: Enhances user interaction.
-TensorFlow: Backend for the Keras model.
File Structure
/Image-Classifier-WebApp
├── app.py                # Main Flask app file
├── cnn_image_classifier.keras # Pre-trained Keras model
├── static/               # Folder for static assets (images, CSS)
├── templates/            # HTML files for the frontend
├── requirements.txt      # Python dependencies
└── README.md             # This file
Acknowledgements
Thanks to the Keras and TensorFlow libraries for providing powerful tools for building and training the model.
Inspired by various open-source image classification tutorials.
