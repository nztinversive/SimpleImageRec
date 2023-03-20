from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from flask import render_template

app = Flask(__name__)

model = load_model('cifar10_cnn_model.h5')
label_map = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_image(image):
    image = Image.open(image).convert('RGB')
    image = image.resize((32, 32))
    image = np.array(image).astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    image = request.files['file']
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_label = label_map[np.argmax(predictions)]

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    # Run the Flask app on a specified host and port
    app.run(host='0.0.0.0', port=3000)
