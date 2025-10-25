import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
import io
from PIL import Image

# 1. Initialization
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "my_model.keras"
model = load_model(MODEL_PATH)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] 


def model_predict(image_data, model):
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((32, 32))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    
    return CLASSES[predicted_class_index], predictions[0]

# 3. Flask Routes
@app.route('/', methods=['GET'])
def index():
    # Render the HTML template for the home page (where the user uploads an image)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the request
        file = request.files['file']
        if file:
            # Read the image data
            img_data = file.read()
            # Get prediction
            label, probabilities = model_predict(img_data, model)
            
            # Return prediction result
            return jsonify({
                'prediction': label,
                'confidence': f"{np.max(probabilities) * 100:.2f}%"
            })
    return jsonify({'error': 'No file uploaded or invalid request.'})

if __name__ == '__main__':
    app.run(debug=True)
