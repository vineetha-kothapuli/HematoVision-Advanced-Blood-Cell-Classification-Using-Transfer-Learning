import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Ensure 'static' folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load your trained model
model = load_model('BloodCellClassifier.h5')  # Ensure this file exists

# Update this based on your trained model's classes
class_names = ['WBC', 'RBC', 'Platelets']  # Replace with your actual class labels

# Prediction function
def model_predict(img_path, model):
    img = load_img(img_path, target_size=(224, 224))  # Match input size used in model
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)
    return predicted_class, confidence

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "❌ No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "❌ No selected file", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    prediction, confidence = model_predict(filepath, model)

    return render_template('result.html',
                           prediction=prediction,
                           confidence=confidence,
                           image_file=filepath)

if __name__ == '__main__':
    app.run(debug=True)
