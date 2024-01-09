from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image
from io import BytesIO

app = Flask(__name__, template_folder="template")
DAC = load_model('classification.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    if request.method == 'POST':
        file = request.files['file']
        img = image.load_img(BytesIO(file.read()), target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        result = DAC.predict(img_array)
        prediction = 'Dog' if result[0][0] > 0.5 else 'Cat'
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
