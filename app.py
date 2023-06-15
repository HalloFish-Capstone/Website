from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np


app = Flask(__name__)
model = tf.keras.models.load_model('path/to/model.h5')

@app.route('/')
def home():
    return render_template('uploadprediksi.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil file gambar dari form upload
    image = request.files['file']

    # Baca file gambar dan ubah ke format yang sesuai untuk model
    img = Image.open(image)
    img = img.resize((224, 224))  # Sesuaikan ukuran gambar dengan model
    img = np.expand_dims(img, axis=0)
    img = np.array(img) / 255.0  # Normalisasi piksel gambar

    # Lakukan prediksi menggunakan model
    result = model.predict(img)
    classes = ['REDSPOT', 'WHITESPOT', 'HEALTHY', 'BACTERIAL']
    predicted_class = classes[np.argmax(result)]

    # Kembalikan hasil prediksi ke halaman
    return render_template('result.html', prediction=predicted_class)
if __name__ == '__main__':
    app.run(debug=True)
