import keras.utils as image
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)#masih pakai flash

dic = {0 : 'WHITESPOT', 1 : 'BACTERIALFIN', 2 : 'REDSPOT', 3 : 'HEALTHY'}

model = load_model('pat/model.h5')#ganti path yg sesuai

model.make_predict_function()

def predict_label(img_path):
	i = tf.keras.utils.load_img(img_path, target_size=(224,224))
	i = tf.keras.utils.img_to_array(i)/255.0
	i = i.reshape(1, 224, 224, 3)
	p = np.argmax(model.predict(i),axis=1)
	return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
