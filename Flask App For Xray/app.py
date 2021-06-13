
#::: Import modules and packages :::
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam

# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os

#::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)

# ::: Prepare Keras Model :::
# Model files
MODEL_ARCHITECTURE = './model/Chest_Xray_Pneumonia_json.json' 
MODEL_WEIGHTS = './model/Chest_Xray_Pneumonia.h5'

# Load the model from external files
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Get weights into the model
model.load_weights(MODEL_WEIGHTS)
print('Model loaded. Check http://127.0.0.1:5000/')


# ::: MODEL FUNCTIONS :::
def model_predict(img_path, model):
	
	print(model)
	Xray = image.load_img(img_path, target_size=(150, 150))
	Xray = image.img_to_array(Xray)
	Xray = np.expand_dims(Xray, axis=0)
	Xray = Xray / 255
	print(Xray.shape)
	
	model.compile(loss= 'binary_crossentropy', optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics = ['accuracy'])
	print(model)

	prediction = model.predict_classes(Xray)
	print("Prediction Class: ", prediction)

	return prediction


# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':


		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		# Make a prediction
		prediction = model_predict(file_path, model)

		Answer = ""
		if (prediction < 1):
    			Answer = "Congratulation You are Normal"
		else:
    			Answer = "Sorry but you have Pneumonia"
				
		return Answer

if __name__ == '__main__':
	app.run(debug = True)