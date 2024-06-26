from __future__ import division, print_function
from flask import Flask,render_template,request
import os
import numpy as np
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from werkzeug.utils import secure_filename

app=Flask(__name__)

def model_predict(img_path):
    np.set_printoptions(suppress=True)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(img_path)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    model = tensorflow.keras.models.load_model('ripeness.h5')
    preds = ""
    prediction = model.predict(data)
    if np.argmax(prediction)==0:
        preds = f"Unripe"
    elif np.argmax(prediction)==1:
        preds = f"Overripe"
    else :
        preds = f"ripe"
    return preds
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))#This is going to be your folder name
        f.save(file_path)
        preds = model_predict(file_path)
        return render_template('success.html',preds=preds)
    return None
if __name__=='__main__':
    app.run(debug=True)