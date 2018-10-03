import cv2
import numpy as np
import sys
import io
from PIL import Image
import base64
from flask import Flask, jsonify, request
import os, shutil
from keras.models import load_model
from keras import Model
from keras.preprocessing import image

# Load model

async def getModel():
   model = await load_model("../Keras/zahlErkennung2.h5")
   return model


def loadImage(request):
   '''
   img = Image.open(request.files['fileImage'])
   img = np.array(img)
   img = cv2.resize(img, (56, 56))
   img = img / 255
   img = cv2.cvtColor(np.array(img), cv2.)
   img = np.expand_dims(img, axis=0)
   '''

   img = image.load_img(request.files['fileImage'], target_size=(56, 56))
   imgArray = image.img_to_array(img)
   imgArray = imgArray / 255.0
   imgArray = np.expand_dims(imgArray, axis=0)
   print(imgArray.shape, sys.stderr)

   return imgArray

app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
   return "clgt"

@app.route("/sample", methods=["POST"])
def running():
   return "a Post request"

@app.route("/hello", methods=["POST"])
def hello():
   img = loadImage(request)
   #model = load_model("../Keras/zahlErkennung2.h5")
   model = load_model("../Keras/zahlErkennung2.h5")
   results = model.predict(img)
   print(np.argmax(results, axis=1) + 10, file=sys.stderr)
   del model

   return "clgt"


if __name__ == "__main__":
   app.run(debug=False)