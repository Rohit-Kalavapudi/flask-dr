from flask import Flask
from keras.models import load_model
import tensorflow_hub as hub
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello_world():
    # left=cv2.imread("D:/DR/Data/1_Aug_1/36_left.jpeg_0_355.jpeg")
    left=None

    model=tf.keras.models.load_model('final.h5', custom_objects={'KerasLayer':hub.KerasLayer})
    res1=predict(left,model)
    return str(res1)

def predict(img,model):
    img1=mpimg.imread(r"D:\DR\Data\1_Aug_1\36_left.jpeg_0_355.jpeg")
    img1=cv2.resize(img1,(224,224),3)
    img1=np.array(img1)/255.0
    img1[np.newaxis,...].shape
    prediction=model.predict(img1[np.newaxis,...])
    prediction=np.argmax(prediction)
    # ar=np.array([left])
    # prediction=model.predict()
    # print(prediction)
    if (prediction==0):
        res='no dr'
    elif(prediction==1):
        res= 'mild dr'
    elif(prediction==2):
        res= 'moderate dr'
    elif(prediction==3):
        res= 'severe'
    else:
        res= 'proliferate' 
    # res="sdf" 
    return res


def pred():
    left=plt.imread(r"D:\DR\Data\1_Aug_1\36_left.jpeg_0_355.jpeg")
    model=tf.keras.models.load_model('final.h5', custom_objects={'KerasLayer':hub.KerasLayer})
    res1=predict(left,model)
    res2=predict(right,model)

