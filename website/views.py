import io

import cv2
import numpy as np
from flask import Blueprint, render_template, redirect, url_for, Response, request
from .camera import Camera
from PIL import Image
import time

views = Blueprint("views", __name__)

img_byte_arr = None


@views.route('/', methods=['POST', 'GET'])
def base():
    if request.method == "POST":
        global img_byte_arr
        img = Image.open(request.files["userfile"])
        img = faceDetect(img)

        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format="JPEG")
        img_byte_arr = imgByteArr.getvalue()
    return redirect(url_for('.stream'))


@views.route('/stream')
def stream():
    return render_template("stream.html")


@views.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def gen():
    while True:
        if img_byte_arr is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_byte_arr + b'\r\n')


def faceDetect(image):
    cv2Image = np.array(image.convert("RGB"))
    gray_image = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    for (x, y, w, h) in face:
        cv2.rectangle(cv2Image, (x, y), (x + w, y + h), (0, 255, 0), 4)

    sendAudio()
    return Image.fromarray(cv2Image)

def sendAudio():


