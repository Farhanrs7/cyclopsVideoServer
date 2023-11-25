import io
import socket
import time
import urllib
from threading import Thread

import cv2
import numpy as np
from flask import Blueprint, render_template, redirect, url_for, Response, request
from PIL import Image
import pyaudio

import torch

# # Model
from matplotlib import pyplot as plt

model = torch.hub.load("C:/Users/farha/PycharmProjects/pythonProject2/yolov7", 'custom',
                       'C:/Users/farha/PycharmProjects/pythonProject2/yolov7.pt',
                       source='local')
model.classes = [0]
model.cpu()

views = Blueprint("views", __name__)

img_byte_arr = None

# put the current ip address of client in the network
# android_client_address = "192.168.43.1"
android_client_address = "192.168.0.161"
location_url = "http://" + android_client_address + ":10000"

server_socket_feature_enable = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket_feature_enable.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

client_feature_enable_address = (android_client_address, 50000)

client_audio_address = (android_client_address, 50005)

p = pyaudio.PyAudio()  # Create an interface to PortAudio
chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
fs = 16000
channels = 1

audioStream = True
audioReceived = bytes()

# location information
latitude = ''
longitude = ''

videoStatus = False
frameImg = None




@views.route('/')
def base():
    return redirect(url_for('.stream'))


@views.route('/stream')
def stream():
    return render_template("stream.html")


@views.route('/talk')
def talk():
    return render_template("talk.html")


@views.route('/startTalking')
def startTalking():
    print("Starting talking service")
    message = "start_audio".encode('utf-8')
    global audioStream
    audioStream = True
    server_socket_feature_enable.sendto(message, client_feature_enable_address)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 50005))

    recThread = Thread(target=receiveAudio, args=(server_socket,))
    sendThread = Thread(target=sendAudio, args=(server_socket,))
    recThread.start()
    sendThread.start()
    recThread.join()
    sendThread.join()

    return redirect(url_for(".stream"))


@views.route('/stopTalking')
def stopTalking():
    print("Stopping the talk")
    message = "stop_audio".encode('utf-8')
    server_socket_feature_enable.sendto(message, client_feature_enable_address)
    global audioStream
    audioStream = False
    return redirect(url_for('.stream'))


def receiveAudio(server_socket1):
    streamOut = p.open(format=sample_format,
                       channels=channels,
                       rate=fs,
                       output=True)

    while audioStream:
        message, address = server_socket1.recvfrom(1024 * 8)
        streamOut.write(message)


def sendAudio(server_socket1):
    streamIn = p.open(format=sample_format, channels=channels,
                      rate=fs, frames_per_buffer=chunk, input=True,
                      input_device_index=2)
    while audioStream:
        address = (android_client_address, 50005)
        send_bytes = streamIn.read(chunk)
        server_socket1.sendto(send_bytes, address)

@views.route('/location')
def location():
    return render_template("location.html", longitude=longitude, latitude=latitude)


@views.route('/getLocation')
def getLocation():
    print("getting location")
    data = ["null:0", "null:0"]
    try:
        for index, line in enumerate(urllib.request.urlopen(location_url)):
            if index == 2:
                break
            data[index] = line.decode('utf-8')
            print(line.decode('utf-8'))


    except:
        print("error")
    global latitude, longitude
    longitude = float(data[0].split(':')[1])
    latitude = float(data[1].split(':')[1])
    return redirect(url_for('.location'))


@views.route('/video_feed')
def video_feed():
    print(videoStatus)
    if videoStatus:
        return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response()


@views.route('/enableVideoStream')
def enableVideoStream():
    print("Starting video streaming")
    message = "start_video".encode('utf-8')
    server_socket_feature_enable.sendto(message, client_feature_enable_address)
    global videoStatus
    videoStatus = True

    # start ai detection algorithm
    aiThread = Thread(target=obstacleDetect, args=())
    aiThread.daemon = True
    aiThread.start()

    return redirect(url_for('.stream'))


@views.route('/disableVideoStream')
def disableVideoStream():
    print("Stopping video streaming")
    message = "stop_video".encode('utf-8')
    server_socket_feature_enable.sendto(message, client_feature_enable_address)
    global videoStatus
    videoStatus = False
    return redirect(url_for('.stream'))


def gen():
    video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    video_socket.bind(('0.0.0.0', 50006))
    while videoStatus:
        message, address = video_socket.recvfrom(65000)
        global frameImg
        frameImg = Image.open(io.BytesIO(message))

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + message + b'\r\n')


def obstacleDetect():
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    ai = True
    startTime = time.time()
    while videoStatus:
        if frameImg is not None:
            if ai:
                results = model(frameImg, size=160)
                if len(results.pred[0]) > 0:
                    index = int(results.pred[0][0][-1])
                    res = results.names[index]
                    print(res)
                    alert = ("alert_" + res + ".").encode('utf-8')
                    server_socket_feature_enable.sendto(alert, client_feature_enable_address)
                    ai = False
                    startTime = time.time()

            elapsedTime = time.time() - startTime
            if elapsedTime >= 5:
                ai = True

            img = np.array(frameImg)
            kernel_size = 5
            blur_gray = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            res = cv2.Canny(blur_gray, 50, 200)
            res = cv2.merge([res, res, res])

            # image with line
            crop_ratio = 4
            img_width = img.shape[1]
            img_height = img.shape[0]
            img_black = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            white = (255, 255, 255)
            red = (0, 0, 255)
            x1y1 = (int(img_width / 2), img_height)
            x2y2 = (int(img_width / 2), img_height - int(img_height / crop_ratio))
            cv2.line(img_black, x1y1, x2y2, white)

            # drawing line for original image as well
            # cv2.line(img, x1y1, x2y2, white)

            intersect = cv2.bitwise_and(img_black, res)
            if np.sum(intersect) > 0:
                alert = ("alert_" + "beep" + ".").encode('utf-8')
                server_socket_feature_enable.sendto(alert, client_feature_enable_address)
                cv2.line(img_black, x1y1, x2y2, red)

            res = cv2.bitwise_or(res, img_black)

            cv2.imshow('image', res)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
