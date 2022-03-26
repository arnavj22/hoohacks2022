from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
@app.route('/stream')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/analyzesitup')
def asu():
    return render_template('analyzesitup.html')
@app.route('/analyzepushup')
def apu():
    return render_template('analyzepushup.html')
@app.route('/analyzeplank')
def apl():
    return render_template('analyzeplank.html')
@app.route('/analyzebicep')
def abc():
    return render_template('analyzebicep.html')
@app.route('/analyzejump')
def aj():
    return render_template('analyzejump.html')
@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
     app.run(host = '127.0.0.1', port = 5000)
