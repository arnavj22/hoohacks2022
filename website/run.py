from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from threading import Thread
import time
from processing import *
global rec, rec_frame, out
rec = False

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # use 0 for web camera
def pushup():
    pushups()
    return render_template('analyzepushup.html', data = {'done': 'done'})
def situp():
    curl_ups()
    return render_template('analyzesitup.html', data = {'done': 'done'})
def jump():
    return render_template('analyzejump.html', data = {'done': 'done'})
def plank():
    planked()
    return render_template('analyzeplank.html', data = {'done': 'done'})
def squat():
    squats()
    return render_template('analyzesquats.html', data = {'done': 'done'})
def bicep():
    bicepcurls()
    return render_template('analyzebicep.html', data = {'done': 'done'})

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)
def gen_frames():  # generate frame by frame from camera
    global out, rec_frame
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if(rec):
            rec_frame=frame
            frame= cv2.putText(cv2.flip(frame,1),"Recording", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
            frame = cv2.flip(frame,1)
        try:
            ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass
                
        else:
            pass
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/requests',methods=['POST','GET'])
def tasks():
    global camera
    if request.method == 'POST':
        if  request.form.get('rec').startswith('Start/Stop'):
            global rec, out
            rec= not rec
            if(rec):
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(camera.get(3)),int(camera.get(4))))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                if(request.form.get('rec').endswith('Pushup')):
                    return pushup()
                elif(request.form.get('rec').endswith('Situp')):
                    return situp()
                elif(request.form.get('rec').endswith('Jump')):
                    return jump()
                elif(request.form.get('rec').endswith('Plank')):
                    return plank()
                elif(request.form.get('rec').endswith('Squats')):
                    return squat()
                elif(request.form.get('rec').endswith('Bicep')):
                    return bicep()
    elif request.method=='GET':
        return render_template('analyzepushup.html')
    if request.form.get('rec').endswith('Pushup'):
        return apu()
    elif request.form.get('rec').endswith('Situp'):
        return asu()
    elif request.form.get('rec').endswith('Jump'):
        return aj()
    elif request.form.get('rec').endswith('Plank'):
        return apl()
    elif request.form.get('rec').endswith('Squats'):
        return asq()
    return abc()
   
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
@app.route('/analyzesquats')
def asq():
    return render_template('analyzesquats.html')
@app.route('/features')
def features():
    return render_template('features.html')
	
@app.route('/uploaderpushup', methods = ['GET', 'POST'])
def uploadpu():
    if request.method == 'POST':
        f = request.files['file']
        f.save("output.mp4")
        return pushup()
@app.route('/uploadersitup', methods = ['GET', 'POST'])
def uploadsu():
    if request.method == 'POST':
        f = request.files['file']
        f.save("output.mp4")
        return situp()
@app.route('/uploaderplank', methods = ['GET', 'POST'])
def uploadp():
    if request.method == 'POST':
        f = request.files['file']
        f.save("output.mp4")
        return plank()
@app.route('/uploaderbicep', methods = ['GET', 'POST'])
def uploadbc():
    if request.method == 'POST':
        f = request.files['file']
        f.save("output.mp4")
        return bicep()
@app.route('/uploaderjump', methods = ['GET', 'POST'])
def uploadj():
    if request.method == 'POST':
        f = request.files['file']
        f.save("output.mp4")
        return jump()
@app.route('/uploadersquat', methods = ['GET', 'POST'])
def uploadsq():
    if request.method == 'POST':
        f = request.files['file']
        f.save("output.mp4")
        return squat()
	
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
     app.run(host = '127.0.0.1', port = 5000, threaded = True)
