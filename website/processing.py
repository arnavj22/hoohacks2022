import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import math
from matplotlib.scale import NaturalLogTransform
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
def vid_to_frames():
    cap = cv2.VideoCapture("output.mp4")

    imgs, frames = [], []
    count = 0
    total_frames = 0
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        res=(int(width), int(height))

        frame = None
        while True:
            total_frames += 1
            try:
                is_success, frame = cap.read()
            except cv2.error:
                continue
            if not is_success:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            lnd = get_key_landmarks(image)
            if lnd != -1:
                imgs.append(image)
                frames.append(lnd)
            #plt.imshow(image); plt.show()
            count += 15
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    cap.release()
    return frames, imgs
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image, title=None, filename = None):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("graphs/" + filename, img)
#   plt.title(title)
#   plt.axis(False)
#   plt.imshow(img); plt.show()

def show_marks(image, title=None, filename = None):
  # Run MediaPipe Pose and draw pose landmarks.
  with mp_pose.Pose(
      static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
    # Convert the BGR image to RGB and process it with MediaPipe Pose.
    # results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    results = pose.process(image)
    # Print nose landmark.
    image_hight, image_width, _ = image.shape

    # Draw pose landmarks.
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    resize_and_show(annotated_image, title=title, filename = filename)
# Run MediaPipe Pose and draw pose landmarks.
def get_key_landmarks(frame):
  with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
    # results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    try:
      results = pose.process(frame)
      left_shoulder = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z)
      right_shoulder = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z)
      right_hip = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z)
      left_hip = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z)
      right_heel = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].z)
      left_heel = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].z)
      right_feet = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z)
      left_feet = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z)
      left_elbow = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z)
      right_elbow = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z)
      left_wrist = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z)
      right_wrist = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z)
      left_knee = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z)
      right_knee = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z)
      return {'l_shoulder':left_shoulder, 'r_shoulder': right_shoulder, 'l_hip': left_hip, 'r_hip': right_hip, 'l_heel': left_heel, 
          'r_heel': right_heel, 'r_foot': right_feet, 'l_foot': left_feet, 'l_elbow' : left_elbow, 
          'r_elbow' : right_elbow, 'l_wrist' : left_wrist, 'r_wrist' : right_wrist, 'l_knee' : left_knee, 'r_knee' : right_knee}
    except:
      return -1
def pushups():
    frames, imgs = vid_to_frames()
    topFrame, topIdx = frames[0], 0
    switchFrame, switchIdx = [], []
    for i, frame in enumerate(frames):
        # plt.title(f'Left Shoulder z: {frames[i]["l_shoulder"][2]}')
        # plt.imshow(imgs[i]); plt.show()
        if i == 0 or i == len(frames)-1: continue
        
        if ((frames[i]['l_shoulder'][2]+0.01) - (frames[i-1]['l_shoulder'][2]))*((frames[i]['l_shoulder'][2]+0.01) - (frames[i+1]['l_shoulder'][2])) > 0:
            switchFrame.append(frames[i])
            switchIdx.append(i)

    for i, frame in enumerate(switchFrame):
        message = ""
        if abs(frames[switchIdx[i]]['l_shoulder'][2]) > abs(frames[switchIdx[i]]['l_elbow'][2]) or abs(frames[switchIdx[i]]['r_shoulder'][2]) > abs(frames[switchIdx[i]]['r_elbow'][2]):
            message = 'not going down far enough'
        else:
            message = 'Nice Form'
        show_marks(imgs[switchIdx[i]], title=f'Pose Estimation Results 00:01:15: {message}',  filename = "pushups" + str(i + 1) + ".png")
def squats():
    frames, imgs = vid_to_frames()
    topFrame, topIdx = frames[0], 0
    bottomFrame, bottomIdx = [], []
    for i, frame in enumerate(frames):
        if i == 0: continue
        if i == len(frames)-1: continue
        if frames[i]['l_hip'][2]+0.002 < frames[i-1]['l_hip'][2] and frames[i]['l_hip'][2]+0.002 < frames[i+1]['l_hip'][2]:
            bottomFrame.append(frames[i])
            bottomIdx.append(i)
        if frame['l_hip'][2] > topFrame['l_hip'][2]:
            topFrame = frame
            topIdx = i

    #print(len(bottomFrame))
    #print(topFrame)
    #print(bottomFrame)
    #plt.imshow(imgs[topIdx])
    #plt.show()
    #plt.imshow(imgs[bottomIdx])
    #plt.show()
    bottomFrame.append(topFrame)
    bottomIdx.append(topIdx)
    for i, frame in enumerate(bottomFrame):
        plt.imshow(imgs[bottomIdx[i]])
        plt.show()
        if frames[bottomIdx[i]]['l_knee'][0]+0.05 < frames[bottomIdx[i]]['l_foot'][0]:
            message = 'too far forward'
        if frames[bottomIdx[i]]['r_hip'][2] + 0.06 > frames[bottomIdx[i]]['r_knee'][2]:
            message = 'not going down far enough'
        show_marks(imgs[bottomIdx[i]], title=f'Pose Estimation Results 00:01:15: {message}', filename = "squats" + str(i + 1) + ".png")

def get_feats(frame):
  with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    try:
      left_shoulder = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z)
      right_shoulder = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z)
      right_hip = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z)
      left_hip = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z)
      right_heel = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].z)
      left_heel = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].z) 
      right_feet = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z)
      left_feet = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z)
      left_elbow = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z)
      right_elbow = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z)
      left_wrist = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z)
      right_wrist = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z)
      left_knee = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z)
      right_knee = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z)
      left_foot = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z)
      right_foot = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z)
      nos = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z)
    except:
      #print('wtf')
      return 'idfk'
  return {'l_shoulder':left_shoulder, 'r_shoulder': right_shoulder, 'l_hip' : left_hip, 'r_hip' : right_hip, 'l_heel' : left_heel, 'r_heel' : right_heel, 'l_feet' : left_feet, 'r_feet' : right_feet, 'l_elbow' : left_elbow, 'r_elbow' : right_elbow, 'l_wrist' : left_wrist, 'r_wrist' : right_wrist, 'l_knee' : left_knee, 'r_knee' : right_knee, 'l_foot' : left_foot, 'r_foot' : right_foot, 'nose' : nos}
def curl_ups_input():
    cu = cv2.VideoCapture("output.mp4")

    cui, cuf = [], []
    count = 0

    if cu.isOpened():
        width = int(cu.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cu.get(cv2.CAP_PROP_FRAME_HEIGHT))

        res=(int(width), int(height))

        frame = None
        while True:
            
            try:
                is_success, frame = cu.read()
            except cv2.error:
                continue
            if not is_success:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cui.append(image)
            cuf.append(get_feats(image))
            #plt.imshow(image); plt.show()
            count += 15
            cu.set(cv2.CAP_PROP_POS_FRAMES, count)

    cleaned = [i for i, val in enumerate(cuf) if val != 'idfk']
    capImages = [cui[i] for i in cleaned]
    capFrames = [cuf[i] for i in cleaned]

    cu.release()
    return capImages, capFrames

def curl_ups():
    capImages, capFrames = curl_ups_input()
    comp = []
    for i, info in enumerate(capFrames):
        if i == 0 or i == len(capFrames)-1: continue
        if info['l_shoulder'][2] < capFrames[i+1]['l_shoulder'][2] and info['l_shoulder'][2] < capFrames[i-1]['l_shoulder'][2]:
            comp.append((info['l_shoulder'][2], i))
    comp.sort()
    #to get it working for this jawn
    del comp[11]
    for i in range(13):
        # plt.imshow(capImages[comp[i][1]])
        # plt.show()
        dtls = capFrames[comp[i][1]]
        if dtls['l_knee'][0]+0.05 < dtls['l_wrist'][0]:
            message = 'not going far enough'
            show_marks(capImages[comp[i][1]], title=f'Pose Estimation Results: {message}', filename = "curlups" + str(i + 1) + ".png")