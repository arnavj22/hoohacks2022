import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import math
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.putText(img,title, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),2)
    cv2.imwrite("static/graphs/" + filename, img)
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
            message = 'Student is not going down far enough'
        else:
            message = 'Student is having good form'
        show_marks(imgs[switchIdx[i]], title=message,  filename = "pushups" + str(i + 1) + ".png")
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
        # plt.imshow(imgs[bottomIdx[i]])
        # plt.show()
        message = ""
        if frames[bottomIdx[i]]['l_knee'][0]+0.05 < frames[bottomIdx[i]]['l_foot'][0]:
            message = 'Student is too far forward'
            show_marks(imgs[bottomIdx[i]], title=message, filename = "squats" + str(i + 1) + ".png")
        if frames[bottomIdx[i]]['r_hip'][2] + 0.06 > frames[bottomIdx[i]]['r_knee'][2]:
            message = 'Student is not going down far enough'
            show_marks(imgs[bottomIdx[i]], title=message, filename = "squats" + str(i + 1) + ".png")

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
            message = 'Student is not going far enough'
            show_marks(capImages[comp[i][1]], title=message, filename = "curlups" + str(i + 1) + ".png")
def read_biceps():
    bc = cv2.VideoCapture("output.mp4")

    bci, bcf = [], []
    count = 0

    if bc.isOpened():
        width = int(bc.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(bc.get(cv2.CAP_PROP_FRAME_HEIGHT))

        res=(int(width), int(height))

        frame = None
        while True:
            
            try:
                is_success, frame = bc.read()
            except cv2.error:
                continue
            if not is_success:
                break

            image = cv2.cvtColor(frame[:, 160:480], cv2.COLOR_BGR2RGB)
            bci.append(image)
            bcf.append(get_feats(image))
            # plt.imshow(image); plt.show()
            count += 15
            bc.set(cv2.CAP_PROP_POS_FRAMES, count)

    cleaned = [i for i, val in enumerate(bcf) if val != 'idfk']
    bcImages = [bci[i] for i in cleaned]
    bcFrames = [bcf[i] for i in cleaned]
    bc.release()
    return bcImages, bcFrames
def distance(x1, y1, z1, x2, y2, z2):
  return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1)
def bicepcurls():
    bcImages, bcFrames = read_biceps()
    rep1 = [0, 3, 6]
    rep2 = [6, 9, 12]
    rep3 = [12, 15, 21]
    rep4 = [21, 24, 27]
    rep5 = [27, 29, 28]
    rep6 = [0, 1, 6]
    reps = [rep1, rep2, rep3, rep4, rep5, rep6]
    for i, rep in enumerate(reps):
        initial = bcFrames[rep[0]]['l_elbow']
        after = bcFrames[rep[1]]['l_elbow']
        # fig = plt.figure(figsize=(10, 7))
        # fig.add_subplot(1, 3, 1)
        # plt.imshow(bcImages[rep[0]])
        # fig.add_subplot(1, 3, 2)
        # plt.imshow(bcImages[rep[1]])
        # fig.add_subplot(1, 3, 3)
        # plt.imshow(bcImages[rep[2]])
        # plt.show()
        hcon = cv2.hconcat([bcImages[rep[0]], bcImages[rep[1]], bcImages[rep[2]]])
        if distance(initial[0], initial[1], 0, after[0], after[1], 0) > 0.05:
            message = 'Student is swinging elbows'
        start = bcFrames[rep[0]]['l_wrist']
        end = bcFrames[rep[2]]['l_wrist']
        if abs(end[0] - start[0]) > 0.15:
            message = 'Student is not completely straightening arm after completing rep'
            show_marks(hcon, title=message, filename = "bicepcurls" + str(i + 1) + ".png")
        if abs(bcFrames[rep[0]]['l_wrist'][2] - bcFrames[rep[1]]['l_wrist'][2]) < 1:
            message = 'Student is not going high enough on the way up'
            show_marks(hcon, title=message, filename = "bicepcurls" + str(i + 1) + ".png")
def planked():
    frames, imgs = vid_to_frames()
    topFrame, topIdx = frames[0], 0
    switchFrame, switchIdx = [], []
    used = set()
    for i, frame in enumerate(frames):
        diff = frames[i]["r_shoulder"][0] - frames[i]["r_elbow"][0]
        if diff < -0.012 and 'Student should move shoulders back' not in used:
            message = 'Student should move shoulders back'
        if diff > 0.012 and 'Student should move shoulders forward' not in used:
            message = 'Student should move shoulders forward'
        hip_to_shoulder = frames[i]["r_hip"][0] - frames[i]["r_shoulder"][0]
        if hip_to_shoulder < -0.01 and 'Student is raising his/her back too high' not in used:
            message = 'Student is raising his/her back too high'
        if 0.4 > hip_to_shoulder > 0.2 and 'Student is slightly lowering his/her back' not in used:
            message = 'Student is slightly lowering his/her back'
        if hip_to_shoulder > 0.4 and 'Student is lowering his/her back too much' not in used:
            message = 'Student is lowering his/her back too much'
        if message not in used:
            used.add(message)
            show_marks(imgs[i], title=message, filename = "plank" + str(i + 1) + ".png")