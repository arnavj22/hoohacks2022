import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import math
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
def vid_to_frames():
    cap = cv2.VideoCapture("simple_front_view.mp4")

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
    return frames
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image, title=None):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.title(title)
  plt.axis(False)
  plt.imshow(img); plt.show()

def show_marks(image, title=None):
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
    resize_and_show(annotated_image, title=title)
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
    frames = vid_to_frames()
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
        show_marks(imgs[switchIdx[i]], title=f'Pose Estimation Results 00:01:15: {message}')