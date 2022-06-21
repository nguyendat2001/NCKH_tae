import cv2
import mediapipe as mp
import pandas as pd
import os
# Đọc ảnh từ webcam
# cap = cv2.VideoCapture('./data_train/dt_1/dt_tae_1.mp4')
cap = cv2.VideoCapture('./data_train/dt_6/tae_6.mp4')

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "dt_6"
no_of_frames = 900

labels = ["pose_1","pose_2","pose_3","pose_4","pose_5",
          "pose_6","pose_7","pose_8","pose_9","pose_10",
          "pose_11","pose_12","pose_13","pose_14","pose_15",
          "pose_16","pose_17","pose_18"]

def make_landmark_timestep(results):
    print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    # Vẽ các đường nối
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img


while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    if ret:
        # Nhận diện pose
        frame = cv2.resize(frame,(1080,720)) 
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            # Ghi nhận thông số khung xương
            lm = make_landmark_timestep(results)
            
            lm_list.append(lm)
            # Vẽ khung xương lên ảnh
            frame = draw_landmark_on_image(mpDraw, results, frame)

        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

# Write vào file csv
df  = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
# df.to_csv(label + ".txt" , mode='a', header=not os.path.exists(label + ".txt"))
cap.release()
cv2.destroyAllWindows()