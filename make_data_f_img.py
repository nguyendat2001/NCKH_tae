import os
import pandas as pd
# from scipy.misc import imread
from imageio import imread
import math
import numpy as np
import cv2
import keras
import mediapipe as mp
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential

# Đọc ảnh từ webcam
# cap = cv2.VideoCapture('./data_train/dt_1/dt_tae_1.mp4')
cap = cv2.VideoCapture('./data_train/dt_chao/dt_chao_1.mp4')

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "dt_chao"
no_of_frames = 600


def load_data(input_size = (64,64), data_path =  'path'):
    
    pixels = []
    labels = []
    # Loop qua các thư mục trong thư mục Images
    for dir in os.listdir(data_path):
        if dir == '.DS_Store':
            continue

        # Đọc file csv để lấy thông tin về ảnh
        class_dir = os.path.join(data_path, dir)
        info_file = pd.read_csv(os.path.join(class_dir, "GT-" + dir + '.csv'), sep=';')

        # Lăp trong file
        for row in info_file.iterrows():
            # Đọc ảnh
            pixel = imread(os.path.join(class_dir, row[1].Filename))
            # Trích phần ROI theo thông tin trong file csv
            pixel = pixel[row[1]['Roi.X1']:row[1]['Roi.X2'], row[1]['Roi.Y1']:row[1]['Roi.Y2'], :]
            # Resize về kích cỡ chuẩn
            img = cv2.resize(pixel, input_size)

            # Thêm vào list dữ liệu
            pixels.append(img)

            # Thêm nhãn cho ảnh
            labels.append(row[1].ClassId)

    return pixels, labels


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
cap.release()
cv2.destroyAllWindows()