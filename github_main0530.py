#!pip install mediapipe

#from google.colab.patches import cv2_imshow

import cv2
import mediapipe as mp
import numpy as np
import time
import json

# 손 2개 인식
max_num_hands = 2

# 제스처 수정 필요
# gesture = { 1:'left',2:'left',3:'right',4:'right',5:'여기',6:'여기',7:'저기',
#             8:'저기',9:'운전 천천히',10:'운전 천천히',11:'운전 빨리', 12:'빨리 가주세요', 13:'시간이 급해요', 14:'시간이 급해요',
#             15:'급해요', 16:'급해요?',17:'약속에 늦었어요',18:'약속에 늦었어요',19:'나',20:'나?', 21:'당신',
#             22:'당신',23:'그 남자 맞다',24:'그 남자 맞다',25:'잘못 말하다',26:'잘못 말해주다',27:'위험',28:'위험?',29:'항상',30:'항상?'
#            }
#gesture = { 1:'left',2:'left',3:'right',4:'right', 5:"here" }
gesture = { 1:'왼쪽1',2:'왼쪽2',3:'오른쪽1',4:'오른쪽2', 5:"여기" }


# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)

# 손가락 각도가 저장된 제스처 파일
file = np.genfromtxt('/content/data10.txt',delimiter=',')

# angle, label을 데이터로 모으기 
angleFile = file[:,:-1]
labelFile = file[:,-1]
angle = angleFile.astype(np.float32)
label = labelFile.astype(np.float32)

# knn 모델에 k-Nearest로 데이터 학습
knn = cv2.ml.KNearest_create()
knn.train(angle,cv2.ml.ROW_SAMPLE,label)

sentence = ''

def most_frequent(data):
    return max(data, key=data.count)

startTime = time.time()
prev_index = 0
sentence = ''
recognizeDelay = 1

cap = cv2.VideoCapture("/content/tttesttt.mov")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#print(length)

fps = cap.get(cv2.CAP_PROP_FPS)

tmp_array = []

while length:
    ret, img = cap.read()
    if not ret:
        break
    imgRGB = cv2.cvtColor( img,cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21,3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0,17,18,19],:]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20],:]

            v = v2 - v1
            v = v/np.linalg.norm(v, axis=1)[:,np.newaxis]
            comparev1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17],:]
            comparev2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],:]
            angle = np.arccos(np.einsum('nt,nt->n', comparev1, comparev2))

            angle = np.degrees(angle)

            data = np.array([angle],dtype=np. float32)
            ret, results, neighbours, dist = knn.findNearest(data,3)
            index = int(results[0][0])

            if index in gesture.keys():
                if index != prev_index:
                    startTime = time.time()
                    prev_index = index
                else:
                  if time.time() - startTime > 1: #현재시각 - 시작시간이므로, 실행시간>1일 
                    tmp_array.append(gesture[index])
                    startTime = time.time()

    if not not tmp_array:
      sentence = most_frequent(tmp_array)
      #print("sentence : %s" %sentence)


print(tmp_array)
