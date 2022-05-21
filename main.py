import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time

# 손 2개 인식
max_num_hands = 2

# 총 27개의 제스처가 저장되어있는 상태 & 총 11개의 제스처를 인식할 수 있는 데이터 있음
gesture = { 0:'a',1:'b',2:'c',3:'0',4:'e',5:'f',6:'g',7:'h',
            8:'i',9:'3',10:'k',11:'1', 12:'m', 13:'n', 14:'0',
            15:'p', 16:'q',17:'r',18:'5',19:'t',20:'u', 21:'v',
            22:'w',23:'x',24:'y',25:'2',26: 'spacing',27:'clear'
           }

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)

f = open('test.txt', 'w')

# 손가락 각도가 저장된 제스처 파일
file = np.genfromtxt('dataSet.txt',delimiter=',')
# angle, label을 데이터로 모으기
angleFile = file[:,:-1]
labelFile = file[:,-1]
angle = angleFile.astype(np.float32)
label = labelFile.astype( np.float32)
# knn 모델에 k-Nearest로 데이터 학습
knn = cv2.ml.KNearest_create()
knn.train(angle,cv2.ml.ROW_SAMPLE,label)

cap = cv2.VideoCapture(0)

startTime = time.time()
prev_index = 0
sentence = ''
recognizeDelay = 1

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21,3))
            for j, lm in enumerate(res. landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0,17,18,19],:]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20],:]

            v = v2 - v1
            v = v/np.linalg.norm(v, axis=1)[:,np.newaxis]
            comparev1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17],:]
            comparev2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],:]
            angle = np.arccos(np.einsum('nt,nt->n', comparev1, comparev2))

            angle = np.degrees(angle)
            if keyboard. is_pressed('a'):
                for num in angle:
                    num = round(num, 6)
                    f.write(str(num))
                    f.write(',')
                f.write("27.000000")
                f.write('\n')
                print("next")
            data = np.array([angle],dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data,3)
            index = int(results[0][0])
            if index in gesture.keys():
                if index != prev_index:
                    startTime = time.time()
                    prev_index = index
                else:
                    if time.time() - startTime > recognizeDelay:
                        if index == 26:
                            sentence += ''
                        elif index == 27:
                            sentence = ''
                        else:
                            sentence += gesture[index]
                        startTime = time.time()

                cv2.putText(img, gesture[index]. upper(),(int(res. Landmark[0].x * img.shape[1] - 10),
                            int(res.landmark[0].y * img.shape[0] + 40)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3)

            mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS)
    cv2.putText(img, sentence, (20,440),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),3)

    cv2.imshow('HandTracking', img)
    cv2.waitKey(1)
    if keyboard. is_pressed('b'):
        break

f.close();

