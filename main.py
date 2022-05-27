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
gesture = { 1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',
            8:'8',9:'9',10:'10',11:'11', 12:'12', 13:'13', 14:'14',
            15:'15', 16:'16',17:'17',18:'18',19:'19',20:'20', 21:'21',
            22:'22',23:'23',24:'24',25:'25',26:'26',27:'27',28:'28',29:'29',30:'30'
           }
# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)

f = open('tmp6.txt', 'w')
f3 = open('tmp4.txt', 'w')

# 손가락 각도가 저장된 제스처 파일
file = np.genfromtxt('tmp5.txt',delimiter=',')
print(file[0])
# angle, label을 데이터로 모으기
angleFile = file[:,:-1]
labelFile = file[:,-1]
angle = angleFile.astype(np.float32)
label = labelFile.astype(np.float32)
print(angle[0])
print(label)
# knn 모델에 k-Nearest로 데이터 학습
knn = cv2.ml.KNearest_create()
knn.train(angle,cv2.ml.ROW_SAMPLE,label)

for i in range(1, 1):
    # cap = cv2.VideoCapture('videos/ml2.mp4')
    cap = cv2.VideoCapture('data/Front/NIA_SL_SEN00%02d_REAL01_F.mp4'%i)
    mor = open('morpheme/Front/NIA_SL_SEN00%02d_REAL01_F_morpheme.json'%i, 'r', encoding='UTF8')
    jsonStr = json.load(mor)
    start = jsonStr["data"][0]['start']
    if len(jsonStr["data"]) > 1:
        end = jsonStr["data"][1]['end']
    else:
        end = jsonStr["data"][0]['end']
    print(start, end)
    startTime = time.time()
    prev_index = 0
    sentence = ''
    recognizeDelay = 1
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(i)
    for tmp in range(0, length):
    # while length:
        ret, img = cap.read()
        if(img is None):
            print('passed')
            break
        # if(tmp < start*30 - 10):
        #     continue
        # elif(tmp > end*30 + 10):
        #     continue
        else:
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
                    for num in angle:
                        num = round(num, 6)
                        f.write(str(num))
                        f.write(',')
                    f.write("%02d.000000"%i) #라벨을 가변 필드로 바꿔야 함
                    f.write('\n')
                    data = np.array([angle],dtype=np.float32)

                    ### sentence를 저장(bbbbcbbbbdbb)
                    ### sentence에 저장된 값 중 가장 많은 빈도로 나온 값을 출력
                    ret, results, neighbours, dist = knn.findNearest(data,3)
                    index = int(results[0][0])
                    if index in gesture.keys():
                        if index != prev_index:
                            startTime = time.time()
                            prev_index = index
                        else:
                            if time.time() - startTime > 1: 
                                print('called')
                                sentence = gesture[index]
                                startTime = time.time()
                    #######################################################
                    #######################################################

                        cv2.putText(img, gesture[index]. upper(),(int(res. landmark[0].x * img.shape[1] - 10),
                                    int(res.landmark[0].y * img.shape[0] + 40)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3)
                    mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS)
            cv2.putText(img, sentence, (20,440),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),3)

            cv2.imshow('HandTracking', img)
            cv2.waitKey(1)
    f3.write(str(sentence))
    f3.write('\n')
f.close()
f3.close()