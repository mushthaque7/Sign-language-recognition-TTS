import os
import time
import numpy as np
import cv2 
from pathlib import Path


DATA_DIR = './data/twohand'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 2
dataset_size = 200

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        
        cv2.putText(frame, 'Class: {}'.format(j), (frame.shape[1] - 150, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            done = True
            break
    
    if done:
        time.sleep(2)

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if counter ==dataset_size-1:
            cv2.putText(frame, 'Class: {} Data Collected'.format(j), (int(frame.shape[1] / 2) - 225, int(frame.shape[0] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.waitKey(2500) 
        else:
            cv2.putText(frame, 'Capturing...', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, 'Frame: {}'.format(counter+1), (frame.shape[1] - 200, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1
          
    
cap.release()
cv2.destroyAllWindows()
