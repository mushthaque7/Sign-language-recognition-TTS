import threading
import time
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import streamlit as st


model_dict1 = pickle.load(open('./model1.p', 'rb'))
model_dict2 = pickle.load(open('./model2.p', 'rb'))
# model_dict3 = pickle.load(open('./model3.p', 'rb'))
model1 = model_dict1['model1']
model2 = model_dict2['model2']
# model3 = model_dict3['model3']


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2 ,min_detection_confidence=0.3)

labels_dict1 = {0: 'B', 1: 'D', 2: 'K', 3: 'V', 4: '4', 5: '1', 6:'2'}
labels_dict2 = {0: 'Camera', 1: 'Rectangle'}
# labels_dict3 = {0: 'Camera', 1: 'Rectangle'}

speech_engine = pyttsx3.init()

# Initialize Streamlit
st.title("Sign Language Recognition")
st.subheader('This web app is used to recognise sign language gestures and convert it into speech.')
# Create a canvas to display the video feed
canvas = st.image([])


cap = cv2.VideoCapture(0)

# Initialize variables for gesture tracking
current_gesture = None
gesture_start_time = None

# Initialize lock for shared variables
lock = threading.Lock()


def text_to_speech(gesture):
    speech_engine.say(gesture)
    speech_engine.runAndWait()


def gesture_timer():
    global current_gesture, gesture_start_time

    while True:
        with lock:
            if current_gesture is not None and gesture_start_time is not None:
                elapsed_time = time.time() - gesture_start_time
                if elapsed_time >= 2:
                    threading.Thread(target=text_to_speech, args=(current_gesture,)).start()
                    current_gesture = None
                    gesture_start_time = None

        time.sleep(0.1)


# Start gesture timer thread
timer_thread = threading.Thread(target=gesture_timer)
timer_thread.daemon = True
timer_thread.start()



while True:

    data_aux = []
   
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        n = len(results.multi_hand_landmarks)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
                
        if n==1:
            
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction1 = model1.predict([np.asarray(data_aux)])
            # prediction3 = model3.predict([np.asarray(data_aux)])
            predicted_character1 = labels_dict1[int(prediction1[0])]
            # predicted_character3 = labels_dict3[int(prediction3[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character1, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            # cv2.putText(frame, predicted_character3, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
            #             cv2.LINE_AA)
            with lock:
                if predicted_character1 != current_gesture:
                    current_gesture = predicted_character1
                    gesture_start_time = time.time()
            # with lock:
            #     if predicted_character3 != current_gesture:
            #         current_gesture = predicted_character3
            #         gesture_start_time = time.time()

        else:
             x1 = int(min(x_) * W) - 10
             y1 = int(min(y_) * H) - 10

             x2 = int(max(x_) * W) - 10
             y2 = int(max(y_) * H) - 10

             prediction2 = model2.predict([np.asarray(data_aux)])

             predicted_character2 = labels_dict2[int(prediction2[0])]

             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
             cv2.putText(frame, predicted_character2, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
             
             with lock:
                if predicted_character2 != current_gesture:
                    current_gesture = predicted_character2
                    gesture_start_time = time.time()

    # Display the frame in the Streamlit canvas
    canvas.image(frame, channels="BGR")

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
