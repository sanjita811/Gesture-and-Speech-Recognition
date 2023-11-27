import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from backbone import *
import gymnasium as gym
import os,sys


class GestureRecognition:
    def __init__(self,queue, model_path='97p-model'):
        self.model = self.load_model(model_path)
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.actions = np.array(['left_swipe', 'right_swipe', 'stop', 'thumbs_down', 'thumbs_up'])
        self.label_map = {label: num for num, label in enumerate(self.actions)}
        self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (200, 117, 245), (70, 117, 15)]
        self.threshold = 0.7
        self.sequence = []
        self.sentence = []
        self.action_curr = ''
        self.action_prev = ''
        self.command=None
        self.queue = queue

    def load_model(self, model_path):
        model = get_model()
        model.load_weights(model_path)
        return model

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])

    def prob_viz(self, res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return output_frame
    
    def is_no_action(self,sequence):
        i = 0
        x = True
        while (x == True) and (i < 30):
            hands = list(sequence[i])[1536:]
            x = all(val == 0 for val in hands)
            i += 1
        if x == True:
            return 1
        else:
            return 0
        
    def perform_action(self, action_prev, action_curr):
        actions = {"thumbs_down": 0, "right_swipe": 2, "left_swipe": 3, "thumbs_up": 1,"stop":None}
        for i in actions:
            if action_curr == i and action_prev != i:
                if i!="stop":
                    action=i
                    self.command=actions[i]
                    self.queue.put(self.command)
                    break


    def run_gesture_recognition(self):
        cap = cv2.VideoCapture(0)

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                image, results = self.mediapipe_detection(frame, holistic)
                self.draw_styled_landmarks(image, results)
                keypoints = self.extract_keypoints(results)

                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]
                if len(self.sequence) == 30:
                    no_action = self.is_no_action(self.sequence)
                    if no_action == 1:
                        prediction = 'no_action'
                        res = [0.0, 0.0, 0.0, 0.0, 0.0]
                        prob = 0.99
                    else:
                        with open(os.devnull, 'w') as fnull:
                            sys.stdout = fnull
                            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                            sys.stdout = sys.__stdout__ 
                        prob = res[np.argmax(res)]
                        prediction = self.actions[np.argmax(res)]

                    
                    if prediction!='no_action' and prob > self.threshold: 
                        self.action_prev = self.action_curr
                        self.action_curr = self.actions[np.argmax(res)]
                        self.perform_action(self.action_prev, self.action_curr)
                        
                        if len(self.sentence) > 0: 
                            if self.actions[np.argmax(res)] != self.sentence[-1]:
                                self.sentence.append(self.action_curr)
                        else:
                            self.sentence.append(self.action_curr)

                    if len(self.sentence) > 5: 
                        self.sentence = self.sentence[-5:]
                        
                    image = self.prob_viz(res, self.actions, image, self.colors)
                    
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(self.sentence), (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.namedWindow("OpenCV Feed",cv2.WINDOW_NORMAL)
                cv2.imshow('OpenCV Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        
                





