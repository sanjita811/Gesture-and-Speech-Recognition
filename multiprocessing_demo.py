import multiprocessing
import queue
from audio_with_gym import Transcriber
from gesture_with_gym import GestureRecognition
from robot import CustomEnv
import os
import pyttsx3
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def handle_message_queue(audio_queue, gesture_queue, action_queue):
    global env
    while True:
        try:
            try:
                audio_command = audio_queue.get(timeout=0.5)
                action_queue.put(audio_command)
            except queue.Empty:
                pass
            try:
                gesture_command = gesture_queue.get(timeout=0.5)
                action_queue.put(gesture_command)
            except queue.Empty:
                pass
            while not action_queue.empty():
                command = action_queue.get()
                action=next((key for key, value in commands.items() if value == command), None)
                print('\nAction Performed by Robot: ',action,'\n')
                observation, reward, done, info,_ = env.step(command)
                if done:
                    env.reset()
                
        except KeyboardInterrupt:
            break
        finally:
            cv2.destroyAllWindows()

def run_audio_process(queue):
    transcriber = Transcriber(queue=queue,model="base", non_english=False, energy_threshold=1000, record_timeout=2, phrase_timeout=3, default_microphone=None)
    transcriber.run()

def run_gesture_process(queue):
    gesture_recognition = GestureRecognition(queue)
    gesture_recognition.run_gesture_recognition()

if __name__ == '__main__':
    commands = {"right": 2, "left": 3, "front": 1, "back": 0, "pick":4,"drop":5}
    env = CustomEnv()
    env.reset()
    multiprocessing.freeze_support()  
    audio_queue = multiprocessing.Queue()
    gesture_queue = multiprocessing.Queue()
    action_queue = multiprocessing.Queue()
    audio_process = multiprocessing.Process(target=run_audio_process, args=(audio_queue,))
    audio_process.start()
    gesture_process = multiprocessing.Process(target=run_gesture_process, args=(gesture_queue,))
    gesture_process.start()
    handle_message_queue(audio_queue, gesture_queue, action_queue)
    audio_process.join()
    gesture_process.join()
    audio_process.terminate()
    gesture_process.terminate()

