import io
import os
import speech_recognition as sr
import whisper
import torch
import re
from queue import Queue
from tempfile import NamedTemporaryFile
import time
from datetime import datetime, timedelta

class Transcriber:
    def __init__(self, queue,model="base", non_english=False, energy_threshold=1000, record_timeout=3, phrase_timeout=3, default_microphone=None):
        self.result = ""
        self.transcription = [""]
        self.model = model if non_english else model + ".en"
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout
        self.energy_threshold=energy_threshold
        self.data_queue = Queue()
        self.phrase_time = None
        self.last_sample = bytes()
        self.temp_file = NamedTemporaryFile().name
        self.new_transcription=''
        self.flag=1
        self.instructions=[]
        self.setup_microphone(default_microphone)
        self.load_whisper_model()
        self.queue=queue
        self.command=None
        self.hello_flag=False
        self.bye_flag=False
        self.commands = {"right": 2, "left": 3, "front": 1, "back": 0, "pick":4,"drop":5}

    def setup_microphone(self, default_microphone):
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.energy_threshold

        if default_microphone:
            self.select_microphone(default_microphone)
        else:
            self.select_default_microphone()

    def select_microphone(self, microphone_name):
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if microphone_name in name:
                source = sr.Microphone(sample_rate=16000, device_index=index)
                break
        self.source = source

    def select_default_microphone(self):
        self.source = sr.Microphone(sample_rate=16000)

    def load_whisper_model(self):
        self.audio_model = whisper.load_model(self.model)

    def extract_text_between_keywords(self, paragraph, start_keyword="hello", end_keyword="bye"):
        paragraph = paragraph.lower()
        paragraph = re.sub(r'[^a-zA-Z0-9]', ' ', paragraph)
        paragraph = re.sub(r'\s+', ' ', paragraph)
        ind=paragraph.find("hello")
        bye_ind= paragraph.find("bye")
        if bye_ind!=-1 : 
            self.bye_flag=True
            self.hello_flag=False
        if ind!=-1: 
            self.hello_flag=True
            self.bye_flag=False
            self.new_transcription=paragraph[ind+6:]
        elif self.bye_flag==False: 
            self.hello_flag=True
            self.new_transcription=paragraph
        
    def get_command(self):
        if self.bye_flag==True:
            self.new_transcription=''
            self.command=None
            self.transcription=[]

        elif self.hello_flag==True:
            words = re.findall(r'\b\w+\b', self.new_transcription.lower())
            for word in words:
                for command in self.commands:
                    pattern = re.compile(r'\b' + re.escape(command) + r'\b', re.IGNORECASE)
                    if pattern.search(word):
                        action = command
                        self.command = self.commands[command]
                        self.queue.put(self.command)
            self.new_transcription=''
            self.transcription=[]
            self.command=None

    def listen_and_transcribe(self):
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        def record_callback(_, audio: sr.AudioData) -> None:
            data = audio.get_raw_data()
            self.data_queue.put(data)

        self.recorder.listen_in_background(self.source, record_callback, phrase_time_limit=self.record_timeout)

    def run(self):
        self.listen_and_transcribe()
        while True:
            try:
                now = datetime.utcnow()
                if not self.data_queue.empty():
                    phrase_complete = False
                    if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                        self.last_sample = bytes()
                        phrase_complete = True
                    self.phrase_time = now
                    while not self.data_queue.empty():
                        data = self.data_queue.get()
                        self.last_sample += data
                    audio_data = sr.AudioData(self.last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())

                    with open(self.temp_file, 'w+b') as f:
                        f.write(wav_data.read())
                    result = self.audio_model.transcribe(self.temp_file, fp16=torch.cuda.is_available())
                    text = result['text'].strip()
                    if phrase_complete:
                        self.transcription.append(text)
                    else:
                        try:
                            self.transcription[-1] = text
                        except:
                            self.transcription.append(text)
                    os.system('cls' if os.name=='nt' else 'clear')
                    self.extract_text_between_keywords(' '.join(self.transcription).lower())
                    self.get_command()
                    print('', end='', flush=True)
            except KeyboardInterrupt:
                break



