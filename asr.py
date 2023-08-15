#IMPORTS
import vosk #install 
import wave
import subprocess
import os
import yaml #install 
import torch
from torch import package

def convert_video_to_wav(video_path, output_audio_path):
    # Use subprocess to call ffmpeg to extract audio
    #instal ffmpeg and add it to path 
    subprocess.call(['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_audio_path])

def transcribe_audio(audio_path):
    model = vosk.Model(r"C:\Users\LENOVO\Downloads\archive\vosk-model-en-us-0.22") # 4 models of different sizes available here and can be downloaded : https://alphacephei.com/vosk/models

    sample_rate = 16000

    wf = wave.open(audio_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        return

    recognizer = vosk.KaldiRecognizer(model, sample_rate)
    transcript = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            result_dict = eval(result)
            transcript += result_dict["text"]

    result = recognizer.FinalResult()
    result_dict = eval(result)
    transcript += result_dict["text"]
    return transcript

#TEXT enhancement - repunctuation https://github.com/snakers4/silero-models
torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml','latest_silero_models.yml',progress=False)

with open('latest_silero_models.yml', 'r', encoding='utf-8') as yaml_file:
    models = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    
model_conf = models.get('te_models').get('latest')

model_url = model_conf.get('package')

model_dir = "downloaded_model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, os.path.basename(model_url))

if not os.path.isfile(model_path):
    torch.hub.download_url_to_file(model_url, model_path, progress=True)

imp = package.PackageImporter(model_path)
model = imp.load_pickle("te_model", "model")
example_texts = model.examples

def apply_te(text, lan='en'):
    return model.enhance_text(text, lan)

#IMPLEMENTING
VP = r""
AP = r""
convert_video_to_wav(video_path= VP, output_audio_path=AP)
full_transcript = transcribe_audio(audio_path=AP)
print(full_transcript)
formatted_transcript = apply_te(full_transcript, lan='en')
print(formatted_transcript)
