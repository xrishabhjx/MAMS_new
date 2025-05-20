import os
import time
import threading
import wave
import pyaudio
from datetime import datetime

# Settings
AUDIO_DIR = "assets/audio_data"
RECORD_SECONDS = 5  # Duration of each audio snippet
INTERVAL_MINUTES = 5  # Interval between recordings
SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16

# Ensure audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)

def record_audio(file_path):
    print(f"Recording audio: {file_path}")
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=SAMPLE_RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    for _ in range(0, int(SAMPLE_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

def analyze_sentiment(audio_path):
    # TODO: Call your sentiment model here
    print(f"Analyzing sentiment for: {audio_path}")
    # e.g., result = my_model.predict(audio_path)
    # print(result)

def capture_audio_periodically():
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_file = os.path.join(AUDIO_DIR, f"speech_{timestamp}.wav")
        record_audio(audio_file)
        analyze_sentiment(audio_file)
        time.sleep(INTERVAL_MINUTES * 60)

def start_audio_capture():
    print("Audio capture service started. Recording every 5 minutes...")
    thread = threading.Thread(target=capture_audio_periodically)
    thread.daemon = True
    thread.start()

if __name__ == "__main__":
    start_audio_capture()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Audio capture terminated.")