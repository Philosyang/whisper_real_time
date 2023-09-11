import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)  
    args = parser.parse_args()
    
    phrase_time = None
    last_sample = bytes()
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False
    
    source = sr.Microphone(sample_rate=16000)
        
    model = args.model
    if args.model != "large":
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']
    
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print(torch.cuda.memory_allocated())
    print(torch.cuda.get_device_properties(0).total_memory)
    print("Model loaded. Begin now:\n")

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                phrase_time = now

                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                for line in transcription:
                    if not len(line):
                        continue
                    print(line)
                    with open('out', 'a', encoding='utf-8') as f:
                        f.write(line + '\n')

                print('', end='', flush=True)

                sleep(0.25)
        except KeyboardInterrupt:
            break

    # print("\n\nTranscription:")
    # for line in transcription:
    #     print(line)


if __name__ == "__main__":
    main()