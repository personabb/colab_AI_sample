import sounddevice as sd
from module.module_whisper import FasterWhisperModel
from module.module_recorder import Recorder

def main():

    recorder = Recorder()
    fasterWhispermodel = FasterWhisperModel()
    while True:
        audio_data = recorder.speech2audio()
        text = fasterWhispermodel.audio2text(audio_data)
        print(text)

if __name__ == "__main__":
    main()