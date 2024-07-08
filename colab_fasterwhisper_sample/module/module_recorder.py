import numpy as np
import sounddevice as sd

import os
import configparser
# ファイルの存在チェック用モジュール
import errno

class Recorderconfig:
    def __init__(self, config_ini_path = './configs/config.ini'):
        # iniファイルの読み込み
        self.config_ini = configparser.ConfigParser()
        
        # 指定したiniファイルが存在しない場合、エラー発生
        if not os.path.exists(config_ini_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)
        
        self.config_ini.read(config_ini_path, encoding='utf-8')
        Recorder_items = self.config_ini.items('Recorder')
        self.Recorder_config_dict = dict(Recorder_items)

class Recorder:
    def __init__(self, config_ini_path = './configs/config.ini'):
            
            Recorder_config = Recorderconfig(config_ini_path = config_ini_path)
            config_dict = Recorder_config.Recorder_config_dict
            
            self.fs = int(config_dict["fs"])
            self.silence_threshold = float(config_dict["silence_threshold"])
            self.min_duration = float(config_dict["min_duration"])
            self.amplitude_threshold = float(config_dict["amplitude_threshold"])
            self.start_threshold = float(config_dict["start_threshold"])

    def speech2audio(self):
        record_Flag = False

        non_recorded_data = []
        recorded_audio = []
        silent_time = 0
        input_time = 0
        start_threshold = 0.3
        all_time = 0
        
        with sd.InputStream(samplerate=self.fs, channels=1) as stream:
            while True:
                data, overflowed = stream.read(int(self.fs * self.min_duration))
                all_time += 1
                if all_time == 10:
                    print("stand by ready OK")
                elif all_time >=10:
                    if np.max(np.abs(data) > self.amplitude_threshold) and not record_Flag:
                        input_time += self.min_duration
                        if input_time >= start_threshold:
                            record_Flag = True
                            print("recording...")
                            recorded_audio=non_recorded_data[int(-1*start_threshold*10)-2:]  

                    else:
                        input_time = 0

                    if overflowed:
                        print("Overflow occurred. Some samples might have been lost.")
                    if record_Flag:
                        recorded_audio.append(data)

                    else:
                        non_recorded_data.append(data)

                    if np.all(np.abs(data) < self.amplitude_threshold):
                        silent_time += self.min_duration
                        if (silent_time >= self.silence_threshold) and record_Flag:
                            print("finished")
                            record_Flag = False
                            break
                    else:
                        silent_time = 0

        audio_data = np.concatenate(recorded_audio, axis=0)

        return audio_data