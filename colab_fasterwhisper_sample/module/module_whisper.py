from faster_whisper import WhisperModel
import numpy as np
import torch

import os
import configparser
# ファイルの存在チェック用モジュール
import errno

class FasterWhisperconfig:
    def __init__(self, config_ini_path = './configs/config.ini'):
        # iniファイルの読み込み
        self.config_ini = configparser.ConfigParser()
        
        # 指定したiniファイルが存在しない場合、エラー発生
        if not os.path.exists(config_ini_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)
        
        self.config_ini.read(config_ini_path, encoding='utf-8')
        FasterWhisper_items = self.config_ini.items('FasterWhisper')
        self.FasterWhisper_config_dict = dict(FasterWhisper_items)

class FasterWhisperModel:
    def __init__(self,device = None, config_ini_path = './configs/config.ini'):
        FasterWhisper_config = FasterWhisperconfig(config_ini_path = config_ini_path)
        config_dict = FasterWhisper_config.FasterWhisper_config_dict

        if device is not None:
            self.DEVICE = device
        else:
            device = config_dict["device"]

            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            if device != "auto":
                self.DEVICE = device
            
        self.BEAM_SIZE = int(config_dict["gpu_beam_size"]) if self.DEVICE == "cuda" else int(config_dict["cpu_beam_size"])
        self.language = config_dict["language"]
        self.COMPUTE_TYPE = config_dict["gpu_compute_type"] if self.DEVICE == "cuda" else config_dict["cpu_compute_type"]
        self.MODEL_TYPE = config_dict["gpu_model_type"] if self.DEVICE == "cuda" else config_dict["cpu_model_type"]
        self.kotoba_chunk_length = int(config_dict["chunk_length"])
        self.kotoba_condition_on_previous_text = config_dict["condition_on_previous_text"]
        if self.kotoba_condition_on_previous_text == "True":
            self.kotoba_condition_on_previous_text = True
        else:
            self.kotoba_condition_on_previous_text = False

        if config_dict["use_kotoba"] == "True":
            self.use_kotoba = True
        else:
            self.use_kotoba = False

        if not self.use_kotoba:
            self.model = WhisperModel(self.MODEL_TYPE, device=self.DEVICE, compute_type=self.COMPUTE_TYPE)
        else:
            self.MODEL_TYPE = config_dict["kotoba_model_type"]
            #self.model = WhisperModel(self.MODEL_TYPE, device=self.DEVICE, compute_type=self.cotoba_compute_type)
            self.model = WhisperModel(self.MODEL_TYPE)


    def audio2text(self, data):
        result = ""
        data = data.flatten().astype(np.float32)
        if not self.use_kotoba:
            segments, _ = self.model.transcribe(data, beam_size=self.BEAM_SIZE,language=self.language)
        else:
            segments, _ = self.model.transcribe(data, beam_size=self.BEAM_SIZE,language=self.language, chunk_length=self.kotoba_chunk_length, condition_on_previous_text=self.kotoba_condition_on_previous_text)
        
        for segment in segments:
            result += segment.text
        
        return result
            
    def audioFile2text(self, file_path):
        result = ""
        if not self.use_kotoba:
            segments, _ = self.model.transcribe(file_path, beam_size=self.BEAM_SIZE,language=self.language)
        else:
            segments, _ = self.model.transcribe(file_path, beam_size=self.BEAM_SIZE,language=self.language, chunk_length=self.kotoba_chunk_length, condition_on_previous_text=self.kotoba_condition_on_previous_text)
        
        for segment in segments:
            result += segment.text

        return result