import os
import numpy as np
from pathlib import Path
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from pathlib import Path
from style_bert_vits2.tts_model import TTSModel
import queue
from style_bert_vits2.logging import logger
from pathlib import Path
import torch

import glob

import configparser
# ファイルの存在チェック用モジュール
import errno

class SBV2config:
    def __init__(self, config_ini_path = './configs/config.ini'):
        # iniファイルの読み込み
        self.config_ini = configparser.ConfigParser()
        
        # 指定したiniファイルが存在しない場合、エラー発生
        if not os.path.exists(config_ini_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)
        
        self.config_ini.read(config_ini_path, encoding='utf-8')
        SBV2_items = self.config_ini.items('SBV2')
        self.SBV2_config_dict = dict(SBV2_items)

        
    
    


class SBV2:
    def __init__(self,device = None,voice = None,config_ini_path = './configs/config.ini'):
        logger.remove()
        sbv2_config = SBV2config(config_ini_path = config_ini_path)
        config_dict = sbv2_config.SBV2_config_dict
        

        if device is not None:
            self.DEVICE = device
        else:
            device = config_dict["device"]

            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            if device != "auto":
                self.DEVICE = device
    

        bert_models.load_model(Languages.JP, config_dict["bert_models_model"])
        bert_models.load_tokenizer(Languages.JP, config_dict["bert_models_tokenizer"])

        if voice is not None:
            self.voice = voice
        else:
            self.voice = config_dict["ai_agent"]

        assets_root = config_dict["agent_dir"] + "/" + self.voice

        style_file = glob.glob(f'{assets_root}/**/*.npy',recursive=True)[0]
        config_file = glob.glob(f'{assets_root}/**/*.json',recursive=True)[0]
        model_file = glob.glob(f'{assets_root}/**/*.safetensors',recursive=True)[0]

        print(style_file)
        print(config_file)
        print(model_file)

        
        self.model_TTS = TTSModel(
            model_path=model_file,
            config_path=config_file,
            style_vec_path=style_file,
            device=self.DEVICE
        )

    def call_TTS(self,message):
        sr, audio = self.model_TTS.infer(text=message)

        return sr, audio

    
