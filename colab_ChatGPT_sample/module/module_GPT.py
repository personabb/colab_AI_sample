from openai import OpenAI
import os
import configparser
# ファイルの存在チェック用モジュール
import errno

class GPTconfig:
    def __init__(self, config_ini_path = './configs/config.ini'):
        # iniファイルの読み込み
        self.config_ini = configparser.ConfigParser()
        
        # 指定したiniファイルが存在しない場合、エラー発生
        if not os.path.exists(config_ini_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)
        
        self.config_ini.read(config_ini_path, encoding='utf-8')
        GPT_items = self.config_ini.items('GPT')
        self.GPT_config_dict = dict(GPT_items)


class GPT:
    def __init__(self, api_key, agent = None, config_ini_path = './configs/config.ini') :
        
        GPT_config = GPTconfig(config_ini_path = config_ini_path)
        config_dict = GPT_config.GPT_config_dict
            
        self.client = OpenAI(api_key=api_key)
        self.simpleGPT_messages = []
        self.model = config_dict["gpt_model"]
        self.temperature = config_dict["temperature"]
        
        SYSTEM_PROMPT_FILE = config_dict["system_prompt_file_path"]
        
        if agent is not None:
            AI_AGENT = agent
        else:
            AI_AGENT = config_dict["ai_agent"]
        AGENT_DIR = config_dict["agent_dir"]
        
        if SYSTEM_PROMPT_FILE == "None":
            SYSTEM_PROMPT_FILE = None

        if SYSTEM_PROMPT_FILE is not None:
            with open(AGENT_DIR+"/"+AI_AGENT+"/Prompt/"+SYSTEM_PROMPT_FILE) as f:
                self.sys_prompt = f.read()
        else:
            self.sys_prompt = None
    

    def simpleGPT(self, user_prompt, temp_sys_prompt = None):
        if temp_sys_prompt is not None:
            self.simpleGPT_messages.append({"role": "system", "content": temp_sys_prompt})
        else:
            self.simpleGPT_messages.append({"role": "system", "content": self.sys_prompt})

        self.simpleGPT_messages.append({"role": "user", "content": user_prompt})
    

        res = self.client.chat.completions.create(
            model=self.model,
            messages = self.simpleGPT_messages,
            temperature=float(self.temperature)
        )
        return res.choices[0].message.content
