from llama_cpp import Llama

import os
import configparser
# ファイルの存在チェック用モジュール
import errno


class LLMconfig:
    def __init__(self, config_ini_path = './configs/config.ini'):
        # iniファイルの読み込み
        self.config_ini = configparser.ConfigParser()
        
        # 指定したiniファイルが存在しない場合、エラー発生
        if not os.path.exists(config_ini_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)
        
        self.config_ini.read(config_ini_path, encoding='utf-8')
        LLM_items = self.config_ini.items('LLM')
        self.LLM_config_dict = dict(LLM_items)

class LLM:
    def __init__(self, config_ini_path = './configs/config.ini') :
        
        LLM_config = LLMconfig(config_ini_path = config_ini_path)
        config_dict = LLM_config.LLM_config_dict
        
        self.simpleLLM_messages=[]
        self.model = config_dict["llm_model"]
        self.temperature = float(config_dict["temperature"])
        self.max_tokens = int(config_dict["max_tokens"])
        self.n_gpu_layers_num = int(config_dict["n_gpu_layers_num"])
        self.n_ctx = int(config_dict["n_ctx"])
        
        
        SYSTEM_FIRST_PROMPT_FILE = config_dict["system_prompt_first_file_path"]
        SYSTEM_END_PROMPT_FILE = config_dict["system_prompt_end_file_path"]
        SYSTEM_PROMPT_FILE = config_dict["system_prompt_file_path"]

        AI_AGENT = config_dict["ai_agent"]
        AGENT_DIR = config_dict["agent_dir"]
        
        if SYSTEM_PROMPT_FILE == "None":
            SYSTEM_PROMPT_FILE = None
        
        with open(AGENT_DIR+"/"+AI_AGENT+"/Prompt/"+SYSTEM_PROMPT_FILE) as f:
            self.sys_prompt = f.read()
    

        with open(SYSTEM_FIRST_PROMPT_FILE) as f:
            self.sys_first_prompt = f.read()
        
        with open(SYSTEM_END_PROMPT_FILE) as f:
            self.sys_end_prompt = f.read()
            
        #from huggingface_hub import hf_hub_download
        #llm_C = Llama(n_gpu_layers = self.n_gpu_layers_num,n_ctx = self.n_ctx)
        self.llm = Llama(model_path=self.model,n_gpu_layers = self.n_gpu_layers_num,n_ctx = self.n_ctx)
        """self.llm = Llama.from_pretrained(
                        repo_id="TheBloke/Swallow-13B-Instruct-GGUF",
                        filename="*Q4_K_M.gguf",
                        verbose=False
                    )"""

    def change_prompt(self,messages):
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += self.sys_first_prompt + message["content"] + self.sys_end_prompt
            elif message["role"] == "assistant":
                prompt += "assistant:" + message["content"] + "\n"
            elif message["role"] == "user":
                prompt += "user:" + message["content"] + "\n"
        prompt += "assistant:"
        return prompt

    def simpleLLM(self, user_prompt, temp_sys_prompt = None):
        if temp_sys_prompt is not None:
            self.simpleLLM_messages.append({"role": "system", "content": temp_sys_prompt})
        else:
            self.simpleLLM_messages.append({"role": "system", "content": self.sys_prompt})
        self.simpleLLM_messages.append({"role": "user", "content": user_prompt})

        making_prompt = self.change_prompt(self.simpleLLM_messages)

        res = self.llm.create_completion(
            making_prompt, 
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop = ["user"],
            )


        return res['choices'][0]['text']
    
    def simpleLLMstream_prepare(self, user_prompt, temp_sys_prompt = None):
        if temp_sys_prompt is not None:
            self.simpleLLM_messages.append({"role": "system", "content": temp_sys_prompt})
        else:
            self.simpleLLM_messages.append({"role": "system", "content": self.sys_prompt})
        self.simpleLLM_messages.append({"role": "user", "content": user_prompt})

        making_prompt = self.change_prompt(self.simpleLLM_messages)

        return making_prompt, self.max_tokens, self.temperature

