from diffusers import AutoPipelineForText2Image, AutoencoderKL, EulerAncestralDiscreteScheduler
import torch

import os
import configparser
# ファイルの存在チェック用モジュール
import errno

class SDXLconfig:
    def __init__(self, config_ini_path = './configs/config.ini'):
        # iniファイルの読み込み
        self.config_ini = configparser.ConfigParser()
        
        # 指定したiniファイルが存在しない場合、エラー発生
        if not os.path.exists(config_ini_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)
        
        self.config_ini.read(config_ini_path, encoding='utf-8')
        SDXL_items = self.config_ini.items('SDXL-turbo')
        self.SDXL_config_dict = dict(SDXL_items)

class SDXL:
    def __init__(self,device = None, config_ini_path = './configs/config.ini'):
        
        SDXL_config = SDXLconfig(config_ini_path = config_ini_path)
        config_dict = SDXL_config.SDXL_config_dict

        if device is not None:
            self.device = device
        else:
            device = config_dict["device"]

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if device != "auto":
                self.device = device
        
        self.n_steps = int(config_dict["n_steps"])
        self.seed = int(config_dict["seed"])
        self.guidance_scale = float(config_dict["guidance_scale"])
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        self.vae_model_path = config_dict["vae_model_path"]
        self.VAE_FLAG = True
        if self.vae_model_path == "None":
            self.vae_model_path = None
            self.VAE_FLAG = False
            
        self.base_model_path = config_dict["base_model_path"]
        self.cfg_scale = float(config_dict["cfg_scale"])
        self.width = int(config_dict["width"])
        self.height = int(config_dict["height"])

        self.base  = self.preprepare_model()
        

    def preprepare_model(self):
        if self.VAE_FLAG:
            print("load vae")
            vae = AutoencoderKL.from_pretrained(
                self.vae_model_path,
                torch_dtype=torch.float16)

            base = AutoPipelineForText2Image.from_pretrained(
                self.base_model_path, 
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            base.to(self.device)
    
            
        else:
            print("non vae")
            base = AutoPipelineForText2Image.from_pretrained(
                self.base_model_path, 
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            base.to(self.device)


        base.scheduler = EulerAncestralDiscreteScheduler.from_config(
                    base.scheduler.config, 
                    timestep_spacing= "trailing"
                    )
                    
        return base

            
    
    def generate_image(self, prompt, neg_prompt,seed = None):
        
        if seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.base(
            prompt=prompt,
            negative_prompt = neg_prompt,
            cfg_scale=self.cfg_scale,
            guidance_scale = self.guidance_scale,
            num_inference_steps=self.n_steps,
            width = self.width, 
            height = self.height,
            generator=self.generator
            ).images[0]

        return image
    
    def generate_image_display(self, prompt, neg_prompt,cnt = 0,seed = None):
        
        if seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.base(
            prompt=prompt,
            negative_prompt = neg_prompt,
            cfg_scale=self.cfg_scale,
            guidance_scale = self.guidance_scale,
            num_inference_steps=self.n_steps,
            width = self.width, 
            height = self.height,
            generator=self.generator
            ).images[0]
        
        image.show()
        image.save("output_images/sample{}.png".format(cnt))

        return image
    
