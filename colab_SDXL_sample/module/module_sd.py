from diffusers import DiffusionPipeline, AutoencoderKL
import torch
from diffusers.schedulers import DPMSolverMultistepScheduler

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
        SDXL_items = self.config_ini.items('SDXL')
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
        self.high_noise_frac = float(config_dict["high_noise_frac"])
        self.seed = int(config_dict["seed"])
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        
        
        self.vae_model_path = config_dict["vae_model_path"]
        self.VAE_FLAG = True
        if self.vae_model_path == "None":
            self.vae_model_path = None
            self.VAE_FLAG = False
            
        self.base_model_path = config_dict["base_model_path"]
        
        self.REFINER_FLAG = True
        self.refiner_model_path = config_dict["refiner_model_path"]
        if self.refiner_model_path == "None":
            self.refiner_model_path = None
            self.REFINER_FLAG = False
            
        self.use_karras_sigmas = config_dict["use_karras_sigmas"]
        if self.use_karras_sigmas == "True":
            self.use_karras_sigmas = True
        else:
            self.use_karras_sigmas = False
        self.scheduler_algorithm_type = config_dict["scheduler_algorithm_type"]
        if config_dict["solver_order"] != "None":
            self.solver_order = int(config_dict["solver_order"])
        else:
            self.solver_order = None
            
        self.cfg_scale = float(config_dict["cfg_scale"])
        self.width = int(config_dict["width"])
        self.height = int(config_dict["height"])
        self.output_type = config_dict["output_type"]
        self.aesthetic_score = float(config_dict["aesthetic_score"])
        self.negative_aesthetic_score = float(config_dict["negative_aesthetic_score"])
        
        self.base , self.refiner = self.preprepare_model()
        

    def preprepare_model(self):
        if self.VAE_FLAG:
            vae = AutoencoderKL.from_pretrained(
                self.vae_model_path,
                torch_dtype=torch.float16)

            base = DiffusionPipeline.from_pretrained(
                self.base_model_path, 
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            base.to(self.device)
    
            if self.REFINER_FLAG:
                refiner = DiffusionPipeline.from_pretrained(
                    self.refiner_model_path,
                    text_encoder_2=base.text_encoder_2,
                    vae=vae,
                    requires_aesthetics_score=True,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True
                )

                refiner.enable_model_cpu_offload()
            else:
                refiner = None
            
        else:
            base = DiffusionPipeline.from_pretrained(
                self.base_model_path, 
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            base.to(self.device)
            
            if self.REFINER_FLAG:
                refiner = DiffusionPipeline.from_pretrained(
                    self.refiner_model_path,
                    text_encoder_2=base.text_encoder_2,
                    requires_aesthetics_score=True,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True
                )

                refiner.enable_model_cpu_offload()
            else:
                refiner = None
                
        if self.solver_order is not None:
            base.scheduler = DPMSolverMultistepScheduler.from_config(
                    base.scheduler.config, 
                    use_karras_sigmas=self.use_karras_sigmas,
                    Algorithm_type =self.scheduler_algorithm_type,
                    solver_order=self.solver_order,
                    )
            return base, refiner
        else:
            base.scheduler = DPMSolverMultistepScheduler.from_config(
                    base.scheduler.config, 
                    use_karras_sigmas=self.use_karras_sigmas,
                    Algorithm_type =self.scheduler_algorithm_type,
                    )
            return base, refiner
            
    
    def generate_image(self, prompt, neg_prompt,seed = None):
        
        if seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.base(
            prompt=prompt,
            negative_prompt=neg_prompt, 
            cfg_scale=self.cfg_scale,
            num_inference_steps=self.n_steps,
            denoising_end=self.high_noise_frac,
            output_type=self.output_type,
            width = self.width, 
            height = self.height,
            generator=self.generator
            ).images[0]
        if self.REFINER_FLAG:
            image = self.refiner(
                prompt=prompt,
                negative_prompt=neg_prompt, 
                cfg_scale=self.cfg_scale,
                aesthetic_score = self.aesthetic_score,
                negative_aesthetic_score = self.negative_aesthetic_score,
                num_inference_steps=self.n_steps,
                denoising_start=self.high_noise_frac,
                image=image[None, :]
                ).images[0]
        return image
    
if __name__ == "__main__":
    sd = SDXL()
    prompt = "1 girl ,pink hair ,long hair ,blue ribbon ,train interior ,school uniform ,solo ,smile ,upper body"
    #neg_prompt = "extra limbs ,NSFW ,text ,signature ,bad anatomy ,((worth quality ,low quality)) ,normal quality ,bad face ,bad hand ,missing fingers ,missing limbs ,extra fingers ,extra limbs"
    neg_prompt = "Easy negatvie, (worst quality:2),(low quality:2),(normal quality:2), lowers, normal quality,((monochrome)),((grayscale)),skin spots, acnes, skin blemishes, age spot, nsfw, ugly face, fat, missing fingers,missing limbs, extra fingers, extra arms, extra legs,extra limbs, watermark, text, error, blurry, jpeg artifacts, cropped, bad anatomy, bad hand, big eyes"
    for i in range(3):
        image = sd.generate_image(prompt, neg_prompt)
        image.save("result_{}.png".format(i))