import torch
from diffusers import StableDiffusion3Pipeline, AutoencoderTiny , FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from transformers import T5EncoderModel, BitsAndBytesConfig
from PIL import Image

import os
import configparser
# ファイルの存在チェック用モジュール
import errno
import time

class SD3config:
    def __init__(self, config_ini_path = './configs/config.ini'):
        # iniファイルの読み込み
        self.config_ini = configparser.ConfigParser()
        
        # 指定したiniファイルが存在しない場合、エラー発生
        if not os.path.exists(config_ini_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)
        
        self.config_ini.read(config_ini_path, encoding='utf-8')
        SD3_items = self.config_ini.items('SD3')
        self.SD3_config_dict = dict(SD3_items)

class SD3:
    def __init__(self,device = None, config_ini_path = './configs/config.ini'):
        
        SD3_config = SD3config(config_ini_path = config_ini_path)
        config_dict = SD3_config.SD3_config_dict


        if device is not None:
            self.device = device
        else:
            device = config_dict["device"]

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if device != "auto":
                self.device = device
                
        self.n_steps = int(config_dict["n_steps"])
        self.seed = int(config_dict["seed"])
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.width = int(config_dict["width"])
        self.height = int(config_dict["height"])
        self.guided_scale = float(config_dict["guided_scale"])
        self.shift = float(config_dict["shift"])
        
        self.use_cpu_offload = config_dict["use_cpu_offload"]
        if self.use_cpu_offload == "True":
            self.use_cpu_offload = True
        else:
            self.use_cpu_offload = False
            
        self.use_text_encoder_3 = config_dict["use_text_encoder_3"]
        if self.use_text_encoder_3 == "True":
            self.use_text_encoder_3 = True
        else:
            self.use_text_encoder_3 = False
            
        self.use_T5_quantization = config_dict["use_t5_quantization"]
        if self.use_T5_quantization == "True":
            self.use_T5_quantization = True
        else:
            self.use_T5_quantization = False
        
        self.use_model_compile = config_dict["use_model_compile"]
        if self.use_model_compile == "True":
            self.use_model_compile = True
        else:
            self.use_model_compile = False
        
        self.save_latent = config_dict["save_latent"]
        if self.save_latent == "True":
            self.save_latent = True
        else:
            self.save_latent = False
            
        self.model_path = config_dict["model_path"]
        
    
        if self.use_model_compile:
            self.pipe  = self.preprepare_compile_model()
        else:
            self.pipe  = self.preprepare_model()
        

    def preprepare_model(self):
        
        pipe = None
        
        sampler = FlowMatchEulerDiscreteScheduler(
                    shift = self.shift
                    )

        
        if self.use_text_encoder_3:
            if self.use_T5_quantization:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

                text_encoder = T5EncoderModel.from_pretrained(
                    self.model_path,
                    subfolder="text_encoder_3",
                    quantization_config=quantization_config,
                )
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    self.model_path,
                    scheduler = sampler,
                    text_encoder_3=text_encoder,
                    device_map="balanced",
                    torch_dtype=torch.float16
                )
                
            else:
                pipe = StableDiffusion3Pipeline.from_pretrained(self.model_path, scheduler = sampler, torch_dtype=torch.float16)
        else:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                        self.model_path,
                        scheduler = sampler,
                        text_encoder_3=None,
                        tokenizer_3=None,
                        torch_dtype=torch.float16)
        
        
        if self.use_T5_quantization:
            pass
        elif self.use_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
            
        print(pipe.scheduler.config)

        return pipe
    
    
    def preprepare_compile_model(self):

        torch.set_float32_matmul_precision("high")

        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
        
        pipe = None
        sampler = FlowMatchEulerDiscreteScheduler(
                    shift = self.shift
                    )
        
        if self.use_text_encoder_3:
            if self.use_T5_quantization:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

                text_encoder = T5EncoderModel.from_pretrained(
                    self.model_path,
                    subfolder="text_encoder_3",
                    quantization_config=quantization_config,
                )
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    self.model_path,
                    scheduler = sampler,
                    text_encoder_3=text_encoder,
                    device_map="balanced",
                    torch_dtype=torch.float16
                )
                
            else:
                pipe = StableDiffusion3Pipeline.from_pretrained(self.model_path, scheduler = sampler, torch_dtype=torch.float16)
        else:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                        self.model_path,
                        scheduler = sampler,
                        text_encoder_3=None,
                        tokenizer_3=None,
                        torch_dtype=torch.float16)
        
        
        if self.use_T5_quantization:
            pass
        elif self.use_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
            
        print(pipe.scheduler.config)
            
        pipe.set_progress_bar_config(disable=True)
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)
        
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

        return pipe
        
            
    
    def generate_image(self, prompt, prompt_2 = None, prompt_3 = None, neg_prompt = "", neg_prompt_2 = None, neg_prompt_3 = None,seed = None):
        
        def decode_tensors(pipe, step, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
        
            image = latents_to_rgb(latents,pipe)
            gettime = time.time()
            formatted_time_human_readable = time.strftime("%Y%m%d_%H%M%S", time.localtime(gettime))
            image.save(f"./outputs/latent_{formatted_time_human_readable}_{step}_{timestep}.png")
        
        
            return callback_kwargs
            
        def latents_to_rgb(latents,pipe):

            latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

            img = pipe.vae.decode(latents, return_dict=False)[0]
            img = pipe.image_processor.postprocess(img, output_type="pil")
        
            return StableDiffusion3PipelineOutput(images=img).images[0]
        
        if seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)
            
        if prompt_2 is None:
            prompt_2 = prompt
        if prompt_3 is None:
            prompt_3 = prompt
        if neg_prompt_2 is None:
            neg_prompt_2 = neg_prompt
        if neg_prompt_3 is None:
            neg_prompt_3 = neg_prompt
            
        image = None
        if self.save_latent:
            image = self.pipe(
                prompt=prompt, 
                prompt_2=prompt_2, 
                prompt_3=prompt_3, 
                negative_prompt=neg_prompt,
                negative_prompt_2 = neg_prompt_2,
                negative_prompt_3 = neg_prompt_3,
                height = self.height,
                width = self.width,
                num_inference_steps=self.n_steps,
                guidance_scale=self.guided_scale,
                generator=self.generator,
                callback_on_step_end=decode_tensors,
                callback_on_step_end_tensor_inputs=["latents"],
                ).images[0]
        else:
            image = self.pipe(
                prompt=prompt, 
                prompt_2=prompt_2, 
                prompt_3=prompt_3, 
                negative_prompt=neg_prompt,
                negative_prompt_2 = neg_prompt_2,
                negative_prompt_3 = neg_prompt_3,
                height = self.height,
                width = self.width,
                num_inference_steps=self.n_steps,
                guidance_scale=self.guided_scale,
                generator=self.generator
                ).images[0]
        
        
        return image
        

    
