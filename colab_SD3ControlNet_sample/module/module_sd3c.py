import torch
from diffusers import StableDiffusion3ControlNetPipeline, AutoencoderTiny , FlowMatchEulerDiscreteScheduler
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from transformers import T5EncoderModel, BitsAndBytesConfig
from PIL import Image

import os
import configparser
# ファイルの存在チェック用モジュール
import errno
import time
import numpy as np
import GPUtil
import gc

class SD3Cconfig:
    def __init__(self, config_ini_path = './configs/config.ini'):
        # iniファイルの読み込み
        self.config_ini = configparser.ConfigParser()

        # 指定したiniファイルが存在しない場合、エラー発生
        if not os.path.exists(config_ini_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)

        self.config_ini.read(config_ini_path, encoding='utf-8')
        SD3C_items = self.config_ini.items('SD3C')
        self.SD3C_config_dict = dict(SD3C_items)

class SD3C:
    def __init__(self,device = None, config_ini_path = './configs/config.ini'):

        SD3C_config = SD3Cconfig(config_ini_path = config_ini_path)
        config_dict = SD3C_config.SD3C_config_dict


        if device is not None:
            self.device = device
        else:
            device = config_dict["device"]

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if device != "auto":
                self.device = device

        self.prompt_embeds = None
        self.negative_prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.negative_pooled_prompt_embeds = None
        self.last_prompt = None
        self.last_prompt_2 = None
        self.last_prompt_3 = None
        self.last_neg_prompt = None
        self.last_neg_prompt_2 = None
        self.last_neg_prompt_3 = None

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

        self.save_latent = config_dict["save_latent"]
        if self.save_latent == "True":
            self.save_latent = True
        else:
            self.save_latent = False

        self.model_path = config_dict["model_path"]
        self.controlnet_path = config_dict["controlnet_path"]


        self.pipe  = self.preprepare_model()


    def preprepare_model(self):

        pipe = None

        controlnet = SD3ControlNetModel.from_pretrained(self.controlnet_path, torch_dtype=torch.float16)
        sampler = FlowMatchEulerDiscreteScheduler(
                    shift = self.shift
                    )

        if self.use_text_encoder_3:
            if self.use_T5_quantization:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

                self.text_encoder = T5EncoderModel.from_pretrained(
                    self.model_path,
                    subfolder="text_encoder_3",
                    quantization_config=quantization_config,
                )
                self.prepipe = StableDiffusion3ControlNetPipeline.from_pretrained(
                    self.model_path,
                    controlnet=None,
                    scheduler = None,
                    text_encoder_3=self.text_encoder,
                    transformer=None,
                    vae=None,
                    device_map="balanced"
                )

                pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
                  self.model_path,
                  controlnet=controlnet,
                  scheduler = sampler,
                  text_encoder=None,
                  text_encoder_2=None,
                  text_encoder_3=None,
                  tokenizer=None,
                  tokenizer_2=None,
                  tokenizer_3=None,
                  torch_dtype=torch.float16
                )


            else:
                pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
                    self.model_path,
                    controlnet=controlnet,
                    scheduler = sampler,
                    torch_dtype=torch.float16)
        else:
            pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
                        self.model_path,
                        controlnet=controlnet,
                        scheduler = sampler,
                        text_encoder_3=None,
                        tokenizer_3=None,
                        torch_dtype=torch.float16)


        if self.use_T5_quantization:
            pipe = pipe.to("cpu")
        elif self.use_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")

        print(pipe.scheduler.config)

        return pipe


    def prepare_referimage(self,input_refer_image_path,output_refer_image_path, low_threshold = 100, high_threshold = 200):
        def extract_last_segment(text):
            segments = text.split('-')
            return segments[-1]

        mode = extract_last_segment(self.controlnet_path)

        def prepare_openpose(input_refer_image_path,output_refer_image_path):
            from diffusers.utils import load_image
            from controlnet_aux import OpenposeDetector

            # 初期画像の準備
            init_image = load_image(input_refer_image_path)
            init_image = init_image.resize((self.width, self.height))

            # コントロール画像の準備
            openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            openpose_image = openpose_detector(init_image)

            openpose_image.save(output_refer_image_path)

        def prepare_canny(input_refer_image_path,output_refer_image_path, low_threshold = 100, high_threshold = 200):
            init_image = load_image(input_refer_image_path)
            init_image = init_image.resize((self.width, self.height))
            import cv2


            # コントロールイメージを作成するメソッド
            def make_canny_condition(image, low_threshold = 100, high_threshold = 200):
                image = np.array(image)
                image = cv2.Canny(image, low_threshold, high_threshold)
                image = image[:, :, None]
                image = np.concatenate([image, image, image], axis=2)
                return Image.fromarray(image)

            control_image = make_canny_condition(init_image, low_threshold, high_threshold)
            control_image.save(output_refer_image_path)

        if mode == "Pose":
            prepare_openpose(input_refer_image_path,output_refer_image_path)
        elif mode == "Canny":
            prepare_canny(input_refer_image_path,output_refer_image_path, low_threshold = low_threshold, high_threshold = high_threshold)
        else:
            init_image = load_image(input_refer_image_path)
            init_image.save(output_refer_image_path)
            
    def check_prepipe(self):
        if hasattr(self, 'prepipe'):
            print("prepipe exists.")
        else:
            print("prepipe does not exist.")
            self.pipe = self.pipe.to("cpu")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            self.text_encoder = T5EncoderModel.from_pretrained(
                self.model_path,
                subfolder="text_encoder_3",
                quantization_config=quantization_config,
            )
            self.prepipe = StableDiffusion3ControlNetPipeline.from_pretrained(
                self.model_path,
                controlnet=None,
                scheduler = None,
                text_encoder_3=self.text_encoder,
                transformer=None,
                vae=None,
                device_map="balanced"
            )
            


    def generate_image(self, prompt, prompt_2 = None, prompt_3 = None, neg_prompt = "", neg_prompt_2 = None, neg_prompt_3 = None, image_path = None,  seed = None, controlnet_conditioning_scale = 1.0):

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

        def check_prompts(prompt, prompt_2, prompt_3, neg_prompt, neg_prompt_2, neg_prompt_3):
            # タプルに変数を格納
            last_prompts = (self.last_prompt, self.last_prompt_2, self.last_prompt_3,
                            self.last_neg_prompt, self.last_neg_prompt_2, self.last_neg_prompt_3)

            current_prompts = (prompt, prompt_2, prompt_3,
                            neg_prompt, neg_prompt_2, neg_prompt_3)

            # 全ての変数が一致しているか判定
            return last_prompts == current_prompts

        if image_path is None:
            raise ValueError("ControlNetを利用する場合は、画像のパスをimage_path引数に提示してください。")
        control_image = load_image(image_path)

        if prompt_2 is None:
            prompt_2 = prompt
        if prompt_3 is None:
            prompt_3 = prompt
        if neg_prompt_2 is None:
            neg_prompt_2 = neg_prompt
        if neg_prompt_3 is None:
            neg_prompt_3 = neg_prompt

        image = None
        #T5を利用して、かつ、量子化する場合
        if (self.use_T5_quantization) and (self.use_text_encoder_3):
            #GPUメモリを解放する都合上、繰り返しメソッドを実行するために、T5の量子化を行う場合はプロンプトの埋め込みを保存しておく必要がある。
            #前回の実行とプロンプトが一つでも異なる場合は、そのプロンプトをインスタンス変数に一旦保存した上で、埋め込みを計算し、インスタンス変数に保存する。
            if not check_prompts(prompt, prompt_2, prompt_3, neg_prompt, neg_prompt_2, neg_prompt_3):
                self.check_prepipe()
                
                self.last_prompt = prompt
                self.last_prompt_2 = prompt_2
                self.last_prompt_3 = prompt_3
                self.last_neg_prompt = neg_prompt
                self.last_neg_prompt_2 = neg_prompt_2
                self.last_neg_prompt_3 = neg_prompt_3

                with torch.no_grad():
                    (
                    self.prompt_embeds,
                    self.negative_prompt_embeds,
                    self.pooled_prompt_embeds,
                    self.negative_pooled_prompt_embeds,
                    ) = self.prepipe.encode_prompt(
                        prompt=prompt,
                        prompt_2=prompt_2,
                        prompt_3=prompt_3,
                        negative_prompt=neg_prompt,
                        negative_prompt_2 = neg_prompt_2,
                        negative_prompt_3 = neg_prompt_3,
                        )

                #無理やりGPUメモリを解放する。GoogleColabだと何故かtorch.cuda.empty_cache()を2回実行しないと解放されない。
                GPUtil.showUtilization()
                del self.prepipe
                del self.text_encoder
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                GPUtil.showUtilization()
                ### ここまで

                #メモリが解放されたので、次のモデルをGPUでロード
                self.pipe = self.pipe.to("cuda")

            #プロンプトが一致している場合は、埋め込みを再利用する。
            else:
                pass

            if self.save_latent:
                image = self.pipe(
                    prompt_embeds=self.prompt_embeds.half(),
                    negative_prompt_embeds=self.negative_prompt_embeds.half(),
                    pooled_prompt_embeds=self.pooled_prompt_embeds.half(),
                    negative_pooled_prompt_embeds=self.negative_pooled_prompt_embeds.half(),
                    control_image = control_image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
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
                        prompt_embeds=self.prompt_embeds.half(),
                        negative_prompt_embeds=self.negative_prompt_embeds.half(),
                        pooled_prompt_embeds=self.pooled_prompt_embeds.half(),
                        negative_pooled_prompt_embeds=self.negative_pooled_prompt_embeds.half(),
                        control_image = control_image,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        height = self.height,
                        width = self.width,
                        num_inference_steps=self.n_steps,
                        guidance_scale=self.guided_scale,
                        generator=self.generator
                        ).images[0]

        #T5を利用しない、もしくは、量子化しない場合
        else:
            if self.save_latent:
                image = self.pipe(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    prompt_3=prompt_3,
                    negative_prompt=neg_prompt,
                    negative_prompt_2 = neg_prompt_2,
                    negative_prompt_3 = neg_prompt_3,
                    control_image = control_image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
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
                    control_image = control_image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    height = self.height,
                    width = self.width,
                    num_inference_steps=self.n_steps,
                    guidance_scale=self.guided_scale,
                    generator=self.generator
                    ).images[0]


        return image



