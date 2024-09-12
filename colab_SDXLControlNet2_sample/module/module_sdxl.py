from math import e
from diffusers import DiffusionPipeline, AutoencoderKL, StableDiffusionXLControlNetPipeline, ControlNetModel, StableDiffusionXLPipeline
from diffusers.utils import load_image
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
import torch
from controlnet_aux.processor import Processor
from diffusers.schedulers import DPMSolverMultistepScheduler, LCMScheduler, EulerAncestralDiscreteScheduler, FlowMatchEulerDiscreteScheduler


import glob

import os
import configparser
# ファイルの存在チェック用モジュール
import errno
import cv2
from PIL import Image, ImageFilter
import time
import numpy as np


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

        self.last_latents = None
        self.last_step = -1
        self.last_timestep = 1000

        self.n_steps = int(config_dict["n_steps"])

        self.seed = int(config_dict["seed"])
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        self.use_controlnet = config_dict.get("use_controlnet", "False")
        if self.use_controlnet == "False":
            self.use_controlnet = False
        else:
            self.use_controlnet = True

        self.controlnet_path_list = []
        for i in range(5):
            if config_dict.get(f"controlnet_path{i}", "None") != "None":
                self.controlnet_path_list.append(config_dict[f"controlnet_path{i}"])
            else:
                self.controlnet_path_list.append(None)

        self.single_controlnet_bool_list = []
        for i in range(5):
            if config_dict.get(f"from_single_controlnet_file{i}", "False") != "False":
                self.single_controlnet_bool_list.append(True)
            else:
                self.single_controlnet_bool_list.append(False)

        self.controlnet_variant_bool_list = []
        for i in range(5):
            if config_dict.get(f"use_controlnet_variant{i}", "False") != "False":
                self.controlnet_variant_bool_list.append(True)
            else:
                self.controlnet_variant_bool_list.append(False)

        self.controlnet_config_list = []
        for i in range(5):
            if config_dict.get(f"controlnet_config_repo{i}", "None") != "None":
                self.controlnet_config_list.append(config_dict[f"controlnet_config_repo{i}"])
            else:
                self.controlnet_config_list.append(None)

        if not self.use_controlnet and self.controlnet_path_list == []:
            raise ValueError("controlnet_path is not set")
        if len(self.controlnet_path_list) != len(self.single_controlnet_bool_list):
            raise ValueError("controlnet_path and from_single_controlnet_file is not match")
        if len(self.controlnet_path_list) != len(self.controlnet_variant_bool_list):
            raise ValueError("controlnet_path and use_controlnet_variant is not match")
        if len(self.controlnet_path_list) != len(self.controlnet_config_list):
            raise ValueError("controlnet_path and controlnet_config_repo is not match")

        #ControlNet一つ指定につき、一つのControlModeを指定する
        self.control_modes_list = []
        for i in range(5):
            if config_dict.get(f"control_mode{i}", "None") != "None":
                self.control_modes_list.append(config_dict[f"control_mode{i}"])
            else:   
                self.control_modes_list.append(None)

        if len(self.controlnet_path_list) != len(self.control_modes_list):
            raise ValueError("controlnet_path and control_mode is not match")

        self.from_single_file = config_dict.get("from_single_file", "None")
        self.SINGLE_FILE_FLAG = True
        if self.from_single_file != "True":
            self.from_single_file = None
            self.SINGLE_FILE_FLAG = False


        self.base_model_path = config_dict["base_model_path"]

        self.from_single_vae_file = config_dict.get("from_single_vae_file", "False")
        if self.from_single_vae_file == "False":
            self.from_single_vae_file = False
        else:
            self.from_single_vae_file = True
        self.vae_model_path = config_dict.get("vae_model_path", "None")
        self.VAE_FLAG = True
        if self.vae_model_path == "None":
            self.vae_model_path = None
            self.VAE_FLAG = False



        self.lora_repo_list = []
        self.lora_path_list = []
        self.lora_scale_list = []
        self.lora_trigger_word_list = []
        self.lora_nums = 0
        self.LORA_FLAG = False
        for i in range(10):
            if config_dict.get(f"lora_weight_path{i}", "None") != "None":
                self.LORA_FLAG = True
                #Huggingfaceのrepoから取得する場合は、repoとpath（ファイル名）の両方を指定する
                if config_dict.get(f"lora_weight_repo{i}", "None") != "None":
                    self.lora_repo_list.append(config_dict[f"lora_weight_repo{i}"])
                #Civitaiなどからダウンロードして利用する場合は、ディレクトリ名含むpathのみで指定する
                else:
                    self.lora_repo_list.append(None)
                self.lora_path_list.append(config_dict[f"lora_weight_path{i}"])
                self.lora_nums += 1
                self.lora_scale_list.append(float(config_dict.get(f"lora_scale{i}", "1.0")))
                self.lora_trigger_word_list.append(config_dict.get(f"trigger_word{i}", "None"))


        self.select_solver = config_dict.get("select_solver", "FMEuler")

        self.use_karras_sigmas = config_dict.get("use_karras_sigmas", "True")
        if self.use_karras_sigmas == "True":
            self.use_karras_sigmas = True
        else:
            self.use_karras_sigmas = False

        self.scheduler_algorithm_type = config_dict.get("scheduler_algorithm_type", "dpmsolver++")
        self.solver_order = config_dict.get("solver_order", "None")
        if self.solver_order != "None":
            self.solver_order = int(self.solver_order)
        else:
            self.solver_order = None


        self.cfg_scale = float(config_dict["cfg_scale"])
        self.width = int(config_dict["width"])
        self.height = int(config_dict["height"])
        self.output_type = config_dict["output_type"]
        self.aesthetic_score = float(config_dict["aesthetic_score"])
        self.negative_aesthetic_score = float(config_dict["negative_aesthetic_score"])

        self.save_latent_simple = config_dict["save_latent_simple"]
        if self.save_latent_simple == "True":
            self.save_latent_simple = True
            print("use callback save_latent_simple")
        else:
            self.save_latent_simple = False

        self.save_latent_overstep = config_dict["save_latent_overstep"]
        if self.save_latent_overstep == "True":
            self.save_latent_overstep = True
            print("use callback save_latent_overstep")
        else:
            self.save_latent_overstep = False

        self.save_latent_approximation = config_dict["save_latent_approximation"]
        if self.save_latent_approximation == "True":
            self.save_latent_approximation = True
            print("use callback save_latent_approximation")
        else:
            self.save_latent_approximation = False

        self.save_predict_skip_x0 = config_dict["save_predict_skip_x0"]
        if self.save_predict_skip_x0 == "True":
            self.save_predict_skip_x0 = True
            print("use callback save_predict_skip_x0")
        else:
            self.save_predict_skip_x0 = False

        self.use_callback = False
        if self.save_latent_simple or self.save_latent_overstep or self.save_latent_approximation or self.save_predict_skip_x0:
            self.use_callback = True

        if self.save_predict_skip_x0:
            if self.save_latent_simple or self.save_latent_overstep:
                raise ValueError("save_predict_skip_x0 and (save_latent_simple or save_latent_overstep) cannot be set at the same time")
            if self.use_dpm_solver:
                raise ValueError("save_predict_skip_x0 and use_dpm_solver cannot be set at the same time")
        else:
            if self.save_latent_simple and self.save_latent_overstep:
                raise ValueError("save_latent_simple and save_latent_overstep cannot be set at the same time")


        self.base = self.preprepare_model()


    def preprepare_model(self, controlnet_path = None):
        if controlnet_path is not None:
            self.controlnet_path = controlnet_path

        vae = None
        if self.VAE_FLAG:
            if self.from_single_vae_file:
                vae = AutoencoderKL.from_single_file(
                    self.vae_model_path,
                    torch_dtype=torch.float16)
            else:
                vae = AutoencoderKL.from_pretrained(
                    self.vae_model_path,
                    torch_dtype=torch.float16)

        if self.use_controlnet:
            controlnets = []
            for model_step, cnet_path in enumerate(self.controlnet_path_list):
                if cnet_path is not None:
                    if self.single_controlnet_bool_list[model_step]:
                        if self.controlnet_config_list[model_step] is not None:
                            print(model_step, self.controlnet_config_list[model_step])
                            controlnet = ControlNetModel.from_single_file(cnet_path, config = self.controlnet_config_list[model_step], torch_dtype=torch.float16)
                        else:
                            controlnet = ControlNetModel.from_single_file(cnet_path, torch_dtype=torch.float16)
                    else:
                        if self.controlnet_variant_bool_list[model_step]:
                            controlnet = ControlNetModel.from_pretrained(cnet_path, 
                                        torch_dtype=torch.float16, 
                                        variant="fp16", 
                                        use_safetensors=True
                                        )
                        else:
                            controlnet = ControlNetModel.from_pretrained(cnet_path, torch_dtype=torch.float16)
                    controlnets.append(controlnet)

            print("loaded controlnet")

            if not self.SINGLE_FILE_FLAG:
                if self.VAE_FLAG:
                    base = StableDiffusionXLControlNetPipeline.from_pretrained(
                        self.base_model_path,
                        vae=vae,
                        controlnet=controlnets,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True
                        )
                else:
                    base = StableDiffusionXLControlNetPipeline.from_pretrained(
                        self.base_model_path,
                        controlnet=controlnets,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True
                    )
                print("loaded base model")
            else:
                if self.VAE_FLAG:
                    pipe = StableDiffusionXLPipeline.from_single_file(
                        self.base_model_path,
                        vae=vae,
                        extract_ema=True,
                        torch_dtype=torch.float16
                        )
                else:
                    pipe = StableDiffusionXLPipeline.from_single_file(
                        self.base_model_path,
                        extract_ema=True,
                        torch_dtype=torch.float16
                        )

                base = StableDiffusionXLControlNetPipeline(controlnet = controlnets, **pipe.components)
                print("loaded base model")

        else:
            controlnet = None
            if not self.SINGLE_FILE_FLAG:
                if self.VAE_FLAG:
                    base = StableDiffusionXLPipeline.from_pretrained(
                        self.base_model_path,
                        vae=vae,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True
                        )
                else:
                    base = StableDiffusionXLPipeline.from_pretrained(
                        self.base_model_path,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True
                        )
                print("loaded base model")

            else:
                if self.VAE_FLAG:
                    base = StableDiffusionXLPipeline.from_single_file(
                        self.base_model_path,
                        vae=vae,
                        extract_ema=True,
                        torch_dtype=torch.float16
                        )
                else:
                    base = StableDiffusionXLPipeline.from_single_file(
                        self.base_model_path,
                        extract_ema=True,
                        torch_dtype=torch.float16
                        )
                print("loaded base model")


        base.to(self.device)
        #print("cpu offloading")
        #base.enable_model_cpu_offload()


        lora_adapter_name_list = []
        lora_adapter_weights_list = []
        if self.LORA_FLAG:
            for i in range(self.lora_nums):
                if self.lora_repo_list[i] is not None:
                    base.load_lora_weights(
                        pretrained_model_name_or_path_or_dict = self.lora_repo_list[i],
                        weight_name=self.lora_path_list[i],
                        adapter_name=f"lora{i}")
                else:
                    base.load_lora_weights(self.lora_path_list[i], adapter_name=f"lora{i}")
                lora_adapter_name_list.append(f"lora{i}")
                lora_adapter_weights_list.append(self.lora_scale_list[i])
            if self.lora_nums > 1:
                base.set_adapters(lora_adapter_name_list, adapter_weights=lora_adapter_weights_list)

            print("finish lora settings")

        if self.select_solver == "DPM":
            if self.solver_order is not None:
                base.scheduler = DPMSolverMultistepScheduler.from_config(
                        base.scheduler.config,
                        use_karras_sigmas=self.use_karras_sigmas,
                        Algorithm_type =self.scheduler_algorithm_type,
                        solver_order=self.solver_order,
                        )

            else:
                base.scheduler = DPMSolverMultistepScheduler.from_config(
                        base.scheduler.config,
                        use_karras_sigmas=self.use_karras_sigmas,
                        Algorithm_type =self.scheduler_algorithm_type,
                        )
        elif self.select_solver == "LCM":
            base.scheduler = LCMScheduler.from_config(base.scheduler.config)
        elif self.select_solver == "Eulera":
            base.scheduler = EulerAncestralDiscreteScheduler.from_config(base.scheduler.config)
        elif self.select_solver == "FMEuler":
            base.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base.scheduler.config)
        else:
            raise ValueError("select_solver is only 'DPM' or 'LCM' or 'Eulera' or 'FMEuler'.")

        return base


    def prepare_multi_referimage(self,input_refer_image_folder,output_refer_image_folder, low_threshold = 100, high_threshold = 200, noise_level=25, blur_radius=5):
        #input_refer_image_folderの中にある画像のpathを全て取得する
        def get_image_paths_sorted_by_filename_number(input_refer_image_folder, output_refer_image_folder):
            # 対象の画像拡張子のリスト
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']

            # フォルダ内のファイルを取得し、画像のみフィルタリング
            image_paths = [
                os.path.join(input_refer_image_folder, f)
                for f in os.listdir(input_refer_image_folder)
                if os.path.isfile(os.path.join(input_refer_image_folder, f)) and os.path.splitext(f)[1].lower() in image_extensions
            ]

            print(f"{image_paths=}")
            print(f"{self.control_modes_list=}")

            # ファイル名の数字部分でソート（小さい順）
            #image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

            #self.control_modes_listに入っている文字列がimage_pathに含まれているか確認する
            image_paths_control = []
            for image_path in image_paths:
                for control_mode in self.control_modes_list:
                    if control_mode is not None:
                        if control_mode in image_path:
                            image_paths_control.append(image_path)
                            break


            # 出力用のフォルダパスに変更したリストを作成
            output_image_paths = [
                os.path.join(output_refer_image_folder, os.path.basename(path))
                for path in image_paths_control
            ]

            print(f"{image_paths_control=}, {output_image_paths=}")

            return image_paths_control, output_image_paths

         #output_refer_image_folderが存在しない場合、作成する
        if not os.path.exists(output_refer_image_folder):
            os.makedirs(output_refer_image_folder)

        input_paths, output_paths = get_image_paths_sorted_by_filename_number(input_refer_image_folder, output_refer_image_folder)

        for input_refer_image_path, output_refer_image_path in zip(input_paths, output_paths):
            mode = os.path.splitext(os.path.basename(input_refer_image_path))[0]
            self.prepare_referimage(input_refer_image_path,output_refer_image_path, low_threshold = low_threshold, high_threshold = high_threshold, noise_level=noise_level, blur_radius=blur_radius, mode = mode)




    def prepare_referimage(self,input_refer_image_path,output_refer_image_path, low_threshold = 100, high_threshold = 200, noise_level=25, blur_radius=5, mode = 0):
        #mode = canny , tile , depth , blur , pose , gray , low quality .

        def prepare_openpose(input_refer_image_path,output_refer_image_path, mode):

            # 初期画像の準備
            init_image = load_image(input_refer_image_path)
            init_image = init_image.resize((self.width, self.height))

            processor = Processor(mode)
            processed_image = processor(init_image, to_pil=True)

            processed_image.save(output_refer_image_path)


        def prepare_canny(input_refer_image_path,output_refer_image_path, low_threshold = 100, high_threshold = 200):
            init_image = load_image(input_refer_image_path)
            init_image = init_image.resize((self.width, self.height))

            # コントロールイメージを作成するメソッド
            def make_canny_condition(image, low_threshold = 100, high_threshold = 200):
                image = np.array(image)
                image = cv2.Canny(image, low_threshold, high_threshold)
                image = image[:, :, None]
                image = np.concatenate([image, image, image], axis=2)
                return Image.fromarray(image)

            control_image = make_canny_condition(init_image, low_threshold, high_threshold)
            control_image.save(output_refer_image_path)

        def prepare_depthmap(input_refer_image_path,output_refer_image_path):

            # 初期画像の準備
            init_image = load_image(input_refer_image_path)
            init_image = init_image.resize((self.width, self.height))
            processor = Processor("depth_midas")
            depth_image = processor(init_image, to_pil=True)
            depth_image.save(output_refer_image_path)

        def prepare_zoe_depthmap(input_refer_image_path,output_refer_image_path):

            torch.hub.help(
                "intel-isl/MiDaS",
                "DPT_BEiT_L_384",
                force_reload=True
                )
            model_zoe_n = torch.hub.load(
                "isl-org/ZoeDepth",
                "ZoeD_NK",
                pretrained=True
                ).to("cuda")

            init_image = load_image(input_refer_image_path)
            init_image = init_image.resize((self.width, self.height))

            depth_numpy = model_zoe_n.infer_pil(init_image)  # return: numpy.ndarray

            from zoedepth.utils.misc import colorize
            colored = colorize(depth_numpy) # numpy.ndarray => numpy.ndarray

            # gamma correction
            img = colored / 255
            img = np.power(img, 2.2)
            img = (img * 255).astype(np.uint8)

            Image.fromarray(img).save(output_refer_image_path)

        def prepare_blur(input_refer_image_path, output_refer_image_path, blur_radius=5):
            init_image = load_image(input_refer_image_path)
            init_image = init_image.resize((self.width, self.height))

            # Blur画像を作成するメソッド
            def make_blur_condition(image, blur_radius=5):
                return image.filter(ImageFilter.GaussianBlur(blur_radius))

            blurred_image = make_blur_condition(init_image, blur_radius)
            blurred_image.save(output_refer_image_path)

        def prepare_grayscale(input_refer_image_path, output_refer_image_path):
            init_image = load_image(input_refer_image_path)
            init_image = init_image.resize((self.width, self.height))

            # グレースケール画像を作成するメソッド
            def make_grayscale_condition(image):
                return image.convert("L")

            grayscale_image = make_grayscale_condition(init_image)
            grayscale_image.save(output_refer_image_path)

        def prepare_noise(input_refer_image_path, output_refer_image_path, noise_level=25):
            init_image = load_image(input_refer_image_path)
            init_image = init_image.resize((self.width, self.height))

            # ノイズ付与画像を作成するメソッド
            def make_noise_condition(image, noise_level=25):
                image_array = np.array(image)
                noise = np.random.normal(0, noise_level, image_array.shape)
                noisy_image = image_array + noise
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
                return Image.fromarray(noisy_image)

            noisy_image = make_noise_condition(init_image, noise_level)
            noisy_image.save(output_refer_image_path)



        if mode == "canny":
            prepare_canny(input_refer_image_path,output_refer_image_path, low_threshold = low_threshold, high_threshold = high_threshold)
        elif mode == "tile":
            init_image = load_image(input_refer_image_path)
            init_image = init_image.resize((self.width, self.height))
            init_image.save(output_refer_image_path)
        elif mode == "depth":
            prepare_depthmap(input_refer_image_path,output_refer_image_path)
        elif mode == "zoe_depth":
            prepare_zoe_depthmap(input_refer_image_path,output_refer_image_path)
        elif mode == "blur":
            prepare_blur(input_refer_image_path, output_refer_image_path, blur_radius=5)
        elif mode == "openpose" or mode == "openpose_face" or mode == "openpose_faceonly" or mode == "openpose_full":
            prepare_openpose(input_refer_image_path,output_refer_image_path, mode = mode)
        elif mode == "gray":
            prepare_grayscale(input_refer_image_path, output_refer_image_path)
        elif mode == "lq":
            prepare_noise(input_refer_image_path, output_refer_image_path, noise_level=30)
        else:
            raise ValueError("control_mode is not set")


    def generate_image(self, prompt, neg_prompt = None, image_path = None, seed = None, controlnet_conditioning_scale = [1.0]):
        def decode_tensors(pipe, step, timestep, callback_kwargs):
            if self.save_latent_simple or self.save_predict_skip_x0:
                callback_kwargs = decode_tensors_simple(pipe, step, timestep, callback_kwargs)
            elif self.save_latent_overstep:
                callback_kwargs = decode_tensors_residual(pipe, step, timestep, callback_kwargs)
            else:
                raise ValueError("self.save_predict_skip_x0 or save_latent_simple or save_latent_overstep must be set or 'save_latent_approximation = False'")
            return callback_kwargs


        def decode_tensors_simple(pipe, step, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            if self.save_predict_skip_x0:
                skip_x0 = callback_kwargs["skip_x0"]
            else:
                skip_x0 = None
            imege = None
            prefix = None
            if not self.save_predict_skip_x0:
                prefix = "latents"
                if self.save_latent_simple and not self.save_latent_approximation:
                    image = latents_to_rgb_vae(latents,pipe)
                elif self.save_latent_approximation:
                    image = latents_to_rgb_approximation(latents,pipe)
                else:
                    raise ValueError("save_latent_simple or save_latent_approximation is not set")
            else:
                prefix = "predicted_x0"
                image = latents_to_rgb_vae(skip_x0,pipe)

            gettime = time.time()
            formatted_time_human_readable = time.strftime("%Y%m%d_%H%M%S", time.localtime(gettime))
            image.save(f"./outputs/{prefix}_{formatted_time_human_readable}_{step}_{timestep}.png")

            return callback_kwargs

        def decode_tensors_residual(pipe, step, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            if step > 0:
                residual = latents - self.last_latents
                goal = self.last_latents + residual * ((self.last_timestep) / (self.last_timestep - timestep))
                #print( ((self.last_timestep) / (self.last_timestep - timestep)))
            else:
                goal = latents

            if self.save_latent_overstep and not self.save_latent_approximation:
                image = latents_to_rgb_vae(goal,pipe)
            elif self.save_latent_approximation:
                image = latents_to_rgb_approximation(goal,pipe)
            else:
                raise ValueError("save_latent_simple or save_latent_approximation is not set")

            gettime = time.time()
            formatted_time_human_readable = time.strftime("%Y%m%d_%H%M%S", time.localtime(gettime))
            image.save(f"./outputs/latent_{formatted_time_human_readable}_{step}_{timestep}.png")

            self.last_latents = latents
            self.last_step = step
            self.last_timestep = timestep

            if timestep == 0:
                self.last_latents = None
                self.last_step = -1
                self.last_timestep = 100

            return callback_kwargs

        def latents_to_rgb_vae(latents,pipe):

            pipe.upcast_vae()
            latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
            images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            images = pipe.image_processor.postprocess(images, output_type='pil')
            pipe.vae.to(dtype=torch.float16)

            return StableDiffusionXLPipelineOutput(images=images).images[0]

        def latents_to_rgb_approximation(latents, pipe):
            weights = (
                (60, -60, 25, -70),
                (60,  -5, 15, -50),
                (60,  10, -5, -35)
            )

            weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
            biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
            rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
            image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
            image_array = image_array.transpose(1, 2, 0)  # Change the order of dimensions

            return Image.fromarray(image_array)


        def load_image_path(image_path):
            # もし image_path が str ならファイルパスとして画像を読み込む
            if isinstance(image_path, str):
                image = load_image(image_path)
                print("Image loaded from file path.")
            # もし image_path が PIL イメージならそのまま使用
            elif isinstance(image_path, Image.Image):
                image = image_path
                print("PIL Image object provided.")
            # もし image_path が Torch テンソルならそのまま使用
            elif isinstance(image_path, torch.Tensor):
                image = image_path.unsqueeze(0)
                image = image.permute(0, 3, 1, 2)
                image = image/255.0
                print("Torch Tensor object provided.")
            else:
                raise TypeError("Unsupported type. Provide a file path, PIL Image, or Torch Tensor.")

            return image

        def find_file_with_extension(image_path, i):
            # パターンに一致するファイルを検索
            file_pattern = f"{image_path}/{i}.*"
            matching_files = glob.glob(file_pattern)

            # マッチするファイルが存在する場合、そのファイルのパスを返す
            if matching_files:
                # 例: ./image_path/0.png のような完全なファイルパスが取得される
                return matching_files[0]
            else:
                # マッチするファイルがない場合
                raise FileNotFoundError(f"No file found matching pattern: {file_pattern}")
                return None

        if seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)

        control_image_list = []
        if self.use_controlnet:
            print("use controlnet mode: ",self.control_modes_list)

            for i in self.control_modes_list:
                if i is not None:
                    control_image_name = find_file_with_extension(image_path, i)
                    control_image = load_image_path(control_image_name)
                    control_image_list.append(control_image)


        lora_weight_average = 0
        if self.LORA_FLAG:
            print("use LoRA")
            lora_weight_average = sum(self.lora_scale_list) / len(self.lora_scale_list)
            for word in self.lora_trigger_word_list:
                if (word is not None) and (word != "None"):
                    prompt = prompt + ", " + word

        image = None
        if self.use_callback:
            if self.LORA_FLAG:
                if self.use_controlnet:
                    image = self.base(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        image=control_image_list,
                        cfg_scale=self.cfg_scale,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        num_inference_steps=self.n_steps,
                        output_type=self.output_type,
                        width = self.width,
                        height = self.height,
                        generator=self.generator,
                        callback_on_step_end=decode_tensors,
                        callback_on_step_end_tensor_inputs=["latents"],#skip_x0
                        cross_attention_kwargs={"scale": lora_weight_average},
                        ).images[0]
                else:
                    image = self.base(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        cfg_scale=self.cfg_scale,
                        num_inference_steps=self.n_steps,
                        output_type=self.output_type,
                        width = self.width,
                        height = self.height,
                        generator=self.generator,
                        callback_on_step_end=decode_tensors,
                        callback_on_step_end_tensor_inputs=["latents"],
                        cross_attention_kwargs={"scale": lora_weight_average},
                        ).images[0]

            #LORAを利用しない場合
            else:
                if self.use_controlnet:
                    image = self.base(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        image=control_image_list,
                        cfg_scale=self.cfg_scale,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        num_inference_steps=self.n_steps,
                        output_type=self.output_type,
                        width = self.width,
                        height = self.height,
                        generator=self.generator,
                        callback_on_step_end=decode_tensors,
                        callback_on_step_end_tensor_inputs=["latents"],#skip_x0
                        ).images[0]

                else:
                    image = self.base(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        cfg_scale=self.cfg_scale,
                        num_inference_steps=self.n_steps,
                        output_type=self.output_type,
                        width = self.width,
                        height = self.height,
                        generator=self.generator,
                        callback_on_step_end=decode_tensors,
                        callback_on_step_end_tensor_inputs=["latents"],
                        ).images[0]

        #latentを保存しない場合
        else:
            if self.LORA_FLAG:
                if self.use_controlnet:
                    image = self.base(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        image=control_image_list,
                        cfg_scale=self.cfg_scale,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        num_inference_steps=self.n_steps,
                        output_type=self.output_type,
                        width = self.width,
                        height = self.height,
                        generator=self.generator,
                        cross_attention_kwargs={"scale": lora_weight_average},
                        ).images[0]

                else:
                    image = self.base(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        cfg_scale=self.cfg_scale,
                        num_inference_steps=self.n_steps,
                        output_type=self.output_type,
                        width = self.width,
                        height = self.height,
                        generator=self.generator,
                        cross_attention_kwargs={"scale": lora_weight_average},
                        ).images[0]

            # LORAを利用しない場合
            else:
                if self.use_controlnet:
                    image = self.base(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        image=control_image_list,
                        cfg_scale=self.cfg_scale,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        num_inference_steps=self.n_steps,
                        output_type=self.output_type,
                        width = self.width,
                        height = self.height,
                        generator=self.generator,
                        ).images[0]

                else:
                    image = self.base(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        cfg_scale=self.cfg_scale,
                        num_inference_steps=self.n_steps,
                        output_type=self.output_type,
                        width = self.width,
                        height = self.height,
                        generator=self.generator,
                        ).images[0]


        return image