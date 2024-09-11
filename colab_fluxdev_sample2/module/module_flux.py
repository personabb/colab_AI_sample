from diffusers import FluxControlNetPipeline
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from diffusers import FluxTransformer2DModel, FluxControlNetModel
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FluxControlNetModel, FluxMultiControlNetModel

from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
import torch
from diffusers.schedulers import DPMSolverMultistepScheduler, LCMScheduler, EulerAncestralDiscreteScheduler, FlowMatchEulerDiscreteScheduler
from controlnet_aux.processor import Processor

import glob

import os
import configparser
# ファイルの存在チェック用モジュール
import errno
import cv2
from PIL import Image, ImageFilter
import time
import numpy as np


class FLUXconfig:
    def __init__(self, config_ini_path = './configs/config.ini'):
        # iniファイルの読み込み
        self.config_ini = configparser.ConfigParser()

        # 指定したiniファイルが存在しない場合、エラー発生
        if not os.path.exists(config_ini_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)

        self.config_ini.read(config_ini_path, encoding='utf-8')
        FLUX_items = self.config_ini.items('FLUX')
        self.FLUX_config_dict = dict(FLUX_items)

class FLUX:
    def __init__(self,device = None, config_ini_path = './configs/config.ini'):

        FLUX_config = FLUXconfig(config_ini_path = config_ini_path)
        config_dict = FLUX_config.FLUX_config_dict


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

        self.model_mode = config_dict.get("model_mode", "None")
        if self.model_mode == "t2i":
            self.model_mode = "t2i"
            print("t2i mode")
        elif self.model_mode == "i2i":
            self.model_mode = "i2i"
            print("i2i mode")
        else:
            raise ValueError("model_mode is only 't2i' or 'i2i'.")
        
        self.strength = config_dict.get("strength", "None")
        if self.strength == "None":
            self.strength = None
            if self.model_mode == "i2i":
                raise ValueError("strength is not set")
        else:
            self.strength = float(self.strength)

        self.controlnet_path = config_dict.get("controlnet_path", "None")
        if not self.use_controlnet and self.controlnet_path == "None":
            raise ValueError("controlnet_path is not set")

        self.control_modes_number_list = []
        control_mode_dict = {
            "canny":0,
            "tile":1,
            "depth":2,
            "blur":3,
            "pose":4,
            "gray":5,
            "lq":6
        }
        for i in range(7):
            if config_dict.get(f"control_mode{i}", "None") != "None":
                self.control_modes_number_list.append(control_mode_dict[config_dict[f"control_mode{i}"]])

        self.from_single_file = config_dict.get("from_single_file", "None")
        self.SINGLE_FILE_FLAG = True
        if self.from_single_file != "True":
            self.from_single_file = None
            self.SINGLE_FILE_FLAG = False


        self.base_model_path = config_dict["base_model_path"]


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

        self.use_callback = False
        if self.save_latent_simple or self.save_latent_overstep:
            self.use_callback = True

        if (self.save_latent_simple and self.save_latent_overstep):
            raise ValueError("save_latent_simple and save_latent_overstep cannot be set at the same time")

        if self.model_mode == "t2i":
            if self.LORA_FLAG:
                self.base = self.preprepare_model_withLoRAandQuantize()
            else:
                self.base = self.preprepare_model()
        elif self.model_mode == "i2i":
            if self.LORA_FLAG:
                self.base = self.preprepare_img2img_model_withLoRAandQuantize()
            else:
                self.base = self.preprepare_img2img_model()
        else:
            raise ValueError("model_mode is only 't2i' or 'i2i'.")

    def preprepare_img2img_model(self):
        if not self.SINGLE_FILE_FLAG:
            transformer = FluxTransformer2DModel.from_pretrained(
                self.base_model_path,
                subfolder="transformer",
                torch_dtype=torch.float16
            )
            print("transformer quantizing")
            quantize(transformer, weights=qfloat8)
            freeze(transformer)
            print("loaded transformer")

            text_encoder_2 = T5EncoderModel.from_pretrained(
                self.base_model_path,
                subfolder="text_encoder_2",
                torch_dtype=torch.float16
            )
            print("text_encoder_2 quantizing")
            quantize(text_encoder_2, weights=qfloat8)
            freeze(text_encoder_2)
            print("loaded text_encoder_2")

            base = FluxImg2ImgPipeline.from_pretrained(
                self.base_model_path, 
                transformer=transformer, 
                text_encoder_2=text_encoder_2,
                torch_dtype=torch.float16
                )

            print("loaded base model")
        else:
            base = FluxImg2ImgPipeline.from_pretrained(self.base_model_path, torch_dtype=torch.float16)
            print("transformer quantizing")
            quantize(base.transformer, weights=qfloat8)
            freeze(base.transformer)
            print("text_encoder_2 quantizing")
            quantize(base.text_encoder_2, weights=qfloat8)
            freeze(base.text_encoder_2)

            print("loaded base model")

        print("cpu offloading")
        base.enable_model_cpu_offload()

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

    def preprepare_img2img_model_withLoRAandQuantize(self):
        if not self.SINGLE_FILE_FLAG:
            base = FluxImg2ImgPipeline.from_pretrained(
                self.base_model_path, 
                torch_dtype=torch.float16
                )

            print("loaded base model")
        else:
            base = FluxImg2ImgPipeline.from_pretrained(self.base_model_path, torch_dtype=torch.float16)

            print("loaded base model")

        print("cpu offloading")
        base.enable_model_cpu_offload()

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


    def preprepare_model(self, controlnet_path = None):
        if controlnet_path is not None:
            self.controlnet_path = controlnet_path

        #重みのサイズが大きい順に量子化して、RAMの最大使用量を減らす
        if self.use_controlnet:
            controlnet = FluxControlNetModel.from_pretrained(self.controlnet_path, torch_dtype=torch.float16)
            controlnet = FluxMultiControlNetModel([controlnet])
            print("controlnet quantizing")
            quantize(controlnet, weights=qfloat8)
            freeze(controlnet)
            print("loaded controlnet")

            if not self.SINGLE_FILE_FLAG:
                transformer = FluxTransformer2DModel.from_pretrained(
                    self.base_model_path,
                    subfolder="transformer",
                    torch_dtype=torch.float16
                )
                print("transformer quantizing")
                quantize(transformer, weights=qfloat8)
                freeze(transformer)
                print("loaded transformer")

                text_encoder_2 = T5EncoderModel.from_pretrained(
                    self.base_model_path,
                    subfolder="text_encoder_2",
                    torch_dtype=torch.float16
                )
                print("text_encoder_2 quantizing")
                quantize(text_encoder_2, weights=qfloat8)
                freeze(text_encoder_2)
                print("loaded text_encoder_2")

                base = FluxControlNetPipeline.from_pretrained(
                    self.base_model_path, 
                    transformer=transformer, 
                    text_encoder_2=text_encoder_2,
                    controlnet=controlnet, 
                    torch_dtype=torch.float16
                    )
                print("loaded base model")
            else:
                pipe = FluxPipeline.from_pretrained(self.base_model_path, torch_dtype=torch.float16)

                print("transformer quantizing")
                quantize(pipe.transformer, weights=qfloat8)
                freeze(pipe.transformer)
                print("text_encoder_2 quantizing")
                quantize(pipe.text_encoder_2, weights=qfloat8)
                freeze(pipe.text_encoder_2)

                base = FluxControlNetPipeline(controlnet = controlnet, **pipe.components)
                print("loaded base model")

        else:
            controlnet = None
            if not self.SINGLE_FILE_FLAG:
                transformer = FluxTransformer2DModel.from_pretrained(
                    self.base_model_path,
                    subfolder="transformer",
                    torch_dtype=torch.float16
                )
                print("transformer quantizing")
                quantize(transformer, weights=qfloat8)
                freeze(transformer)
                print("loaded transformer")

                text_encoder_2 = T5EncoderModel.from_pretrained(
                    self.base_model_path,
                    subfolder="text_encoder_2",
                    torch_dtype=torch.float16
                )
                print("text_encoder_2 quantizing")
                quantize(text_encoder_2, weights=qfloat8)
                freeze(text_encoder_2)
                print("loaded text_encoder_2")

                base = FluxPipeline.from_pretrained(
                    self.base_model_path, 
                    transformer=transformer, 
                    text_encoder_2=text_encoder_2,
                    torch_dtype=torch.float16
                    )
                print("loaded base model")

            else:
                base = FluxPipeline.from_pretrained(self.base_model_path, torch_dtype=torch.float16)

                print("transformer quantizing")
                quantize(base.transformer, weights=qfloat8)
                freeze(base.transformer)
                print("text_encoder_2 quantizing")
                quantize(base.text_encoder_2, weights=qfloat8)
                freeze(base.text_encoder_2)

                print("loaded base model")


        print("cpu offloading")
        base.enable_model_cpu_offload()


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

    def preprepare_model_withLoRAandQuantize(self, controlnet_path = None):
        if controlnet_path is not None:
            self.controlnet_path = controlnet_path

        #重みのサイズが大きい順に量子化して、RAMの最大使用量を減らす
        if not self.SINGLE_FILE_FLAG:
            transformer = FluxTransformer2DModel.from_pretrained(
                self.base_model_path,
                subfolder="transformer",
                torch_dtype=torch.float16
            )

            text_encoder_2 = T5EncoderModel.from_pretrained(
                self.base_model_path,
                subfolder="text_encoder_2",
                torch_dtype=torch.float16
            )

            base = FluxPipeline.from_pretrained(
                self.base_model_path, 
                transformer=transformer, 
                text_encoder_2=text_encoder_2,
                torch_dtype=torch.float16
                )

            print("loaded base model")
        else:
            base = FluxPipeline.from_pretrained(self.base_model_path, torch_dtype=torch.float16)

            print("loaded base model")


        if self.use_controlnet:
            controlnet = FluxControlNetModel.from_pretrained(self.controlnet_path, torch_dtype=torch.float16)
            controlnet = FluxMultiControlNetModel([controlnet])
            print("loaded controlnet")
            base = FluxControlNetPipeline(controlnet = controlnet, **base.components)

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

            print("finish lora settings",f"{self.lora_repo_list = },{lora_adapter_weights_list = }")

        print("cpu offloading")
        base.enable_model_cpu_offload()

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

            # ファイル名の数字部分でソート（小さい順）
            image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

            # 出力用のフォルダパスに変更したリストを作成
            output_image_paths = [
                os.path.join(output_refer_image_folder, os.path.basename(path))
                for path in image_paths
            ]

            return image_paths, output_image_paths

         #output_refer_image_folderが存在しない場合、作成する
        if not os.path.exists(output_refer_image_folder):
            os.makedirs(output_refer_image_folder)

        input_paths, output_paths = get_image_paths_sorted_by_filename_number(input_refer_image_folder, output_refer_image_folder)

        for input_refer_image_path, output_refer_image_path in zip(input_paths, output_paths):
            mode = int(os.path.splitext(os.path.basename(input_refer_image_path))[0])
            self.prepare_referimage(input_refer_image_path,output_refer_image_path, low_threshold = low_threshold, high_threshold = high_threshold, noise_level=noise_level, blur_radius=blur_radius, mode = mode)




    def prepare_referimage(self,input_refer_image_path,output_refer_image_path, low_threshold = 100, high_threshold = 200, noise_level=25, blur_radius=5, mode = 0):
        #mode = canny (0), tile (1), depth (2), blur (3), pose (4), gray (5), low quality (6).

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



        if mode == 0:
            prepare_canny(input_refer_image_path,output_refer_image_path, low_threshold = low_threshold, high_threshold = high_threshold)
        elif mode == 1:
            init_image = load_image(input_refer_image_path)
            init_image.save(output_refer_image_path)
        elif mode == 2:
            prepare_depthmap(input_refer_image_path,output_refer_image_path)
        elif mode == 3:
            prepare_blur(input_refer_image_path, output_refer_image_path, blur_radius=5)
        elif mode == 4:
            prepare_openpose(input_refer_image_path,output_refer_image_path, mode = "openpose_full")
        elif mode == 5:
            prepare_grayscale(input_refer_image_path, output_refer_image_path)
        elif mode == 6:
            prepare_noise(input_refer_image_path, output_refer_image_path, noise_level=30)
        else:
            raise ValueError("control_mode is not set")


    def generate_image(self, prompt, neg_prompt = None, c_image_path = None, i_image_path = None, seed = None, controlnet_conditioning_scale = [1.0], temp_strength = None):
        def decode_tensors(pipe, step, timestep, callback_kwargs):
            if self.save_latent_simple:
                callback_kwargs = decode_tensors_simple(pipe, step, timestep, callback_kwargs)
            elif self.save_latent_overstep:
                callback_kwargs = decode_tensors_residual(pipe, step, timestep, callback_kwargs)
            else:
                raise ValueError("save_latent_simple or save_latent_overstep must be set")
            return callback_kwargs


        def decode_tensors_simple(pipe, step, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            imege = None
            if self.save_latent_simple:
                image = latents_to_rgb_vae(latents,pipe)
            else:
                raise ValueError("save_latent_simple is not set")
            gettime = time.time()
            formatted_time_human_readable = time.strftime("%Y%m%d_%H%M%S", time.localtime(gettime))
            image.save(f"./outputs/latent_{formatted_time_human_readable}_{step}_{timestep}.png")

            return callback_kwargs

        def decode_tensors_residual(pipe, step, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            if step > 0:
                residual = latents - self.last_latents
                goal = self.last_latents + residual * ((self.last_timestep) / (self.last_timestep - timestep))
                #print( ((self.last_timestep) / (self.last_timestep - timestep)))
            else:
                goal = latents

            if self.save_latent_overstep:
                image = latents_to_rgb_vae(goal,pipe)
            else:
                raise ValueError("save_latent_overstep is not set")

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

            latents = pipe._unpack_latents(latents, self.height, self.width, pipe.vae_scale_factor)
            latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

            image = pipe.vae.decode(latents, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type=self.output_type)

            return FluxPipelineOutput(images=image).images[0]


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
        if temp_strength is not None:
            self.strength = temp_strength

        control_image_list = []
        if self.use_controlnet and self.model_mode == "t2i":
            print("use controlnet mode: ",self.control_modes_number_list)
            for i in self.control_modes_number_list:
                if c_image_path is None:
                    raise ValueError("when use controlnet ,control_image_path must be set")
                control_image_name = load_image_path(find_file_with_extension(c_image_path, i))
                control_image = load_image_path(control_image_name)
                control_image_list.append(control_image)
        
        init_image = None
        if self.model_mode == "i2i":
            print("use i2i mode")
            if i_image_path is None:
                raise ValueError("when use i2i mode, init_image_path must be set")
            init_image = load_image_path(i_image_path)


        lora_weight_average = 0
        if self.LORA_FLAG:
            print("use LoRA")
            lora_weight_average = sum(self.lora_scale_list) / len(self.lora_scale_list)
            for word in self.lora_trigger_word_list:
                if (word is not None) and (word != "None"):
                    prompt = prompt + ", " + word

        image = None
        if self.model_mode == "i2i":
            if self.use_callback:
                if self.LORA_FLAG:
                    image = self.base(
                        prompt=prompt,
                        image=init_image, 
                        strength= self.strength,
                        guidance_scale=self.cfg_scale,
                        num_inference_steps=self.n_steps,
                        output_type=self.output_type,
                        width = self.width,
                        height = self.height,
                        generator=self.generator,
                        callback_on_step_end=decode_tensors,
                        callback_on_step_end_tensor_inputs=["latents"],
                        joint_attention_kwargs={"scale": lora_weight_average},
                        ).images[0]
                #LORAを利用しない場合
                else:
                    image = self.base(
                        prompt=prompt,
                        image=init_image, 
                        strength= self.strength,
                        guidance_scale=self.cfg_scale,
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
                    image = self.base(
                        prompt=prompt,
                        image=init_image, 
                        strength= self.strength,
                        guidance_scale=self.cfg_scale,
                        num_inference_steps=self.n_steps,
                        output_type=self.output_type,
                        width = self.width,
                        height = self.height,
                        generator=self.generator,
                        joint_attention_kwargs={"scale": lora_weight_average},
                        ).images[0]
                # LORAを利用しない場合
                else:
                    image = self.base(
                        prompt=prompt,
                        image=init_image, 
                        strength= self.strength,
                        guidance_scale=self.cfg_scale,
                        num_inference_steps=self.n_steps,
                        output_type=self.output_type,
                        width = self.width,
                        height = self.height,
                        generator=self.generator
                        ).images[0]

        #t2iモードの場合
        elif self.model_mode == "t2i":
            if self.use_callback:
                if self.LORA_FLAG:
                    if self.use_controlnet:
                        image = self.base(
                            prompt=prompt,
                            control_image=control_image_list,
                            control_mode=self.control_modes_number_list,
                            guidance_scale=self.cfg_scale,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            num_inference_steps=self.n_steps,
                            output_type=self.output_type,
                            width = self.width,
                            height = self.height,
                            generator=self.generator,
                            callback_on_step_end=decode_tensors,
                            callback_on_step_end_tensor_inputs=["latents"],
                            joint_attention_kwargs={"scale": lora_weight_average},
                            ).images[0]
                    else:
                        image = self.base(
                            prompt=prompt,
                            guidance_scale=self.cfg_scale,
                            num_inference_steps=self.n_steps,
                            output_type=self.output_type,
                            width = self.width,
                            height = self.height,
                            generator=self.generator,
                            callback_on_step_end=decode_tensors,
                            callback_on_step_end_tensor_inputs=["latents"],
                            joint_attention_kwargs={"scale": lora_weight_average},
                            ).images[0]
                #LORAを利用しない場合
                else:
                    if self.use_controlnet:
                        image = self.base(
                            prompt=prompt,
                            control_image=control_image_list,
                            control_mode=self.control_modes_number_list,
                            guidance_scale=self.cfg_scale,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            num_inference_steps=self.n_steps,
                            output_type=self.output_type,
                            width = self.width,
                            height = self.height,
                            callback_on_step_end=decode_tensors,
                            callback_on_step_end_tensor_inputs=["latents"],
                            generator=self.generator
                            ).images[0]
                    else:
                        image = self.base(
                            prompt=prompt,
                            guidance_scale=self.cfg_scale,
                            num_inference_steps=self.n_steps,
                            output_type=self.output_type,
                            width = self.width,
                            height = self.height,
                            callback_on_step_end=decode_tensors,
                            callback_on_step_end_tensor_inputs=["latents"],
                            generator=self.generator
                            ).images[0]
            #latentを保存しない場合
            else:
                if self.LORA_FLAG:
                    if self.use_controlnet:
                        image = self.base(
                            prompt=prompt,
                            control_image=control_image_list,
                            control_mode=self.control_modes_number_list,
                            guidance_scale=self.cfg_scale,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            num_inference_steps=self.n_steps,
                            output_type=self.output_type,
                            width = self.width,
                            height = self.height,
                            generator=self.generator,
                            joint_attention_kwargs={"scale": lora_weight_average},
                            ).images[0]
                    else:
                        image = self.base(
                            prompt=prompt,
                            guidance_scale=self.cfg_scale,
                            num_inference_steps=self.n_steps,
                            output_type=self.output_type,
                            width = self.width,
                            height = self.height,
                            generator=self.generator,
                            joint_attention_kwargs={"scale": lora_weight_average},
                            ).images[0]

                # LORAを利用しない場合
                else:
                    if self.use_controlnet:
                        image = self.base(
                            prompt=prompt,
                            control_image=control_image_list,
                            control_mode=self.control_modes_number_list,
                            guidance_scale=self.cfg_scale,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            num_inference_steps=self.n_steps,
                            output_type=self.output_type,
                            width = self.width,
                            height = self.height,
                            generator=self.generator
                            ).images[0]
                    else:
                        image = self.base(
                            prompt=prompt,
                            guidance_scale=self.cfg_scale,
                            num_inference_steps=self.n_steps,
                            output_type=self.output_type,
                            width = self.width,
                            height = self.height,
                            generator=self.generator
                            ).images[0]
        else:
            raise ValueError("model_mode is only 't2i' or 'i2i'.")


        return image