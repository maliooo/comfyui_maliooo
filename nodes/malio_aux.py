import comfy.model_management as model_management
# /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/utils.py
# /workspace/ComfyUI/custom_nodes/comfyui_maliooo/nodes/malio_aux.py
import os
import sys
from PIL import Image
from .utils import  pil2tensor, tensor2pil, pil2base64
import cv2
import traceback
import numpy as np
# custom_nodes_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.join(custom_nodes_path, "comfyui_controlnet_aux/src"))
import requests
import base64
import io
import torch
import time
import json
from pathlib import Path
import base64
import io

# from .comfyui_controlnet_aux.src.custom_controlnet_aux.oneformer import OneformerSegmentor

# class Malio_OneFormer_ADE20K_SemSegPreprocessor:
#     """"只处理一张图"""

#     def __init__(self):
#         try:
#             self.model = OneformerSegmentor.from_pretrained(filename="250_16_swin_l_oneformer_ade20k_160k.pth")
#         except Exception as e:
#             print(f"加载OneFormer模型出错，错误原因:{e},  出错文件为{__file__}")
#             self.model = None
#             raise e

#     @classmethod
#     def INPUT_TYPES(s):
#         MAX_RESOLUTION=16384
#         def RESOLUTION(default=512, min=64, max=MAX_RESOLUTION, step=64): 
#             return ("INT", dict(default=default, min=min, max=max, step=step))
        
#         return {
#             "required": {
#                 "image":("IMAGE",),
#                 "resolution": RESOLUTION(),
#             }
#         }

#     RETURN_TYPES = ("IMAGE", "LIST", "LIST")
#     RETURN_NAMES = ("image", "tag_list", "radio_list")
#     FUNCTION = "semantic_segmentate"


#     def semantic_segmentate(self, image, resolution=512):
#         try:
#             device = model_management.get_torch_device()
#             offload_device = model_management.unet_offload_device()

#             self.model = self.model.to(device=device)
#             new_image = tensor2pil(image)[0]
#             image_cv2 = cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)
#             detected_map, label_text_list, label_text_ratio = self.model(image_cv2, output_type="pil", is_text=False, detect_resolution=resolution)   # 这里会下采样4倍
            
#             self.model.to(offload_device)  # 释放cuda资源
#             model_management.soft_empty_cache()

#             return (pil2tensor(detected_map), label_text_list, label_text_ratio,)
#         except Exception as e:
#             print(f"调用OneFormer进行语义分割出错，错误原因:{e},  出错文件为{__file__}")
#             traceback.print_exc()
#             return (image, [], [])



class Malio_OneFormer_ADE20K_SemSegPreprocessor:
    """"只处理一张图"""

    def __init__(self):
        try:
            # self.model = OneformerSegmentor.from_pretrained(filename="250_16_swin_l_oneformer_ade20k_160k.pth")
            pass
        except Exception as e:
            print(f"加载OneFormer模型出错，错误原因:{e},  出错文件为{__file__}")
            self.model = None
            raise e

    @classmethod
    def INPUT_TYPES(s):
        MAX_RESOLUTION=16384
        def RESOLUTION(default=512, min=64, max=MAX_RESOLUTION, step=64): 
            return ("INT", dict(default=default, min=min, max=max, step=step))
        
        return {
            "required": {
                "image":("IMAGE",),
                "resolution": RESOLUTION(),
            }
        }

    RETURN_TYPES = ("IMAGE", "LIST", "LIST")
    RETURN_NAMES = ("image", "tag_list", "radio_list")
    FUNCTION = "semantic_segmentate"


    def semantic_segmentate(self, image, resolution=512):
        try:
            # device = model_management.get_torch_device()
            # offload_device = model_management.unet_offload_device()

            # self.model = self.model.to(device=device)
            new_image = tensor2pil(image)[0]
            image_base64 = pil2base64(new_image)

            # 请求服务
            service_url = "http://172.31.48.6:28002/segment_image_base64"
            service_url = self.ip_map(service_url)

            resize_to = resolution
            show_text = False
            vis_image_tensor, masks_tensor, label_info_list, areas_ratios, num_labels = self.segment_base64_from_url(image_base64, service_url, resize_to, show_text)

            return (vis_image_tensor, label_info_list, areas_ratios,)
        except Exception as e:
            print(f"调用OneFormer进行语义分割出错，错误原因:{e},  出错文件为{__file__}")
            traceback.print_exc()
            return (image, [], [])
        
    def segment_base64_from_url(self, image_base64, service_url, resize_to, show_text = False):
        service_url = self.ip_map(service_url)  # 替换url
        if not image_base64:
            raise ValueError("image_base64 is required.")

        payload = {
            "image_url_or_base64": image_base64,
            "resize_to": resize_to,
            "show_text": show_text
        }

        try:
            response = requests.post(service_url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            vis_image_b64 = result.get('vis_image')
            if not vis_image_b64:
                raise ValueError("Response does not contain 'vis_image'")
            
            image_data = base64.b64decode(vis_image_b64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            vis_image_tensor = torch.from_numpy(image_np)[None,]

            mask_images_b64 = result.get('mask_images', [])
            masks = []
            for mask_b64 in mask_images_b64:
                mask_data = base64.b64decode(mask_b64)
                mask_image = Image.open(io.BytesIO(mask_data)).convert("L")
                mask_np = np.array(mask_image).astype(np.float32) / 255.0
                masks.append(torch.from_numpy(mask_np))
            
            if masks:
                masks_tensor = torch.stack(masks)
            else:
                h, w, _ = vis_image_tensor.shape[1:]
                masks_tensor = torch.zeros((0, h, w), dtype=torch.float32)

            end_time = time.time()
            assert len(result.get('label_info_list', [])) == len(result.get('areas_ratios', [])) == len(mask_images_b64), "label_info_list、areas_ratios、mask_images_b64长度不一致"

            return (
                vis_image_tensor, 
                masks_tensor, 
                result.get('label_info_list', []), 
                [float(item) for item in result.get('areas_ratios', [])], 
                len(result.get('label_info_list', []))
            )

        except requests.exceptions.RequestException as e:
            error_message = f"API request failed: {e}"
            print(f"[ERROR] {error_message}")
            raise RuntimeError(error_message)
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            print(f"[ERROR] {error_message}")
            raise RuntimeError(error_message)
        
    def ip_map(self, url):
        """"
        读取本地的ip_map.json文件，如果存在则替换ip地址。 如果是线上生成机器，则没有这个配置文件，这个ip则不替换。
        """
         # 1. 读取配置文件，替换ip地址
        florence_json_con_path = Path(__file__).parent / "ip_map.json"
        if os.path.exists(florence_json_con_path):
            with open(florence_json_con_path, "r") as f:
                config = json.load(f)
            florence_ip_map = config.get("florence_ip_map", {})
            if florence_ip_map:
                for source_ip, target_ip in florence_ip_map.items():
                    if source_ip in url:
                        url = url.replace(source_ip, target_ip)
                        print(f"[yellow]调用ip_map函数，替换ip地址: {source_ip} -> {target_ip}[/yellow]")

        return url