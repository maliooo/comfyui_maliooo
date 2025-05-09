import comfy.model_management as model_management
# /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/utils.py
# /workspace/ComfyUI/custom_nodes/comfyui_maliooo/nodes/malio_aux.py
import os
import sys
from PIL import Image
from .utils import  pil2tensor, tensor2pil
import cv2
import numpy as np
# custom_nodes_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.join(custom_nodes_path, "comfyui_controlnet_aux/src"))
from .comfyui_controlnet_aux.src.custom_controlnet_aux.oneformer import OneformerSegmentor

class Malio_OneFormer_ADE20K_SemSegPreprocessor:
    """"只处理一张图"""

    def __init__(self):
        try:
            self.model = OneformerSegmentor.from_pretrained(filename="250_16_swin_l_oneformer_ade20k_160k.pth")
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
            device = model_management.get_torch_device()
            offload_device = model_management.unet_offload_device()

            self.model = self.model.to(device=device)
            new_image = tensor2pil(image)[0]
            image_cv2 = cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)
            detected_map, label_text_list, label_text_ratio = self.model(image_cv2, output_type="pil", is_text=False, detect_resolution=resolution)   # 这里会下采样4倍
            
            self.model.to(offload_device)  # 释放cuda资源
            model_management.soft_empty_cache()

            return (pil2tensor(detected_map), label_text_list, label_text_ratio,)
        except Exception as e:
            print(f"调用OneFormer进行语义分割出错，错误原因:{e},  出错文件为{__file__}")
            return (image, [], [])
    
