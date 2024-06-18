import folder_paths
import comfy
import node_helpers
import torch

import os
import sys

from PIL import Image, ImageOps, ImageSequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

# import comfy.diffusers_load
# import comfy.samplers
# import comfy.sample
# import comfy.sd
# import comfy.utils
# import comfy.controlnet

# import comfy.clip_vision

# import comfy.model_management




class Malio_CheckpointLoaderSimple:
    """comfyui 原始的checkpoint加载器，用于加载checkpoint文件，返回模型、CLIP和VAE。"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            },
            "optional": {
                "override": ("STRING", {"default": "", 'multiline': False, "forceInput": False, "dynamicPrompts": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "model_path")
    FUNCTION = "load_checkpoint"

    CATEGORY = "🐼malio/loaders"

    def load_checkpoint(self, ckpt_name, override):

        if override:  # 如果有override，就用override
            checkpoint_names = folder_paths.get_filename_list("checkpoints")  # 本地的controlnet文件
            if isinstance(override, str):
                file_suffix = override.split(".")[-1]
                if file_suffix not in ["safetensors", "ckpt", "pth"]:
                    for _suffix in ["safetensors", "ckpt", "pth"]:
                        if (override + "." + _suffix) in checkpoint_names:
                            ckpt_name = override + "." + _suffix
                            print(f"override覆盖模型为: {ckpt_name}")
                            break
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return (out[0], out[1], out[2], ckpt_path)


class Malio_LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "🐼malio/image"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "image_name", "image_info")
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        
        img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # 获得图片名
        image_name = os.path.basename(image_path)

        image_info = ""
        try:
            image_info = img.info["parameters"].strip()
        except Exception as e:
            print(f"图片提取info信息出错，Malio_LoadImage: {e}")

        return (output_image, output_mask, image_name, image_info)