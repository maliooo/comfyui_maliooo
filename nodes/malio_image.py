import math
from .components.fields import Field
from .components.sizes import get_image_size
import os
from comfy.utils import common_upscale
import torch
import numpy as np
import sys
from PIL import Image, ImageOps, ImageSequence
import requests
from .utils import pil2tensor
import random
import node_helpers


scale_methods = ["nearest-exact", "bilinear", "bicubic", "bislerp", "area", "lanczos"]

class Malio_ImageScale_Side:
    """Image Scale Side 按边缩放图片"""
    scale_methods = scale_methods
    crop_methods = ["disabled", "center"]

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": Field.image(),
                "side_length": Field.int(),
                "side": Field.combo(["Longest", "Shortest", "Width", "Height"]),
                "upscale_method": Field.combo(cls.scale_methods),
                "crop": Field.combo(cls.crop_methods),
                "divisible_num": ("INT", {"default": 1, "min": 1, "max": 32}), # 整除数字
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "🐼malio/image/image_scale_side"

    def upscale(self, image, upscale_method, side_length: int, side: str, crop, divisible_num: int):
        samples = image.movedim(-1, 1)

        size = get_image_size(image)

        width_B = int(size[0])
        height_B = int(size[1])

        width = width_B
        height = height_B

        def determineSide(_side: str) -> tuple[int, int]:
            width, height = 0, 0
            if _side == "Width":
                heigh_ratio = height_B / width_B
                width = side_length
                height = heigh_ratio * width
            elif _side == "Height":
                width_ratio = width_B / height_B
                height = side_length
                width = width_ratio * height
            return width, height

        if side == "Longest":
            if width > height:
                width, height = determineSide("Width")
            else:
                width, height = determineSide("Height")
        elif side == "Shortest":
            if width < height:
                width, height = determineSide("Width")
            else:
                width, height = determineSide("Height")
        else:
            width, height = determineSide(side)

        width = math.ceil(width)
        height = math.ceil(height)

        # 整除数字
        width = width // divisible_num * divisible_num
        height = height // divisible_num * divisible_num


        res = common_upscale(samples, width, height, upscale_method, crop)
        res = res.movedim(1, -1)  # 移动维度
        return (res,)


class Malio_ImageScale_Side_Cascade:
    """Image Scale Side 按边缩放图片_Cascade专用"""
    scale_methods = scale_methods
    crop_methods = ["disabled", "center"]

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": Field.image(),
                "side_length": ("INT", {"default": 1024, "min": -sys.maxsize, "max": sys.maxsize, "step":1}),
                "side": Field.combo(["Longest", "Shortest", "Width", "Height"]),
                "upscale_method": Field.combo(cls.scale_methods),
                "crop": Field.combo(cls.crop_methods),
                "divisible_num": ("INT", {"default": 1, "min": 1, "max": 32}), # 整除数字
                "use_cascade": ("BOOLEAN", {"default": False}),  # 是否使用cascade比例缩放图片
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "🐼malio/image/image_scale_side_cascade"

    def upscale(self, image, upscale_method, side_length: int, side: str, crop, divisible_num: int, use_cascade:bool):
        samples = image.movedim(-1, 1)

        size = get_image_size(image)

        width_B = int(size[0])
        height_B = int(size[1])

        width = width_B
        height = height_B

        def determineSide(_side: str) -> tuple[int, int]:
            width, height = 0, 0
            if _side == "Width":
                heigh_ratio = height_B / width_B
                width = side_length
                height = heigh_ratio * width
            elif _side == "Height":
                width_ratio = width_B / height_B
                height = side_length
                width = width_ratio * height
            return width, height

        if side == "Longest":
            if width > height:
                width, height = determineSide("Width")
            else:
                width, height = determineSide("Height")
        elif side == "Shortest":
            if width < height:
                width, height = determineSide("Width")
            else:
                width, height = determineSide("Height")
        else:
            width, height = determineSide(side)

        width = math.ceil(width)
        height = math.ceil(height)

        if use_cascade is False:
            # 整除数字
            width = width // divisible_num * divisible_num
            height = height // divisible_num * divisible_num
        else:
            # 缩放到cascade的比例
            nums = [256,320,384,512,576,640,768,816,832,896,1024,1088,1152,1280,1344,1440,1536,1920]
            short_side = min(width, height)
            # 计算短边最接近的数值
            short_side = min(nums, key=lambda x:abs(x-short_side))
            if width > height:
                height = short_side
            else:
                width = short_side

        res = common_upscale(samples, width, height, upscale_method, crop)
        res = res.movedim(1, -1)  # 移动维度
        return (res,)


class Maliooo_LoadImageFromUrl:
    """Load an image from the given URL"""
    # 来源：\custom_nodes\comfy_mtb\nodes\image_processing.py

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "default": "https://image9.znzmo.com/256ded95-6af1-4874-8d22-73c6d8da9f1c.png"
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE","STRING","STRING","STRING","STRING")
    RETURN_NAMES = ("IMAGE","image_info","positive_prompt", "negative_prompt", "params" )
    FUNCTION = "load"
    CATEGORY = "🐼malio/image"

    def load(self, url):
        # get the image from the url
        image = Image.open(requests.get(url, stream=True).raw)
        image = ImageOps.exif_transpose(image)
        
        # 提取info信息
        info = None
        positive_text = None
        negative_text = None
        image_info = None
        try:
            image_info = image.info["parameters"].strip()
            info = image_info.split("\n")
            positive_text = info[0].strip()
            negative_text = info[1].strip()
            if negative_text.startswith("Negative prompt"):
                negative_text = negative_text[len("Negative prompt:"):].strip()
            params = info[2].strip()
        except Exception as e:
            print(f"图片提取info信息出错，Maliooo_LoadImageFromUrl: {e}")
        
        if image_info and len(info) == 3:
            return (pil2tensor(image),image_info, positive_text, negative_text, params)
        else:
            return (pil2tensor(image),None, None, None, None)



class Maliooo_LoadImageByPathSequence:
    """参考二狗，单张顺序随机加载图片"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dir_path": ("STRING", {}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "is_random": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ('IMAGE', "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_path", "image_name", "image_info")
    FUNCTION = "load_images_sequence"
    CATEGORY = "🐼malio/image"
    


    def load_images_sequence(self, dir_path, seed, is_random=False):
        """顺序加载文件夹中的图片，支持随机加载。 一张一张加载"""
        try:
            if os.path.isdir(dir_path):
                image_path_list = []
                for filename in os.listdir(dir_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        _img_path = os.path.join(dir_path, filename)
                        image_path_list.append(_img_path)
                
                image_path_list = sorted(image_path_list)
                if is_random:  # 随机旋转
                    selected_image_path = random.choice(image_path_list)
                else:
                    selected_image_path = image_path_list[seed % len(image_path_list)]
                

                image = Image.open(selected_image_path).convert('RGBA')
                image_info = ""
                try:
                    image_info = image.info["parameters"].strip()
                except Exception as e:
                    print(f"图片提取info信息出错，Maliooo_LoadImageByPathSequence: {e}")
                    
                image = ImageOps.exif_transpose(image)  # 旋转图片
                image_tensor = pil2tensor(image)
                    
                selected_image_name = os.path.basename(selected_image_path)

                
                return (image_tensor, selected_image_path, selected_image_name, image_info)
        
        except Exception as e:
            print(f"2🐕温馨提示处理图像时出错请重置节点：{e}")
            return (None, None, None, None)

        
class Maliooo_LoadImageByPath:
    """从文件路径加载图片"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {}),
            },
        }
    RETURN_TYPES = ('IMAGE', "MASK","STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask","image_path", "image_name", "image_info")
    FUNCTION = "load_image_by_file_path"
    CATEGORY = "🐼malio/image"
    


    def load_image_by_file_path(self, file_path):
        """从本地图片文件路径加载图片"""
        
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            if os.path.exists(file_path):
                
                img = node_helpers.pillow(Image.open, file_path)
        
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
                image_name = os.path.basename(file_path)

                image_info = ""
                try:
                    image_info = img.info["parameters"].strip()
                except Exception as e:
                    print(f"图片提取info信息出错, Malio_LoadImage: {e}")

                return (output_image, output_mask, file_path, image_name, image_info)

        print(f"文件路径错误：{file_path}")
        return (None, None, file_path, None, None)