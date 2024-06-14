import math
from .components.fields import Field
from .components.sizes import get_image_size
from comfy.utils import common_upscale
import sys
import os
from PIL import Image, ImageOps
import numpy as np
import torch
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import VolcEngineMaasLLM
import requests
from .utils import pil2tensor

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
                        "default": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Example.jpg/800px-Example.jpg"
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE","STRING","STRING","STRING")
    RETURN_NAMES = ("IMAGE", "positive_prompt", "negative_prompt", "params" )
    FUNCTION = "load"
    CATEGORY = "mtb/IO"

    def load(self, url):
        # get the image from the url
        image = Image.open(requests.get(url, stream=True).raw)
        image = ImageOps.exif_transpose(image)
        
        info = None
        positive_text = None
        negative_text = None
        try:
            info = image.info["parameters"].split("\n")
            positive_text = info[0]
            negative_text = info[1]
            params = info[2]
        except Exception as e:
            print(f"图片提取info信息出错，Maliooo_LoadImageFromUrl: {e}")
        
        if info and len(info) == 3:
            return (pil2tensor(image), positive_text, negative_text, params)
        else:
            return (pil2tensor(image), None, None, None)




        
        


