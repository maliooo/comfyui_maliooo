import math
from .components.fields import Field
from .components.sizes import get_image_size
import os
from comfy.utils import common_upscale
from comfy.cli_args import args
import torch
import numpy as np
import sys
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import requests
from .utils import pil2tensor, get_comfyui_images, get_column_type
from .utils_image_info_secret import decrypt
import random
import node_helpers
import folder_paths
import json
import base64
import datetime
import time
import cv2
import io
from tqdm import tqdm
from urllib3.exceptions import InsecureRequestWarning
import pandas as pd

# 禁用不安全请求的警告
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


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
                "divisible_num": (
                    "INT",
                    {"default": 1, "min": 1, "max": 32},
                ),  # 整除数字
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "🐼malio/image/image_scale_side"

    def upscale(
        self,
        image,
        upscale_method,
        side_length: int,
        side: str,
        crop,
        divisible_num: int,
    ):
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
                "side_length": (
                    "INT",
                    {
                        "default": 1024,
                        "min": -sys.maxsize,
                        "max": sys.maxsize,
                        "step": 1,
                    },
                ),
                "side": Field.combo(["Longest", "Shortest", "Width", "Height"]),
                "upscale_method": Field.combo(cls.scale_methods),
                "crop": Field.combo(cls.crop_methods),
                "divisible_num": (
                    "INT",
                    {"default": 1, "min": 1, "max": 32},
                ),  # 整除数字
                "use_cascade": (
                    "BOOLEAN",
                    {"default": False},
                ),  # 是否使用cascade比例缩放图片
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "🐼malio/image/image_scale_side_cascade"

    def upscale(
        self,
        image,
        upscale_method,
        side_length: int,
        side: str,
        crop,
        divisible_num: int,
        use_cascade: bool,
    ):
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
            nums = [
                256,
                320,
                384,
                512,
                576,
                640,
                768,
                816,
                832,
                896,
                1024,
                1088,
                1152,
                1280,
                1344,
                1440,
                1536,
                1920,
            ]
            short_side = min(width, height)
            # 计算短边最接近的数值
            short_side = min(nums, key=lambda x: abs(x - short_side))
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

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "IMAGE",
        "image_info",
        "positive_prompt",
        "negative_prompt",
        "params",
        "image_name",
    )
    FUNCTION = "load"
    CATEGORY = "🐼malio/image"

    def load(self, url):
        # get the image from the url

        img = node_helpers.pillow(Image.open, requests.get(url, stream=True).raw)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # 获得图片名
        # https://image9.znzmo.com/256ded95-6af1-4874-8d22-73c6d8da9f1c.png
        image_name = url.split("/")[-1]

        image_info = ""
        positive_text = ""
        negative_text = ""
        params = ""
        try:
            image_info = img.info["parameters"].strip()
            info = image_info.split("\n")
            if len(info) != 3:
                info = decrypt(image_info)
                if str(info).startswith("parameters="):
                    info = info[len("parameters=") :]
                    image_info = info
                info = info.split("\n")
            positive_text = info[0].strip()
            negative_text = info[1].strip()
            if negative_text.startswith("Negative prompt"):
                negative_text = negative_text[len("Negative prompt:") :].strip()
            params = info[2].strip()
        except Exception as e:
            print(f"图片提取info信息出错, Malio_LoadImage: {e}")

        #  ("IMAGE","image_info","positive_prompt", "negative_prompt", "params" )

        return (
            output_image,
            image_info,
            positive_text,
            negative_text,
            params,
            image_name,
        )


class Maliooo_LoadImageByPathSequence:
    """参考二狗，单张顺序随机加载图片"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dir_path": ("STRING", {}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "is_random": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_path", "image_name", "image_info")
    FUNCTION = "load_images_sequence"
    CATEGORY = "🐼malio/image"

    def load_images_sequence(self, dir_path, seed, is_random=False):
        """顺序加载文件夹中的图片，支持随机加载。 一张一张加载"""
        try:
            if os.path.isdir(dir_path):
                image_path_list = []
                for filename in os.listdir(dir_path):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        _img_path = os.path.join(dir_path, filename)
                        image_path_list.append(_img_path)

                image_path_list = sorted(image_path_list)
                if is_random:  # 随机旋转
                    selected_image_path = random.choice(image_path_list)
                else:
                    selected_image_path = image_path_list[seed % len(image_path_list)]

                image = Image.open(selected_image_path).convert("RGB")
                image_info = ""
                try:
                    image_info = image.info["parameters"].strip()
                except Exception as e:
                    print(f"图片提取info信息出错，Maliooo_LoadImageByPathSequence: {e}")

                image = ImageOps.exif_transpose(image)  # 旋转图片
                image_tensor = pil2tensor(image)

                selected_image_name = os.path.basename(selected_image_path)

                return (
                    image_tensor,
                    selected_image_path,
                    selected_image_name,
                    image_info,
                )

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

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "image_path", "image_name", "image_info")
    FUNCTION = "load_image_by_file_path"
    CATEGORY = "🐼malio/image"

    def load_image_by_file_path(self, file_path):
        """从本地图片文件路径加载图片"""

        if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            if os.path.exists(file_path):
                img = node_helpers.pillow(Image.open, file_path)

                output_images = []
                output_masks = []
                w, h = None, None

                excluded_formats = ["MPO"]

                for i in ImageSequence.Iterator(img):
                    i = node_helpers.pillow(ImageOps.exif_transpose, i)

                    if i.mode == "I":
                        i = i.point(lambda i: i * (1 / 255))
                    image = i.convert("RGB")

                    if len(output_images) == 0:
                        w = image.size[0]
                        h = image.size[1]

                    if image.size[0] != w or image.size[1] != h:
                        continue

                    image = np.array(image).astype(np.float32) / 255.0
                    image = torch.from_numpy(image)[None,]
                    if "A" in i.getbands():
                        mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                        mask = 1.0 - torch.from_numpy(mask)
                    else:
                        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
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
                    info = image_info.split("\n")
                    if len(info) != 3:
                        info = decrypt(image_info)
                        if str(info).startswith("parameters="):
                            info = info[len("parameters=") :]
                            image_info = info
                except Exception as e:
                    print(f"图片提取info信息出错, Malio_LoadImage: {e}")

                return (output_image, output_mask, file_path, image_name, image_info)

        print(f"文件路径错误：{file_path}")
        return (None, None, file_path, None, None)

class Maliooo_LoadImageByCsv:
    """从csv文件加载图片"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "csv_file_path": ("STRING", {}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "info_1": ("STRING", {"default": ""}),
                "info_2": ("STRING", {"default": ""}),
                "info_3": ("STRING", {"default": ""}),
                "info_4": ("STRING", {"default": ""}),
                "info_5": ("STRING", {"default": ""}),
                "info_6": ("STRING", {"default": ""}),
                "去重字段_1": ("STRING", {"default": "", "tooltip": "去重字段如果写了风格参考图，则只会保留一个参考图为空"}),
                "去重字段_2": ("STRING", {"default": ""}),
                "去重字段_3": ("STRING", {"default": ""}),
                "不为空字段_1": ("STRING", {"default": ""}),
                "不为空字段_2": ("STRING", {"default": ""}),
                "不为空字段_3": ("STRING", {"default": ""}),
                "为空字段_1": ("STRING", {"default": ""}),
                "为空字段_2": ("STRING", {"default": ""}),
                "为空字段_3": ("STRING", {"default": ""}),
                "等于字段_1_字段名称": ("STRING", {"default": ""}),
                "等于字段_1_字段值": ("STRING", {"default": ""}),
                "等于字段_2_字段名称": ("STRING", {"default": ""}),
                "等于字段_2_字段值": ("STRING", {"default": ""}),
                "等于字段_3_字段名称": ("STRING", {"default": ""}),
                "等于字段_3_字段值": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("INT", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("csv数据条数", "columns", "info_1", "info_2", "info_3", "info_4", "info_5", "info_6")
    FUNCTION = "load_images_sequence"
    CATEGORY = "🐼malio/image"

    def load_images_sequence(self, csv_file_path, seed, 
                             info_1="", info_2="", info_3="", info_4="", info_5="", info_6="",
                            去重字段_1="", 去重字段_2="", 去重字段_3="", 
                            不为空字段_1="", 不为空字段_2="", 不为空字段_3="",
                            为空字段_1="", 为空字段_2="", 为空字段_3="",
                            等于字段_1_字段名称="", 等于字段_1_字段值="",
                            等于字段_2_字段名称="", 等于字段_2_字段值="",
                            等于字段_3_字段名称="", 等于字段_3_字段值=""
                             ):
        """顺序加载文件夹中的图片，支持随机加载。 一张一张加载"""
        # try:
        print(f"开始加载csv文件：{csv_file_path}")
        print(f"传入参数：{info_1}, {info_2}, {info_3}, {info_4}, {info_5}, {info_6}")
        import pandas as pd
        if not os.path.exists(csv_file_path):
            raise Exception(f"csv文件路径错误：{csv_file_path}")
        if csv_file_path.lower().endswith((".csv")):
            df = pd.read_csv(csv_file_path, header=0)
        elif csv_file_path.lower().endswith((".xlsx")):
            df = pd.read_excel(csv_file_path, header=0)
        else:
            raise Exception(f"csv文件路径错误：{csv_file_path}")
        print(f"过滤前csv数据条数：{len(df)}")
        columns = df.columns.tolist()
        print(f"csv文件列名：{columns}")
        duplicate_columns = [去重字段_1, 去重字段_2, 去重字段_3]
        not_null_columns = [不为空字段_1, 不为空字段_2, 不为空字段_3]
        null_columns = [为空字段_1, 为空字段_2, 为空字段_3]
        equal_columns = [(等于字段_1_字段名称, 等于字段_1_字段值), (等于字段_2_字段名称, 等于字段_2_字段值), (等于字段_3_字段名称, 等于字段_3_字段值)]
        
        for col in duplicate_columns:
            if col in columns and col != "":
                df = df.drop_duplicates(subset=[col], keep="first")
        for col in not_null_columns:
            if col in columns and col != "":
                df = df.dropna(subset=[col])
        for col in null_columns:
            if col in columns and col != "":
                df = df[df[col].isnull()]
        
        for col, value in equal_columns:
            if col in columns and col != "":
                if get_column_type(df[col]) == "字符串类型":
                    df = df[df[col] == value]
                elif get_column_type(df[col]) == "整数类型":
                    df = df[df[col] == int(value)]
                elif get_column_type(df[col]) == "浮点数类型":
                    df = df[df[col] == float(value)]
                elif get_column_type(df[col]) == "布尔类型":
                    df = df[df[col] == bool(value)]
                    
        total_num = len(df)
        print(f"过滤后csv数据条数：{total_num}")
        if total_num == 0:
            return (None, None, None, None, None, None, None, None)
        
        iloc = seed % total_num

        return_values = []
        
        args = [info_1, info_2, info_3, info_4, info_5, info_6]
        for para in args:
            print(f"参数：{para}")
            if para == "" or para not in columns or pd.isna(df.iloc[iloc][para]):
                return_values.append(None)
            else:
                return_values.append(str(df.iloc[iloc][para]))
        
        print(f"返回参数：{return_values}")
        return tuple([total_num] + [str(columns)] + return_values)

        # except Exception as e:
        #     print(f"加载csv文件出错，请检查文件路径：{e}")
        #     return (None, None, None, None, None, None, None, None)
            


class Maliooo_LoadImageByCsv_V2:
    """从csv文件加载图片"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "csv_file_path": ("STRING", {}),
            },
            "optional": {
                "获取同一个task_id的其他生成图片": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "key_task_id_字段名称": ("STRING", {"default": ""}),
                "key_输入底图url_字段名称": ("STRING", {"default": ""}),
                "key_输入风格参考图url_字段名称": ("STRING", {"default": ""}),
                "key_输入遮罩url_字段名称": ("STRING", {"default": ""}),
                "key_输入知末生成图url_字段名称": ("STRING", {"default": ""}),
                "info_1": ("STRING", {"default": ""}),
                "info_2": ("STRING", {"default": ""}),
                "info_3": ("STRING", {"default": ""}),
                "info_4": ("STRING", {"default": ""}),
                "info_5": ("STRING", {"default": ""}),
                "info_6": ("STRING", {"default": ""}),
                "去重字段_1": ("STRING", {"default": ""}),
                "去重字段_2": ("STRING", {"default": ""}),
                "去重字段_3": ("STRING", {"default": ""}),
                "不为空字段_1": ("STRING", {"default": ""}),
                "不为空字段_2": ("STRING", {"default": ""}),
                "不为空字段_3": ("STRING", {"default": ""}),
                "为空字段_1": ("STRING", {"default": ""}),
                "为空字段_2": ("STRING", {"default": ""}),
                "为空字段_3": ("STRING", {"default": ""}),
                "等于字段_1_字段名称": ("STRING", {"default": ""}),
                "等于字段_1_字段值": ("STRING", {"default": ""}),
                "等于字段_2_字段名称": ("STRING", {"default": ""}),
                "等于字段_2_字段值": ("STRING", {"default": ""}),
                "等于字段_3_字段名称": ("STRING", {"default": ""}),
                "等于字段_3_字段值": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("INT", "STRING", "STRING", "STRING", "STRING", "STRING", "LIST", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("csv数据条数", "columns", "task_id", "输入底图url", "输入风格参考图url", "输入遮罩url", "输入知末生成图url_list", "info_1", "info_2", "info_3", "info_4", "info_5", "info_6")
    FUNCTION = "load_images_sequence"
    CATEGORY = "🐼malio/image"

    def load_images_sequence(
            self, csv_file_path, 
            获取同一个task_id的其他生成图片, seed, 
            key_task_id_字段名称="",
            key_输入底图url_字段名称="",
            key_输入风格参考图url_字段名称="",
            key_输入遮罩url_字段名称="",
            key_输入知末生成图url_字段名称="",
            info_1="", info_2="", info_3="", info_4="", info_5="", info_6="",
            去重字段_1="", 去重字段_2="", 去重字段_3="", 
            不为空字段_1="", 不为空字段_2="", 不为空字段_3="",
            为空字段_1="", 为空字段_2="", 为空字段_3="",
            等于字段_1_字段名称="", 等于字段_1_字段值="",
            等于字段_2_字段名称="", 等于字段_2_字段值="",
            等于字段_3_字段名称="", 等于字段_3_字段值="",
    ):
        """顺序加载文件夹中的图片，支持随机加载。 一张一张加载"""
        # try:
        print(f"开始加载csv文件：{csv_file_path}")
        print(f"传入参数：获取同一个task_id的其他生成图片：{获取同一个task_id的其他生成图片}, seed：{seed}, key_task_id_字段名称：{key_task_id_字段名称}, key_输入底图url_字段名称：{key_输入底图url_字段名称}, key_输入风格参考图url_字段名称：{key_输入风格参考图url_字段名称}, key_输入遮罩url_字段名称：{key_输入遮罩url_字段名称}, key_输入知末生成图url_字段名称：{key_输入知末生成图url_字段名称}, info_1：{info_1}, info_2：{info_2}, info_3：{info_3}, info_4：{info_4}, info_5：{info_5}, info_6：{info_6}, 去重字段_1：{去重字段_1}, 去重字段_2：{去重字段_2}, 去重字段_3：{去重字段_3}, 不为空字段_1：{不为空字段_1}, 不为空字段_2：{不为空字段_2}, 不为空字段_3：{不为空字段_3}, 为空字段_1：{为空字段_1}, 为空字段_2：{为空字段_2}, 为空字段_3：{为空字段_3}, 等于字段_1_字段名称：{等于字段_1_字段名称}, 等于字段_1_字段值：{等于字段_1_字段值}, 等于字段_2_字段名称：{等于字段_2_字段名称}, 等于字段_2_字段值：{等于字段_2_字段值}, 等于字段_3_字段名称：{等于字段_3_字段名称}, 等于字段_3_字段值：{等于字段_3_字段值}")

        if not os.path.exists(csv_file_path):
            raise Exception(f"csv文件路径错误：{csv_file_path}")
        if csv_file_path.lower().endswith((".csv")):
            df = pd.read_csv(csv_file_path, header=0)
        elif csv_file_path.lower().endswith((".xlsx")):
            df = pd.read_excel(csv_file_path, header=0)
        else:
            raise Exception(f"csv文件路径错误：{csv_file_path}")
        
        print(f"过滤前csv数据条数：{len(df)}")
        columns = df.columns.tolist()
        print(f"csv文件列名：{columns}")

        if key_输入底图url_字段名称 and key_输入风格参考图url_字段名称:
            # 如果输入了底图和参考图, 去除底图和参考图相同的行
            # df = df[df[key_输入底图url_字段名称] != df[key_输入风格参考图url_字段名称]]
            print(f"过滤条件：{key_输入底图url_字段名称} != {key_输入风格参考图url_字段名称}, 过滤后数据条数：{len(df)}")

        raw_df = df

        duplicate_columns = [去重字段_1, 去重字段_2, 去重字段_3]
        not_null_columns = [不为空字段_1, 不为空字段_2, 不为空字段_3]
        null_columns = [为空字段_1, 为空字段_2, 为空字段_3]
        equal_columns = [(等于字段_1_字段名称, 等于字段_1_字段值), (等于字段_2_字段名称, 等于字段_2_字段值), (等于字段_3_字段名称, 等于字段_3_字段值)]
        
        for col in not_null_columns:
            if col in columns and col != "":
                df = df.dropna(subset=[col])
                print(f"过滤条件：{col} 不为空后数据条数：{len(df)}")
        for col in null_columns:
            if col in columns and col != "":
                df = df[df[col].isnull()]
                print(f"过滤条件：{col} 为空后数据条数：{len(df)}")
        for col in duplicate_columns:
            if col in columns and col != "":
                df = df.drop_duplicates(subset=[col], keep="first")
                print(f"过滤条件：{col} 去重后数据条数：{len(df)}")
        
        for col, value in equal_columns:
            if col in columns and col != "":
                if get_column_type(df[col]) == "字符串类型":
                    df = df[df[col] == value]
                elif get_column_type(df[col]) == "整数类型":
                    df = df[df[col] == int(value)]
                elif get_column_type(df[col]) == "浮点数类型":
                    df = df[df[col] == float(value)]
                elif get_column_type(df[col]) == "布尔类型":
                    df = df[df[col] == bool(value)]
                print(f"过滤条件：{col} == {value}, 过滤后数据条数：{len(df)}")

        total_num = len(df)
        print(f"过滤后csv数据条数：{total_num}")
        if total_num == 0:
            return (None, None, None, None, None, None, None, None)
        
        iloc = seed % total_num

        return_values = []
        
        args = [info_1, info_2, info_3, info_4, info_5, info_6]
        for para in args:
            print(f"参数：{para}")
            if para == "" or para not in columns or pd.isna(df.iloc[iloc][para]):
                return_values.append(None)
            else:
                return_values.append(str(df.iloc[iloc][para]))
        
        return_values_2 = []
        for name, col in zip(["key_task_id_字段名称", "key_输入底图url_字段名称", "key_输入风格参考图url_字段名称", "key_输入遮罩url_字段名称", "key_输入知末生成图url_字段名称"],
                             [key_task_id_字段名称, key_输入底图url_字段名称, key_输入风格参考图url_字段名称, key_输入遮罩url_字段名称, key_输入知末生成图url_字段名称]):
            if col in columns and col != "":
                # Literal["字符串类型", "整数类型", "浮点数类型", "布尔类型", "未知类型"]
                if get_column_type(df[col]) == "字符串类型":
                    if name == "key_输入知末生成图url_字段名称":
                        # 获取同一个task_id的其他生成图片
                        if 获取同一个task_id的其他生成图片:
                            task_id = int(df.iloc[iloc][key_task_id_字段名称])
                            znzmo_url_list = raw_df[raw_df[key_task_id_字段名称] == task_id][key_输入知末生成图url_字段名称].tolist()
                            print(f"获取同一个task_id的其他生成图片：{znzmo_url_list}, 类型为：{type(znzmo_url_list)}")
                            return_values_2.append(znzmo_url_list)
                        else:
                            return_values_2.append([str(df.iloc[iloc][col])])
                    else:
                        return_values_2.append(str(df.iloc[iloc][col]))
                elif get_column_type(df[col]) == "整数类型":
                    return_values_2.append(int(df.iloc[iloc][col]))
                elif get_column_type(df[col]) == "浮点数类型":
                    return_values_2.append(float(df.iloc[iloc][col]))
                elif get_column_type(df[col]) == "布尔类型":
                    return_values_2.append(bool(df.iloc[iloc][col]))
                else:
                    return_values_2.append(None)
            else:
                return_values_2.append(None)
        
        print(f"返回参数：{return_values}")
        return tuple([total_num] + [str(columns)] +return_values_2 +  return_values)

        # except Exception as e:
        #     print(f"加载csv文件出错，请检查文件路径：{e}")
        #     return (None, None, None, None, None, None, None, None)
         



class Malio_Repeat_and_Tile_Image:
    """图片复制并且平铺"""

    scale_methods = scale_methods
    crop_methods = ["disabled", "center"]

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": Field.image(),
                "new_width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": -sys.maxsize,
                        "max": sys.maxsize,
                        "step": 1,
                    },
                ),
                "new_height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": -sys.maxsize,
                        "max": sys.maxsize,
                        "step": 1,
                    },
                ),
                "repeat_x": ("INT", {"default": 2, "min": 1, "max": 32}),
                "upscale_method": Field.combo(cls.scale_methods),
                "crop": Field.combo(cls.crop_methods),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "repeat_and_tile"

    CATEGORY = "🐼malio/image/repeat_and_tile_image"

    def repeat_and_tile(
        self, image, new_width, new_height, repeat_x, upscale_method, crop
    ):
        print(image.shape)
        samples = image.movedim(-1, 1)  # [1,512,512,3] -> [1,3,512,512]

        # 将 image 上下左右各复制 repeat_x 次

        # 左右拼接
        # new_image = torch.cat([samples] * repeat_x, dim=2)
        for i in range(repeat_x):
            if i == 0:
                new_image = samples
            else:
                if i % 2 == 0:
                    new_image = torch.cat([new_image, samples], dim=2)
                else:
                    # 左右翻转
                    new_image = torch.cat([new_image, torch.flip(samples, [2])], dim=2)

        # 上下拼接
        # new_image = torch.cat([new_image] * repeat_x, dim=3)
        for i in range(repeat_x):
            if i == 0:
                res = new_image
            else:
                if i % 2 == 0:
                    res = torch.cat([res, new_image], dim=3)
                else:
                    # 上下翻转
                    res = torch.cat([res, torch.flip(new_image, [3])], dim=3)

        print(res.shape)
        # 缩放图片
        res = common_upscale(res, new_width, new_height, upscale_method, crop)
        # 创建一个新图片，用于存放拼接后的图片
        print(res.shape)

        res = res.movedim(1, -1)  # 移动维度

        return (res,)


class Malio_SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    },
                ),
                "image_type": Field.combo(["png", "jpg"]),
                "输出原文件名": ("BOOLEAN", {"default": False}),
                "输出文件质量":("INT", {"default": 100, "min": 1, "max": 100}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(
        self,
        images,
        image_type: str,
        filename_prefix="ComfyUI",
        prompt=None,
        extra_pnginfo=None,
        输出原文件名=False,
        输出文件质量=100,
    ):
        raw_file_name = filename_prefix
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            if image_type == "jpg":
                file = f"{filename_with_batch_num}_{counter:05}_.jpg"
                img = img.convert("RGB")
                if 输出原文件名:  # 如果输出原文件名，则将文件名改为原文件名
                    rindex = raw_file_name.rfind(".")
                    if rindex != -1:
                        file = raw_file_name[:rindex] + ".jpg"
                    else:
                        file = raw_file_name + ".jpg"
                img.save(fp=os.path.join(full_output_folder, file), format="JPEG", quality=输出文件质量)
            else:
                file = f"{filename_with_batch_num}_{counter:05}_.png"
                if 输出原文件名:
                    rindex = raw_file_name.rfind(".")
                    if rindex != -1:
                        file = raw_file_name[:rindex] + ".png"
                    else:
                        file = raw_file_name + ".png"
                img.save(
                    os.path.join(full_output_folder, file),
                    pnginfo=metadata,
                    compress_level=self.compress_level,
                )

            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}}


class Malio_SD35_Image_Resize:
    """Image Resize 缩放到SD35比例
    
    参考：https://www.cnblogs.com/bossma/p/17615201.html
    """

    scale_methods = scale_methods
    crop_methods = ["disabled", "center"]

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": Field.image(),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "🐼malio/image/sdxl_resize"

    def upscale(
        self,
        image,
    ):
        samples = image.movedim(-1, 1)

        size = get_image_size(image)
        size_tuple_width = (
            (1024, 1024),
            (1152, 896),
            (1216, 832),
            (1344, 768),
            # (1536, 640),
        )
        size_tuple_height = (
            (1024, 1024),
            (896, 1152),
            (832, 1216),
            (768, 1344),
            (640, 1536),
        )


        width_B = int(size[0])
        height_B = int(size[1])
        
        if width_B == height_B:
            new_width, new_height = 1024, 1024
        elif width_B > height_B:
            # 选择宽高比例最接近的
            new_width, new_height = min(size_tuple_width, key=lambda x: abs(x[0] / x[1] - width_B / height_B))
        else:
            new_width, new_height = min(size_tuple_height, key=lambda x: abs(x[0] / x[1] - width_B / height_B))
        

        res = common_upscale(samples, new_width, new_height, upscale_method="lanczos", crop="disabled")
        res = res.movedim(1, -1)  # 移动维度
        return (res,)

# class Maliooo_Get_SuappImage:
#     """Load an image from the given URL"""

#     # 来源：\custom_nodes\comfy_mtb\nodes\image_processing.py

#     def __init__(self):
#         self.json_path = os.path.join(r"/data/ai_draw_data", "suapp_config.json")
#         with open(self.json_path, "r") as f:
#             self.config = json.load(f)
#             self.authentication = self.config["authentication"]
#             self.cookie = self.config["cookie"]
#             print(f"读取suapp配置文件: {self.config}")

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "images": ("IMAGE", {"tooltip": "需要suapp处理的底图"}),
#                 "场景": Field.combo(["家装", "工装", "建筑", "景观"]),
#                 "Lora名称": ("STRING", {"default": "", "tooltip": "通过输入的lora名称去判断场景",}),
#                 "正向提示词": ("STRING", {"default": ""}),
#                 "authentication": ("STRING", {"default": "", "tooltip": "默认为空,会读config文件"}),
#                 "cookie": ("STRING", {"default": "", "tooltip": "默认为空,会读config文件"}),
#             }
#         }

#     RETURN_TYPES = ("IMAGE",)
#     RETURN_NAMES = (
#         "IMAGE",
#     )
#     FUNCTION = "get_suapp_image"
#     CATEGORY = "🐼malio/image"

#     def get_suapp_image(
#         self,
#         images,
#         Lora名称: str = "",
#         场景: str = "",
#         authentication: str = "",
#         cookie: str = "",
#         正向提示词: str = "",

#     ):
#         """返回suapp生成的图片的url"""

#         print(f"开始进行suapp生成图片， 传入图片数量：{len(images)}， 场景：{场景}， Lora名称：{Lora名称}， 正向提示词：{正向提示词}")

#         p_prompt = 正向提示词
#         if Lora名称:
#             p_lora_name = Lora名称
#         else:
#             p_lora_name = 场景

#         # ================== 1. 构建请求的信息 ==================
#         user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 SketchUp Pro/23.1 (Mac) Safari/537.36"
#         current_time = datetime.datetime.now()
#         # Set the headers as provided in the curl request
#         headers = {
#             "authentication": authentication if authentication else self.authentication,
#             "user-agent": user_agent,
#             "cookie": cookie if cookie else self.cookie,
#             "origin": "https://ai.sketchupbar.com",
#             "sec-fetch-site": "same-origin",
#             "sec-fetch-mode": "cors",
#             "sec-fetch-dest": "empty",
#             "referer": "https://ai.sketchupbar.com/public/index.html?r={}".format(
#                 current_time.second
#             ),
#             "Content-Type": "application/json",
#             "sec-ch-ua-platform": "Windows",

#         }
#         print(f"请求头headers: {headers}")

#         # "ModernBookstore": "现代风","ModernBookstore": "图书馆空间"
#         lora_dict = {
#             "ModernLandscapeFormation": "公园景观",
#             "modernScienceandtechnologyexhibitionhallV3": "展厅空间",
#             "modernCommercialstreetV3": "商业建筑",
#             "newChineseHomeLivingRoom-V3": "中国风",
#             "modernBar-V3": "酒吧空间",
#             "modernStandardfactory-V4": "工业建筑",
#             "ModernOpenOffice-V2": "办公空间",
#             "newChineseChineserestaurantV3": "中国风",
#             "modernfrontdeskV3": "办公空间",
#             "ModernBookstore": "图书馆空间",
#             "modernHotelLobby-V4": "酒店空间",
#             "wabisabi": "侘寂风",
#             "modernmeetingroomV3": "办公空间",
#             "modernhotelroomV3": "酒店空间",
#             "UrbanConstruction": "商业建筑",
#             "家装" : "现代风",
#             "工装" : "商业空间",
#             "建筑" : "商业建筑",
#             "景观" : "园区景观",
#         }

#         render_style_dict = {
#             "商业建筑": "Architecture",
#             "中式建筑": "ArchitectureCN",
#             "别墅建筑": "ArchiVilla",
#             "乡村建筑": "ArchiRural",
#             "工业建筑": "ArchiIndustrial",
#             "教育建筑": "ArchiEDU",
#             "办公建筑": "ArchiOffice",
#             "住宅建筑": "ArchiResidential",
#             "酒店建筑": "ArchiHotel",
#             "观演建筑": "ArchiTheatrical",
#             "城市透视": "UrbanPerspective",
#             "城市鸟瞰": "UrbanAerial",
#             "总平面图": "MasterPlan",
#             "现代风": "InteriorDesign",
#             "奶油风": "InteriorCream",
#             "侘寂风": "InteriorWabi",
#             "中国风": "InteriorCN",
#             "工业风": "InteriorIndustrial",
#             "轻奢风": "InteriorLuxury",
#             "暗黑风": "InteriorGray",
#             "原木风": "InteriorWood",
#             "色彩风": "InteriorColor",
#             "古典风": "InteriorNeoclassical",
#             "中古风": "InteriorRetro",
#             "乡村风": "InteriorRural",
#             "异域风": "InteriorExotic",
#             "赛博风": "InteriorCyber",
#             "彩平图": "ColorFloorPlan",
#             "办公空间": "InteriorOffice",
#             "餐厅空间": "InteriorRestaurant",
#             "酒店空间": "InteriorHotel",
#             "商业空间": "InteriorCommercial",
#             "车站空间": "InteriorStation",
#             "幼儿园空间": "InteriorKids",
#             "酒吧空间": "InteriorBar",
#             "婚礼空间": "InteriorWedding",
#             "图书馆空间": "InteriorLibrary",
#             "展厅空间": "InteriorExhibition",
#             "健身房空间": "InteriorGYM",
#             "舞台空间": "InteriorAuditorium",
#             "公园景观": "LandscapePark",
#             "园区景观": "LandscapeDesign",
#             "游乐场景观": "LandscapePlayground",
#             "庭院景观": "LandscapeCourtyard",
#             "大门景观": "LandscapeGate",
#             "桥梁景观": "LandscapeBridge",
#             "手工模型": "ManualModel",
#             "建筑马克笔": "ArchiMarker",
#             "景观马克笔": "LandscapeMarker",
#             "室内马克笔": "InteriorMarker",
#             "建筑手绘": "ArchiSketch",
#             "草图手绘": "SimpleSketch",
#             "绘画艺术": "PaintingArt",
#             "扁平插画": "Illustration",
#             "古风彩绘": "ColorPainting",
#         }


#         render_style = render_style_dict[lora_dict[p_lora_name]]
#         print(f"suapp生成图片，传入lora：{p_lora_name},场景：{场景} , suapp的场景：{lora_dict[p_lora_name]}, suapp渲染风格：{render_style}")

#         try:
#             print(f"开始生成SUAPP图片, 传入图片数量：{len(images)}")
#             print(f"正向提示词：{p_prompt}")
#             print(f"images_0.shape: {images[0].shape}")
#             print(f"images.shape: {images.shape}")
#         except Exception as e:
#             print(f"打印suapp函数中出错：{e}")


#         # ================== 2. 遍历每一张图片 ==================
#         suapp_image_list = []
#         for batch_number, image in tqdm(enumerate(images), desc="生成SUAPP图片"):
#             i = 255.0 * image.cpu().numpy()
#             img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
#             img = img.convert("RGB")
#             src_width, src_height = img.size

#             # 将PIL.Image 转为 base64 编码
#             buffered = io.BytesIO()  # 创建一个字节流对象
#             img.save(buffered, format="JPEG")  # 将图像保存到字节流中
#             img_str = base64.b64encode(
#                 buffered.getvalue()
#             )  # 编码字节流为 Base64 字符串
#             image_base64 = img_str.decode(
#                 "utf-8"
#             )  # 将 Base64 字节解码为字符串（如果需要在文本中使用）


#             # ================== 3. 设置图片尺寸 ==================
#             if src_width > src_height:
#                 suapp_width = round(src_width / (src_height / 512), 0)
#                 suapp_height = 512
#             elif src_height == src_width:
#                 suapp_width = 512
#                 suapp_height = 516
#             else:
#                 suapp_width = 683
#                 suapp_height = 516

#             # ================== 4. 发送请求 ==================
#             nn_image = "data:image/jpeg;base64,{}".format(image_base64)  # 图片的 base64 编码
#             max_seed = int(1024 * 1024 * 1024)
#             # Set the payload as provided in the curl request
#             split_str = ","
#             p_prompt = p_prompt.replace("，", ",")
#             prompt = []
#             if p_prompt != "":
#                 for word in p_prompt.split(split_str):
#                     prompt.append({"show": False, "value": word, "weight": 1})
            
#             payload = json.dumps(
#                 {
#                     "prompt": prompt,
#                     "neg_prompt": "",
#                     "renderStyle": render_style,
#                     "ss_scale": 5,
#                     "width": suapp_width,
#                     "height": suapp_height,
#                     "nn_image": "{}".format(nn_image),
#                     "nn_weight": 1,
#                     "outputImageNum": 1,
#                     # 高清渲染，高清渲染不能使用多张
#                     "hdRendering": True,
#                     "hdRenderingType": "01",
#                     "nn_image_scribbled": False,
#                     "scribble_accuracy": 3,
#                     "seed": random.randint(1, max_seed),
#                     "pageID": None,
#                     "camera": None,
#                     "cropInfo": None,
#                     "taskType": "t2i",
#                 }
#             )
#             start_time = time.time()
#             # print("开始时间:{}".format(start_time))
#             # Send the request to the URL provided in the curl request
#             task_response = requests.post(
#                 "https://ai.sketchupbar.com/ai/addTask",
#                 headers=headers,
#                 data=payload,
#                 verify=False,
#             )
#             print(f"suapp生成图片，第一次请求任务结果:{task_response.json()}")

#             # Check if the request was successful
#             if task_response.status_code == 200:
#                 # "{\"code\":200,\"si\":8,\"taskId\":\"1718368_1705930855007\",\"queue\":2,\"inputImagePaths\":{}}"
#                 # print(response.json())
#                 taskId = task_response.json()["taskId"]
#                 print(f"suapp发送任务成功，任务ID：{taskId}")
#                 # print('Request successful!')
#                 # 请求的 URL
#                 url = "https://ai.sketchupbar.com/ai/getTaskResult/{}?skipIMI=true&upscale=false&channel=false".format(
#                     taskId
#                 )


#                 # ================== 5. 等待请求结果 ==================
#                 repeat = 0
#                 while True:
#                     # 发送请求并获取响应
#                     result_response = requests.get(url, headers=headers, verify=False)
#                     print(f"请求结果:{result_response.json()}")
#                     print(f"请求结果:{result_response.status_code}")
#                     print(f"等待次数:{repeat}")
#                     print("-"*20)
#                     # print("响应内容：", result_response.json())
#                     if result_response.json()["msg"] == "处理成功":
#                         image = result_response.json()["image"]
#                         end_time = time.time()
#                         # print("开始时间:{}".format(end_time))
#                         # {"code":200,"msg":"处理成功","image":"air_user_images/1718368/2024/01/23/1718368_1706010776542_out_1.jpg","moreImages":null}
#                         suapp_ai_img_url = "https://ai.sketchupbar.com/{}".format(image)
#                         # img_name = image.split('/')[-1]
#                         print(
#                             "https://ai.sketchupbar.com/{}, 出图时间：{}".format(
#                                 image, (end_time - start_time)
#                             )
#                         )
#                         # 发送HTTP GET请求获取图片数据
#                         response = requests.get(suapp_ai_img_url, verify=False)
#                         # 检查请求是否成功
#                         if response.status_code == 200:
#                             # 得到PIL.Image对象
#                             img = Image.open(io.BytesIO(response.content))
#                             suapp_image_list.append(img)

#                             break
                        
#                         else:
#                             print(f"下载suapp图片失败，HTTP状态码：{response.status_code}")
#                             raise Exception(
#                                 "下载suapp图片失败，HTTP状态码：", response.status_code
#                             )
                    
#                     # 等待 1 秒后再次请求
#                     time.sleep(3)
#                     repeat += 1
#                     if repeat > 50:
#                         print("suapp生成图片超时，请求次数超过50次")
#                         break

#             else:
#                 print("Request failed with status code:", task_response.status_code)
#                 raise Exception("下载suapp图片失败，Request failed with status code:", task_response.status_code)
        
#         # ================== 6. 返回图片 ==================
#         # 将PIL.Image对象列表返回,转换为 comfyui 的 IMAGE 类型
#         output_image, output_mask = get_comfyui_images(suapp_image_list)
#         return (output_image, output_mask,)

