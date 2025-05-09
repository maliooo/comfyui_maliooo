import math
from .components.fields import Field
from .components.sizes import get_image_size
import os
from comfy.utils import common_upscale
import folder_paths
from comfy import model_management
import torch
import numpy as np
import sys
from PIL import Image, ImageOps, ImageSequence
import requests
from .utils import pil2tensor
import random
import node_helpers
import tempfile
import os

from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage, transforms
import torch
import time
import io
import tempfile


import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image, ImageDraw
import torch
import glob


class CNN_Block(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.layer = nn.Sequential(
            # 不改变特征的HW，改变特征的C
            # 完成了CHW层面的像素融合
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(),
            # nn.Dropout(0.2)
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(),
            # nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.layer(x)


# 下采样，/2，C不变。
class DownSample_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            CNN_Block(c, c)
        )

    def forward(self, x):
        return self.layer(x)


# 上采样，*2，C减半
class UpSample_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c // 2),
            nn.LeakyReLU(),
            # nn.Dropout(0.2)
        )

    # x:o5  r:o4
    # x:o6  r:o3
    # x:o7  r:o2
    # x:o8  r:o1
    def forward(self, x, r):
        # 先把HW * 2，再把C / 2
        data = F.interpolate(x, scale_factor=2, mode="nearest")
        data = self.layer(data)
        # 信息补全
        return torch.cat((data, r), dim=1)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CNN_Block(3, 64)
        self.down1 = DownSample_Block(64)
        self.conv2 = CNN_Block(64, 128)
        self.down2 = DownSample_Block(128)
        self.conv3 = CNN_Block(128, 256)
        self.down3 = DownSample_Block(256)
        self.conv4 = CNN_Block(256, 512)
        self.down4 = DownSample_Block(512)
        self.conv5 = CNN_Block(512, 1024)
        self.up1 = UpSample_Block(1024)
        #
        self.conv6 = CNN_Block(1024, 512)
        self.up2 = UpSample_Block(512)
        self.conv7 = CNN_Block(512, 256)
        self.up3 = UpSample_Block(256)
        self.conv8 = CNN_Block(256, 128)
        self.up4 = UpSample_Block(128)
        self.conv9 = CNN_Block(128, 64)
        # 输出层
        self.out_layer = nn.Conv2d(64, 3, 1, 1)

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2(self.down1(o1))
        o3 = self.conv3(self.down2(o2))
        o4 = self.conv4(self.down3(o3))
        o5 = self.conv5(self.down4(o4))
        # 信息补全
        # x:o5  r:o4
        # x:o6  r:o3
        # x:o7  r:o2
        # x:o8  r:o1
        o6 = self.conv6(self.up1(o5, o4))
        o7 = self.conv7(self.up2(o6, o3))
        o8 = self.conv8(self.up3(o7, o2))
        o9 = self.conv9(self.up4(o8, o1))
        return self.out_layer(o9)




class Malio_Image_Watermark_Mask_v0:
    """获取输入图片的水印遮罩"""

    def __init__(self) -> None:
        self.config = None
        self.device = "cpu"
        self.unet = None
        self.transform = transforms.Compose([transforms.ToTensor()])
        # self.weight_path = '/home/public/ai_chat_data/models/temp_models/unet-water_new_815.pt'  # 替换为权重文件路径
        # weights = torch.load(weight_path, map_location=device)
        # unet.load_state_dict(weights)

    def __load_model(self):
        model_path = os.path.join(folder_paths.models_dir, "watermark", "unet-water_new_815.pt")
        # model_path = folder_paths.get_full_path("watermark", "unet-water_new_815.pt")
        self.device = model_management.get_torch_device()
        self.unet = UNet().to(device=self.device)
        weights = torch.load(model_path, map_location=self.device)
        self.unet.load_state_dict(weights)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": Field.image(),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_image_watermark_mask"

    CATEGORY = "🐼malio/水印去除"

    def watermark_detection(self, image_path, threshold_value=180):
        self.__load_model()

        img = Image.open(image_path)
        # img_size = torch.Tensor((1024,1024))
        size_src = img.size
        print(size_src)
        img_size = torch.Tensor(img.size)
        radio = 1024 / img_size[torch.argmax(img_size)]
        size = (img_size * radio).long()
        # size = (img_size).long()
        img = img.resize(list(size))
        background = Image.new('RGB', (1024, 1024), (0, 0, 0))  # 黑色背景
        offset = ((1024 - size[0]) // 2, (1024 - size[1]) // 2)
        background.paste(img, offset)
        img = background
        # bg.paste(img)
        img = self.transform(img)
        img = img.unsqueeze(0)

        t1 = time.time()
        out = self.unet(img.to(self.device))
        t2 = time.time()
        print(f'推理时间:{t2-t1}')

        def binarize(image, threshold):
            return image.point(lambda p: 255 if p > threshold else 0)

        with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as temp:
            temp_path = temp.name
            save_image(out.cpu(), fp=temp_path)
            image = Image.open(temp_path).convert('L')  # 转换为灰度图像

            binary_image = binarize(image, threshold_value)
            x,y = int(offset[0]),int(offset[1])
            binary_image = binary_image.crop((x,y,1024-x,1024-y))
        return binary_image

    def get_image_watermark_mask(self, images):
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img = img.convert("RGB")

            with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as temp:
                temp_path = temp.name
                img.save(temp_path)

                watermark_mask_img = self.watermark_detection(temp_path)


                output_images = []
                w, h = None, None

                excluded_formats = ['MPO']
                
                for i in ImageSequence.Iterator(watermark_mask_img):
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

                if len(output_images) > 1 and img.format not in excluded_formats:
                    output_image = torch.cat(output_images, dim=0)
                else:
                    output_image = output_images[0]

        return (output_image, )


class Malio_Image_Watermark_Mask_yolov8:
    """获取输入图片的水印遮罩"""

    def __init__(self) -> None:
        self.config = None
        self.device = "cpu"
        self.unet = None
        self.transform = transforms.Compose([transforms.ToTensor()])
        # self.weight_path = '/home/public/ai_chat_data/models/temp_models/unet-water_new_815.pt'  # 替换为权重文件路径
        # weights = torch.load(weight_path, map_location=device)
        # unet.load_state_dict(weights)

    def __load_model(self):
        # model_path = os.path.join(folder_paths.models_dir, "watermark", "watermark_train_v8x_1280_batch16_epoch100_v26.pt")
        model = YOLO(self.model_path)
        self.device = model_management.get_torch_device()
        self.model = model


    @classmethod
    def INPUT_TYPES(cls):
        
        watermark_model_path_list = os.listdir(os.path.join(folder_paths.models_dir, "watermark"))
        return {
            "required": {
                "images": Field.image(),
                "model_path": (watermark_model_path_list, {"tooltip": "选择的水印模型路径", "default": "watermark_obb_ID_1280_auto5.pt"}),
            }
        }

    RETURN_TYPES = ("IMAGE","IMAGE")
    RETURN_NAMES = ("raw_image", "mask_image")
    FUNCTION = "get_image_watermark_mask"

    CATEGORY = "🐼malio/水印去除"

    def draw_boxes(self, image:Image.Image, boxes:list):
        """在图片上画出框，框填充颜色为白色"""
        for box in boxes:
            # 在图片上画出检测到的物体
            # box = box.tolist()
            draw = ImageDraw.Draw(image)
            # 在图片上画一个矩形框，整个矩形框为白色
            # 左上，右上，右下，左下
            # [[132.47108459472656, 70.09575653076172], [132.7020721435547, 96.65149688720703], [245.7186737060547, 95.66840362548828], [245.48768615722656, 69.11266326904297]]
            _box = [tuple(item) for item in box]
            draw.polygon(_box, outline="white", fill="white")
            
        return image

    def watermark_detection(self, image_path):
        self.__load_model()

        with torch.no_grad():
            results = self.model(
                image_path,
                save = True,
                device=self.device,
                imgsz=1280,
                show_conf = True,
                show_labels=True,
                iou=0.5,
                conf=0.1
            )

        # print(len(results))
        obb_list = []
        conf_list = []
        for result in results:
            for _obb, _conf in zip(result.obb, result.obb.conf.tolist()):    
                # print((_obb.xyxyxyxy).squeeze().tolist())
                obb_list.append((_obb.xyxyxyxy).squeeze().tolist())
                conf_list.append(_conf)


        image = Image.open(image_path)
        # 创建一个纯黑色图，大小和原图一样
        new_image = Image.new("RGB", image.size, "black")

        # 取conf大于0.25的框, 如果小于0.25的框数量为0，则再取conf大于0.2的框, 如果小于0.2的框数量为0，则再取conf大于0.15的框, 直到conf大于0.1的框
        new_obb_list_025 = [obb for obb, conf in zip(obb_list, conf_list) if conf > 0.25]
        new_obb_list_02 = [obb for obb, conf in zip(obb_list, conf_list) if conf > 0.2]
        new_obb_list_015 = [obb for obb, conf in zip(obb_list, conf_list) if conf > 0.15]
        new_obb_list_01 = [obb for obb, conf in zip(obb_list, conf_list) if conf > 0.1]
        
        if len(new_obb_list_025) > 0:
            new_obb_list = new_obb_list_025
        elif len(new_obb_list_02) > 0:
            new_obb_list = new_obb_list_02
        elif len(new_obb_list_015) > 0:
            new_obb_list = new_obb_list_015
        elif len(new_obb_list_01) > 0:
            new_obb_list = new_obb_list_01
        else:
            new_obb_list = []    

        mask_image = self.draw_boxes(new_image, new_obb_list)
        
        return {
            "mask_image": mask_image,
            "obb_list": obb_list,
            "conf_list": conf_list
        }

    def get_image_watermark_mask(self, images, model_path:str):
        self.model_path = os.path.join(folder_paths.models_dir, "watermark", model_path)
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img = img.convert("RGB")

            with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as temp:
                temp_path = temp.name
                img.save(temp_path)

                watermark_result = self.watermark_detection(temp_path)
                mask_image, obb_list, conf_list = watermark_result["mask_image"], watermark_result["obb_list"], watermark_result["conf_list"]


                output_images = []
                w, h = None, None

                excluded_formats = ['MPO']
                
                for i in ImageSequence.Iterator(mask_image):
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

                if len(output_images) > 1 and img.format not in excluded_formats:
                    output_image = torch.cat(output_images, dim=0)
                else:
                    output_image = output_images[0]

        return (images, output_image, )
    

class Malio_Image_Watermark_EasyOCR:
    """使用easyocr 获取输入图片的水印遮罩
    输出3种mask，全图mask，1/2全图右下角，1/4全图右下角
    其中1/4全图右下角，是x轴大于一半，y轴大于3/4的区域
    """

    def __init__(self) -> None:
        self.config = None
        self.device = "cpu"
        self.transform = transforms.Compose([transforms.ToTensor()])


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": Field.image(),
                "use_gpu": Field.boolean(default=True),
                # "只选择右下角mask": Field.boolean(default=True),
                "置信度": Field.float(default=0.0),
            }
        }

    RETURN_TYPES = ("IMAGE","IMAGE", "IMAGE","IMAGE")
    RETURN_NAMES = ("raw_image", "全图mask", "1/2全图右下角", "1/4全图右下角")
    FUNCTION = "get_image_watermark_easyocr"

    CATEGORY = "🐼malio/水印去除"

    def draw_boxes(self, image:Image.Image, boxes:list):
        """在图片上画出框，框填充颜色为白色"""
        for box in boxes:
            # 在图片上画出检测到的物体
            # box = box.tolist()
            draw = ImageDraw.Draw(image)
            # 在图片上画一个矩形框，整个矩形框为白色
            # 左上，右上，右下，左下
            # [[132.47108459472656, 70.09575653076172], [132.7020721435547, 96.65149688720703], [245.7186737060547, 95.66840362548828], [245.48768615722656, 69.11266326904297]]
            _box = [tuple(item) for item in box]
            draw.polygon(_box, outline="white", fill="white")
            
        return image

    def watermark_detection(self, image_path, user_gpu=False, threshold=0.2):
        # 原文链接：https://blog.csdn.net/yohnyang/article/details/130300923
        import easyocr

        image = Image.open(image_path)
        width, height = image.size

        reader = easyocr.Reader(
            lang_list=['ch_sim', 'en'], # 需要导入的语言识别模型，可以传入多个语言模型，其中英语模型en可以与其他语言共同使用
            gpu=user_gpu, # 默认为false，是否使用GPU
            download_enabled=True # 默认为True，如果 EasyOCR 无法找到模型文件，则启用下载
        )
        
        with torch.no_grad():
            result_list = reader.readtext(image_path, detail=1 ) # 图片可以传入图片路径、也可以传入图片链接。但推荐传入图片路径，会提高识别速度。包含中文会出错。设置detail=0可以简化输出结果，默认为1
        obb_list = []
        obb_list_1_2 = []  # 1/2全图右下角
        obb_list_1_4 = []  # 1/4全图右下角
        conf_list = []
        for res in result_list:
            # ([[753, 456], [941, 456], [941, 508], [753, 508]], '30溜溜网', 0.9172482580194635)
            # 位置，文本，置信度。 位置为四个点的坐标（左上，右上，右下，左下）
            assert len(res) == 3
            if res[2] < threshold:  # 过滤掉置信度小于0.2的结果
                continue
                
            # 全图mask
            obb_list.append(res[0])
            conf_list.append(res[2])

            # 1/2全图右下角
            if res[0][0][0] > width//2 and res[0][0][1] > height//2:
                obb_list_1_2.append(res[0])
                
            # 1/4全图右下角
            if res[0][0][0] > int(width//2) and res[0][0][1] > (height/4*3):  # x轴大于一半，y轴大于3/4
                obb_list_1_4.append(res[0])
        
        # 创建一个纯黑色图，大小和原图一样
        new_image = Image.new("RGB", image.size, "black")
        new_image_1_2 = Image.new("RGB", image.size, "black")
        new_image_1_4 = Image.new("RGB", image.size, "black")
        mask_image = self.draw_boxes(new_image, obb_list)
        mask_image_1_2 = self.draw_boxes(new_image_1_2, obb_list_1_2)
        mask_image_1_4 = self.draw_boxes(new_image_1_4, obb_list_1_4)
        return {
            "mask_image": mask_image,
            "mask_image_1_2": mask_image_1_2,
            "mask_image_1_4": mask_image_1_4,
            "obb_list": obb_list,
            "conf_list": conf_list
        }

    def get_image_watermark_easyocr(self, images, use_gpu:bool=False, 置信度:float=0.0):
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img = img.convert("RGB")

            with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as temp:
                temp_path = temp.name
                img.save(temp_path)

                watermark_result = self.watermark_detection(temp_path, user_gpu=use_gpu, threshold=置信度)
                mask_image, mask_image_1_2, mask_image_1_4 = watermark_result["mask_image"], watermark_result["mask_image_1_2"], watermark_result["mask_image_1_4"]

                output_images = pil2tensor(mask_image)
                output_images_1_2 = pil2tensor(mask_image_1_2)
                output_images_1_4 = pil2tensor(mask_image_1_4)


                # output_images = []
                # w, h = None, None
                # excluded_formats = ['MPO']
                
                # for i in ImageSequence.Iterator(mask_image):
                #     i = node_helpers.pillow(ImageOps.exif_transpose, i)

                #     if i.mode == 'I':
                #         i = i.point(lambda i: i * (1 / 255))
                #     image = i.convert("RGB")

                #     if len(output_images) == 0:
                #         w = image.size[0]
                #         h = image.size[1]
                    
                #     if image.size[0] != w or image.size[1] != h:
                #         continue
                    
                #     image = np.array(image).astype(np.float32) / 255.0
                #     image = torch.from_numpy(image)[None,]
                #     if 'A' in i.getbands():
                #         mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                #         mask = 1. - torch.from_numpy(mask)
                #     else:
                #         mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
                #     output_images.append(image)

                # if len(output_images) > 1 and img.format not in excluded_formats:
                #     output_image = torch.cat(output_images, dim=0)
                # else:
                #     output_image = output_images[0]



        return (images, output_images, output_images_1_2, output_images_1_4)