import os

import cv2
import numpy as np
import torch
from PIL import Image
import time
import sys
import torch
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(__file__))

from custom_controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, HF_MODEL_NAME
from leres.depthmap import estimateboost, estimateleres
from leres.multi_depth_model_woauxi import RelDepthModel
from leres.net_tools import strip_prefix_if_present
from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel
from pix2pix.options.test_options import TestOptions


class LeresDetector:
    def __init__(self, model, pix2pixmodel):
        self.model = model
        self.pix2pixmodel = pix2pixmodel

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, filename="res101.pth", pix2pix_filename="latest_net_G.pth"):
        model_path = custom_hf_download(pretrained_model_or_path, filename)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        model = RelDepthModel(backbone='resnext101')
        model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."), strict=True)
        del checkpoint

        pix2pix_model_path = custom_hf_download(pretrained_model_or_path, pix2pix_filename)

        opt = TestOptions().parse()
        if not torch.cuda.is_available():
            opt.gpu_ids = []  # cpu mode
        pix2pixmodel = Pix2Pix4DepthModel(opt)
        pix2pixmodel.save_dir = os.path.dirname(pix2pix_model_path)
        pix2pixmodel.load_networks('latest')
        pix2pixmodel.eval()

        return cls(model, pix2pixmodel)

    def to(self, device, weight_type=torch.float32):
        self.model.to(device=device, dtype=torch.float32)
        # TODO - refactor pix2pix implementation to support device migration
        # self.pix2pixmodel.to(device)
        return self

    def __call__(self, input_image, thr_a=0, thr_b=0, boost=False, detect_resolution=512, output_type="pil", upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        with torch.no_grad():
            if boost:
                depth = estimateboost(detected_map, self.model, 0, self.pix2pixmodel, max(detected_map.shape[1], detected_map.shape[0]))
            else:
                depth = estimateleres(detected_map, self.model, detected_map.shape[1], detected_map.shape[0])

            numbytes=2
            depth_min = depth.min()
            depth_max = depth.max()
            max_val = (2**(8*numbytes))-1

            # check output before normalizing and mapping to 16 bit
            if depth_max - depth_min > np.finfo("float").eps:
                out = max_val * (depth - depth_min) / (depth_max - depth_min)
            else:
                out = np.zeros(depth.shape)
            
            # single channel, 16 bit image
            depth_image = out.astype("uint16")

            # convert to uint8
            depth_image = cv2.convertScaleAbs(depth_image, alpha=(255.0/65535.0))

            # remove near
            if thr_a != 0:
                thr_a = ((thr_a/100)*255) 
                depth_image = cv2.threshold(depth_image, thr_a, 255, cv2.THRESH_TOZERO)[1]

            # invert image
            depth_image = cv2.bitwise_not(depth_image)

            # remove bg
            if thr_b != 0:
                thr_b = ((thr_b/100)*255)
                depth_image = cv2.threshold(depth_image, thr_b, 255, cv2.THRESH_TOZERO)[1]  

        detected_map = HWC3(remove_pad(depth_image))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map


if __name__ == "__main__":
    # G:\Comfyui\ComfyUI-aki-v1.3\custom_nodes\comfyui_controlnet_aux\ckpts\lllyasviel\Annotators
    leres_model = LeresDetector.from_pretrained(
        pretrained_model_or_path=r"/data/ai_draw_data/custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators"
        )
    leres_model.to("cuda:5", weight_type=torch.float16)

    image_path_list = glob.glob(r"/home/zhangxuqi/malio/test/code/imgs/*.png")
    image_path_list = image_path_list*5
    print(len(image_path_list))
    print(image_path_list)
    # exit()

    time_list = []
    # ============================ 一张图片 ============================
    for image_file_path in image_path_list:
        t1 = time.time()
        # image_file_path = r"/home/zhangxuqi/malio/test/code/seg分割图处理/comfyui_controlnet_aux/PixPin_2024-09-05_23-54-08.jpg"
        input_image = cv2.imread(image_file_path)
        print(f"处理前图像size： {input_image.shape}")
        # 使得input_image的长宽都是4的倍数
        # input_image = cv2.resize(input_image, (input_image.shape[1] // 4 * 4, input_image.shape[0] // 4 * 4))
        input_image = cv2.resize(input_image, (1536, 1536))
        output = leres_model(input_image, output_type="pil", is_text=False, detect_resolution=1536)   # 这里会下采样4倍
        print(f"处理后图像size: {output.size}")
        output.save(r"/home/zhangxuqi/malio/test/code/seg分割图处理/comfyui_controlnet_aux/src/controlnet_aux/oneformer/output.jpg")
        print(f"生成的图片保存在: {r'/home/zhangxuqi/malio/test/code/seg分割图处理/comfyui_controlnet_aux/src/controlnet_aux/oneformer/output.jpg'}")
        print(f"处理一张图片耗时: {time.time() - t1:.3f} seconds.")
        time_list.append(time.time() - t1)
    print(f"平均处理一张图片耗时: {np.mean(time_list):.3f} seconds.")
    exit()



    # ============================ 多张图片 ============================
    image_dir = r"/home/public/ai_chat_data/datas/xiaoguotu/筛选13w效果图无水印/筛选13w效果图无水印"
    is_text = False

    # detect_resolution=0 的时候，输出的图片是原图大小，否则输出的图片是短边512
    detect_resolution = 0
    if detect_resolution != 0 and detect_resolution != 512:
        raise ValueError("像素值detect_resolution 只能是0或512")
    if is_text:
        if detect_resolution == 0:
            output_dir = r"/home/public/ai_chat_data/datas/xiaoguotu/筛选13w效果图无水印/筛选13w效果图无水印_seg图_有文字_完美像素"
        else:
            output_dir = r"/home/public/ai_chat_data/datas/xiaoguotu/筛选13w效果图无水印/筛选13w效果图无水印_seg图_有文字_短边512"
    else:
        if detect_resolution == 0:
            output_dir = r"/home/public/ai_chat_data/datas/xiaoguotu/筛选13w效果图无水印/筛选13w效果图无水印_seg图_无文字_完美像素"
        else:
            output_dir = r"/home/public/ai_chat_data/datas/xiaoguotu/筛选13w效果图无水印/筛选13w效果图无水印_seg图_无文字_短边512"

    os.makedirs(output_dir, exist_ok=True)
    image_file_path_list = glob.glob(os.path.join(image_dir, "*.jpg"))
    print(len(image_file_path_list))
    print(image_file_path_list[0])
    pbar = tqdm(total=len(image_file_path_list))
    for image_file_path in image_file_path_list:
        pbar.update(1)
        if os.path.exists(os.path.join(output_dir, os.path.basename(image_file_path))):
            continue

        input_image = cv2.imread(image_file_path)
        # 使得input_image的长宽都是4的倍数
        # input_image = cv2.resize(input_image, (input_image.shape[1] // 4 * 4, input_image.shape[0] // 4 * 4))
        output = oneformer(input_image, output_type="pil", is_text=is_text, detect_resolution=detect_resolution)
        output = output.convert("RGB")
        print(output.size)
        output.save(os.path.join(output_dir, os.path.basename(image_file_path)), "JPEG")