import os
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import sys
import time
import glob
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, HF_MODEL_NAME
from models.mbv2_mlsd_large import MobileV2_MLSD_Large
from utils import pred_lines


class MLSDdetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, filename="mlsd_large_512_fp32.pth"):
        subfolder = "annotator/ckpts" if pretrained_model_or_path == "lllyasviel/ControlNet" else ''
        model_path = custom_hf_download(pretrained_model_or_path, filename, subfolder=subfolder)
        model = MobileV2_MLSD_Large()
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()

        return cls(model)

    def to(self, device):
        self.model.to(device)
        return self
    
    def __call__(self, input_image, thr_v=0.1, thr_d=0.1, detect_resolution=512, output_type="pil", upscale_method="INTER_AREA", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        img = detected_map
        img_output = np.zeros_like(img)
        try:
            with torch.no_grad():
                lines = pred_lines(img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d)
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        except Exception as e:
            pass

        detected_map = remove_pad(HWC3(img_output[:, :, 0]))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map


if __name__ == "__main__":
    # G:\Comfyui\ComfyUI-aki-v1.3\custom_nodes\comfyui_controlnet_aux\ckpts\lllyasviel\Annotators
    mlsd_model = MLSDdetector.from_pretrained(
        pretrained_model_or_path=r"/data/ai_draw_data/custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators"
        )
    mlsd_model.to("cuda:0")


    # ============================ 一张图片 ============================
    # for _ in range(100):
    #     t1 = time.time()
    #     image_file_path = r"/home/zhangxuqi/malio/test/code/seg分割图处理/comfyui_controlnet_aux/PixPin_2024-09-05_23-54-08.jpg"
    #     input_image = cv2.imread(image_file_path)
    #     print(f"处理前图像size： {input_image.shape}")
    #     # 使得input_image的长宽都是4的倍数
    #     input_image = cv2.resize(input_image, (input_image.shape[1] // 4 * 4, input_image.shape[0] // 4 * 4))
    #     with torch.no_grad():
    #         output = oneformer(input_image, output_type="pil", is_text=False, detect_resolution=1024)   # 这里会下采样4倍
    #     print(f"处理后图像size: {output.size}")
    #     output.save(r"/home/zhangxuqi/malio/test/code/seg分割图处理/分割-代码/comfyui_controlnet_aux/src/custom_controlnet_aux/mlsd/output.jpg")
    #     print(f"生成的图片保存在: {r'/home/zhangxuqi/malio/test/code/seg分割图处理/分割-代码/comfyui_controlnet_aux/src/custom_controlnet_aux/mlsd/output.jpg'}")
    #     print(f"处理一张图片耗时: {time.time() - t1:.3f} seconds.")
    # exit()



    # ============================ 多张图片 ============================
    image_dir = r"/home/public/ai_chat_data/datas/xiaoguotu/筛选13w效果图无水印/筛选13w效果图无水印"

    # detect_resolution=0 的时候，输出的图片是原图大小，否则输出的图片是短边512
    detect_resolution = 512
    if detect_resolution != 0 and detect_resolution != 512:
        raise ValueError("像素值detect_resolution 只能是0或512")

    if detect_resolution == 0:
        output_dir = r"/home/public/ai_chat_data/datas/xiaoguotu/筛选13w效果图无水印/筛选13w效果图无水印_mlsd图_完美像素"
    else:
        output_dir = r"/home/public/ai_chat_data/datas/xiaoguotu/筛选13w效果图无水印/筛选13w效果图无水印_mlsd图_短边512"

    os.makedirs(output_dir, exist_ok=True)
    image_file_path_list = glob.glob(os.path.join(image_dir, "*.jpg"))
    print("一共有图片数量：")
    print(len(image_file_path_list))
    print(image_file_path_list[0])
    print(f"保存地址为: {output_dir}")
    print(f"detect_resolution : {detect_resolution}")
    print("开始处理图片...")
    print("-"*50)
    time.sleep(10)


    pbar = tqdm(total=len(image_file_path_list))
    for image_file_path in image_file_path_list:
        pbar.update(1)
        if os.path.exists(os.path.join(output_dir, os.path.basename(image_file_path))):
            continue

        input_image = cv2.imread(image_file_path)
        # 使得input_image的长宽都是4的倍数
        # input_image = cv2.resize(input_image, (input_image.shape[1] // 4 * 4, input_image.shape[0] // 4 * 4))
        output = mlsd_model(input_image, detect_resolution=detect_resolution)
        output = output.convert("RGB")
        print(output.size)
        output.save(os.path.join(output_dir, os.path.basename(image_file_path)), "JPEG", )