import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from api import make_detectron2_model, semantic_run
from pathlib import Path
import warnings
from custom_controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, HF_MODEL_NAME
import numpy as np
import cv2
from PIL import Image
import time
import glob
from tqdm import tqdm
import torch
import random

DEFAULT_CONFIGS = {
    "coco": {
        "name": "150_16_swin_l_oneformer_coco_100ep.pth",
        "config": os.path.join(os.path.dirname(__file__), 'configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml')
    },
    "ade20k": {
        "name": "250_16_swin_l_oneformer_ade20k_160k.pth",
        "config": os.path.join(os.path.dirname(__file__), 'configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml')
    }
}
class OneformerSegmentor:
    def __init__(self, model, metadata):
        self.model = model
        self.metadata = metadata

    def to(self, device, weight_type=torch.float32):
        self.model.model.to(device=device, dtype=weight_type)
        return self
    
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, filename="250_16_swin_l_oneformer_ade20k_160k.pth", config_path = None):
        # /home/zhangxuqi/malio/test/code/seg分割图处理/分割-代码/comfyui_controlnet_aux/src/custom_controlnet_aux/oneformer/configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml
        config_path = config_path or DEFAULT_CONFIGS["ade20k" if "ade20k" in filename else "coco"]["config"]
        # /data/ai_draw_data/custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/250_16_swin_l_oneformer_ade20k_160k.pth
        model_path = custom_hf_download(pretrained_model_or_path, filename)

        model, metadata = make_detectron2_model(config_path, model_path)

        return cls(model, metadata)
    
    def __call__(self, input_image=None, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", is_text=False,  **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        detected_map, label_text_list, label_text_ratio = semantic_run(input_image, self.model, self.metadata, is_text=is_text)
        detected_map = remove_pad(HWC3(detected_map))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map, label_text_list, label_text_ratio


if __name__ == "__main__":
    # G:\Comfyui\ComfyUI-aki-v1.3\custom_nodes\comfyui_controlnet_aux\ckpts\lllyasviel\Annotators
    oneformer = OneformerSegmentor.from_pretrained(
        pretrained_model_or_path=r"/data/ai_draw_data/custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators"
        )
    oneformer.to(device="cuda:3", weight_type=torch.float32)


    # ============================ 一张图片 ============================
    for _ in range(2):
        t1 = time.time()
        image_file_path = r"/home/zhangxuqi/malio/test/code/seg分割图处理/comfyui_controlnet_aux/PixPin_2024-09-05_23-54-08.jpg"
        input_image = cv2.imread(image_file_path)
        print(f"处理前图像size： {input_image.shape}")
        # 使得input_image的长宽都是4的倍数
        input_image = cv2.resize(input_image, (input_image.shape[1] // 4 * 4, input_image.shape[0] // 4 * 4))
        output, label_text_list, label_text_ratio = oneformer(input_image, output_type="pil", is_text=False, detect_resolution=1024)   # 这里会下采样4倍
        print(f"处理后图像size: {output.size}")
        for text, raito in zip(label_text_list, label_text_ratio):
            print(f"{text}: {raito:.2f}")
        output.save(r"/home/zhangxuqi/malio/test/code/seg分割图处理/分割-代码/comfyui_controlnet_aux/src/custom_controlnet_aux/oneformer/output.jpg")
        print(f"生成的图片保存在: {r'/home/zhangxuqi/malio/test/code/seg分割图处理/分割-代码/comfyui_controlnet_aux/src/custom_controlnet_aux/oneformer/output.jpg'}")
        print(f"处理一张图片耗时: {time.time() - t1:.3f} seconds.")
    # exit()



    # ============================ 多张图片 ============================
    image_dir = r"/home/public/ai_chat_data/datas/xiaoguotu/筛选13w效果图无水印/筛选13w效果图无水印"
    is_text = True  # 图片是否输出文字
    save_text = True  # 是否保存label的txt文件

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
    # 设置随机种子
    random.seed(random.randint(0, 100000))

    # 随机打乱 image_file_path_list
    random.shuffle(image_file_path_list)
    image_file_path_list = image_file_path_list[::-1]  # 反转列表
    print(len(image_file_path_list))
    print(image_file_path_list[0])
    pbar = tqdm(total=len(image_file_path_list))
    for image_file_path in image_file_path_list:
        pbar.update(1)
        text_file_path = os.path.basename(image_file_path.replace(".jpg", ".txt"))
        
        if save_text:
            if os.path.exists(os.path.join(output_dir, os.path.basename(image_file_path))) and os.path.exists(os.path.join(output_dir, text_file_path)):
                print(f"图片和文本已经存在：{os.path.basename(image_file_path)}, {text_file_path}")
                continue
        else:
            if os.path.exists(os.path.join(output_dir, os.path.basename(image_file_path))):
                continue

        input_image = cv2.imread(image_file_path)
        # 使得input_image的长宽都是4的倍数
        # input_image = cv2.resize(input_image, (input_image.shape[1] // 4 * 4, input_image.shape[0] // 4 * 4))
        try:
            with torch.no_grad():
                output, label_text_list, label_text_ratio = oneformer(input_image, output_type="pil", is_text=is_text, detect_resolution=detect_resolution)
        except Exception as e:
            print(f"出现错误：{e}")
            continue

        # 保存文字
        if save_text and not os.path.exists(os.path.join(output_dir, os.path.basename(text_file_path))):
            text_content = "\n".join([f"{text}: {raito:.3f}" for text, raito in zip(label_text_list, label_text_ratio)])
            with open(os.path.join(output_dir, text_file_path), "w") as f:
                f.write(text_content)
        
        # 保存图片
        if not os.path.exists(os.path.join(output_dir, os.path.basename(image_file_path))):
            output = output.convert("RGB")
            print(output.size)
            output.save(os.path.join(output_dir, os.path.basename(image_file_path)), "JPEG")

        # break