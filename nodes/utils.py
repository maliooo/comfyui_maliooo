import torch
import numpy as np
from typing import List, Union
from PIL import Image, ImageOps, ImageSequence
import re
import cv2

def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    """传入一个PIL Image对象或者一个Image对象列表，返回一个tensor"""
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(
        np.array(image).astype(np.float32) / 255.0
    ).unsqueeze(0)


def tensor2pil(tensor: torch.Tensor):
    """传入一个tensor，返回一个PIL Image对象list"""
    image_list:List[Image.Image] = []
    for batch_number, image in enumerate(tensor):
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img = img.convert("RGB")
        image_list.append(img)
    return image_list


def pil2cv2(image: Union[Image.Image, List[Image.Image]]):
    """传入一个PIL Image对象或者一个Image对象列表，返回一个cv2对象"""
    if isinstance(image, list):
        return [pil2cv2(img) for img in image]

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def get_comfyui_images(image_list: List[Image.Image]) -> torch.Tensor:
    """传入一个Image对象列表，返回一个confyui需要的tensor"""
    output_images = []
    output_masks = []
    w, h = None, None
    
    for i in image_list:

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

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)



def extract_info_from_webui_img(info:str):
    """从webui 生成的img中提取信息"""

    """
    输入示例:
    Modern,Tea Room,Modern interior design,<lora:modernTeaRoom-20240515030539:0.7>,8k uhd,dslr,soft lighting,high quality,film grain,Fujifilm XT3
    Negative prompt: naked,people, overweight color,big Blue,big red, distorted,ugly,worst quality,painting,sketch,(worst quality, low quality:1.4),poor anatomy,watermark,text,signature,blurry,messy,Bad Artist Sketch,(Semi-Realistic, Sketch, Cartoon, Drawing, Anime:1.4),Cropped,Out of Frame,Artifacts,Low resolution,bad anatomy,text,(mutation, bad drawing:1.2),obese,bad proportions,animals,low quality,watermark,signature,blurred,worst quality,(nsfw:1.2),realisticvision-negative-embedding
    Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7.0, Seed: 804722363, Size: 1536x1152, Model hash: e6415c4892, Model: Realistic_Vision_V2.0, VAE hash: c6a580b13a, VAE: vae-ft-mse-840000-ema-pruned.ckpt, Denoising strength: 0, ControlNet 0: "Module: depth_leres++, Model: control_v11f1p_sd15_depth [cfd03158], Weight: 0.4, Resize Mode: 1, Low Vram: False, Guidance Start: 0, Guidance End: 0.4, Pixel Perfect: False, Control Mode: 0, Save Detected Map: True", ControlNet 1: "Module: lineart_realistic, Model: control_v11p_sd15_lineart [43d4be0d], Weight: 0.7, Resize Mode: 1, Low Vram: False, Processor Res: 512, Guidance Start: 0.0, Guidance End: 0.8, Pixel Perfect: False, Control Mode: 0, Save Detected Map: True", Lora hashes: "modernTeaRoom-20240515030539: 7163f2e3ce2f", TI hashes: "realisticvision-negative-embedding: 5511b02e263f", Version: v1.6.0
    
    输出示例:
    {
        'params': {
            'prompt': 'Modern,Tea Room,Modern interior design,<lora:modernTeaRoom-20240515030539:0.7>,8k uhd,dslr,soft lighting,high quality,film grain,Fujifilm XT3',
            'negative_prompt': 'naked,people, overweight color,big Blue,big red, distorted,ugly,worst quality,painting,sketch,(worst quality, low quality:1.4),poor anatomy,watermark,text,signature,blurry,messy,Bad Artist Sketch,(Semi-Realistic, Sketch, Cartoon, Drawing, Anime:1.4),Cropped,Out of Frame,Artifacts,Low resolution,bad anatomy,text,(mutation, bad drawing:1.2),obese,bad proportions,animals,low quality,watermark,signature,blurred,worst quality,(nsfw:1.2),realisticvision-negative-embedding',
            'Steps': '20',
            'Sampler': 'DPM++ 2M Karras',
            'CFG scale': '7.0',
            'Seed': '804722363',
            'Size': '1536x1152',
            'Model hash': 'e6415c4892',
            'Model': 'Realistic_Vision_V2.0',
            'VAE hash': 'c6a580b13a',
            'VAE': 'vae-ft-mse-840000-ema-pruned.ckpt',
            'Denoising strength': '0',
            'ControlNet 0': '""',
            'ControlNet 1': '""',
            'Lora hashes': 'modernTeaRoom-20240515030539: 7163f2e3ce2f',
            'TI hashes': 'realisticvision-negative-embedding: 5511b02e263f',
            'Version': 'v1.6.0',
            'controlnets': [
                {
                    'Module': 'depth_leres++',
                    'Model': 'control_v11f1p_sd15_depth [cfd03158]',
                    'Weight': '0.4',
                    'Resize Mode': '1',
                    'Low Vram': 'False',
                    'Guidance Start': '0',
                    'Guidance End': '0.4',
                    'Pixel Perfect': 'False',
                    'Control Mode': '0',
                    'Save Detected Map': 'True'
                },
                {
                    'Module': 'lineart_realistic',
                    'Model': 'control_v11p_sd15_lineart [43d4be0d]',
                    'Weight': '0.7',
                    'Resize Mode': '1',
                    'Low Vram': 'False',
                    'Processor Res': '512',
                    'Guidance Start': '0.0',
                    'Guidance End': '0.8',
                    'Pixel Perfect': 'False',
                    'Control Mode': '0',
                    'Save Detected Map': 'True'
                }
            ]
        },
        'loras': [
            {
                'lora_name': 'modernTeaRoom-20240515030539',
                'lora_weight': 0.7
            }
        ],
        'controlnets': [
            {
                'Module': 'depth_leres++',
                'Model': 'control_v11f1p_sd15_depth [cfd03158]',
                'Weight': '0.4',
                'Resize Mode': '1',
                'Low Vram': 'False',
                'Guidance Start': '0',
                'Guidance End': '0.4',
                'Pixel Perfect': 'False',
                'Control Mode': '0',
                'Save Detected Map': 'True'},
            {
                'Module': 'lineart_realistic',
                'Model': 'control_v11p_sd15_lineart [43d4be0d]',
                'Weight': '0.7',
                'Resize Mode': '1',
                'Low Vram': 'False',
                'Processor Res': '512',
                'Guidance Start': '0.0',
                'Guidance End': '0.8',
                'Pixel Perfect': 'False',
                'Control Mode': '0',
                'Save Detected Map': 'True'}
        ]
    }

    """

    assert isinstance(info, str), "info 类型错误"
    assert info is not None, "info 不能为空"
    assert len(info.split("\n")) == 3, "info 格式错误, 请检查, 分隔后长度应为3"

    params_dict = {}  # 构建参数字典
    # lora_dict = {}
    loras = []
    info = info.split("\n")

    # 1. 正向提示词
    # Japanese,Tea Room,<lora:japaneseTeaRoom-20240515053604:0.7>,8k uhd,dslr,soft lighting,high quality,film grain,Fujifilm XT3
    positive_text = info[0].strip()
    params_dict["prompt"] = positive_text
    # 提取lora, <lora:xxx>
    lora_list = re.findall(r"<lora:(.*?)>", positive_text)
    for i in lora_list:
        positive_text = positive_text.replace(f"<lora:{i}>", "")
    for index, item in enumerate(lora_list):
        _tmp_lora_dict = {}
        lora_name, lora_weight = item.split(":")
        lora_name = lora_name.strip()
        lora_weight = eval(lora_weight.strip())
        _tmp_lora_dict["lora_name"] = lora_name
        _tmp_lora_dict["lora_weight"] = lora_weight
        # lora_dict[f"lora_{index}"] = _tmp_lora_dict
        loras.append(_tmp_lora_dict)

    # 2. 反向提示词
    negative_text = info[1].strip()
    if negative_text.startswith("Negative prompt"):
        negative_text = negative_text[len("Negative prompt:"):].strip()
    params_dict["negative_prompt"] = negative_text
    # Negative prompt: naked,people, overweight color,big Blue,big red, distorted,ugly,worst quality,painting,sketch,(worst quality, low quality:1.4),poor anatomy,watermark,text,signature,blurry,messy,Bad Artist Sketch,(Semi-Realistic, Sketch, Cartoon, Drawing, Anime:1.4),Cropped,Out of Frame,Artifacts,Low resolution,bad anatomy,text,(mutation, bad drawing:1.2),obese,bad proportions,animals,low quality,watermark,signature,blurred,worst quality,(nsfw:1.2),realisticvision-negative-embedding

    # 3. 其他参数
    # Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7.0, Seed: 3086231368, Size: 1536x1152, Model hash: e6415c4892, Model: Realistic_Vision_V2.0, VAE hash: c6a580b13a, VAE: vae-ft-mse-840000-ema-pruned.ckpt, Denoising strength: 0, ControlNet 0: "Module: depth_leres++, Model: control_v11f1p_sd15_depth [cfd03158], Weight: 0.4, Resize Mode: 1, Low Vram: False, Guidance Start: 0, Guidance End: 0.4, Pixel Perfect: False, Control Mode: 0, Save Detected Map: True", ControlNet 1: "Module: lineart_realistic, Model: control_v11p_sd15_lineart [43d4be0d], Weight: 0.7, Resize Mode: 1, Low Vram: False, Processor Res: 512, Guidance Start: 0.0, Guidance End: 0.8, Pixel Perfect: False, Control Mode: 0, Save Detected Map: True", Lora hashes: "japaneseTeaRoom-20240515053604: 9a9a36611610", TI hashes: "realisticvision-negative-embedding: 5511b02e263f", Version: v1.6.0
    # 提取ControlNet 0, ControlNet 1, Lora hashes, TI hashes
    other_params = info[2].strip()

    # 提取ControlNet, Lora hashes, TI hashes
    control_net_list = re.findall(r'ControlNet \d: "(.*?)"', other_params)
    lora_hashes = re.findall(r'Lora hashes: "(.*?)"', other_params)
    ti_hashes = re.findall(r'TI hashes: "(.*?)"', other_params)

    # 去除other_params字符串中的，ControlNet, Lora hashes, TI hashes
    for i in control_net_list:
        other_params = other_params.replace(i, "")
    for i in lora_hashes:
        other_params = other_params.replace(i, "")
    for i in ti_hashes:
        other_params = other_params.replace(i, "")

    # 提取其他参数
    for item in other_params.split(","):
        item = item.strip()
        key, value = item.split(":")
        key = key.strip().lower()
        value = value.strip()
        params_dict[key] = value
    
    # 提取ControlNet
    params_dict["controlnets"] = []
    for index, controlnet in enumerate(control_net_list):
        controlnet_dict = {}
        for item in controlnet.split(","):
            item = item.strip()
            key, value = item.split(":")
            key = key.strip()
            value = value.strip()
            controlnet_dict[key] = value

        controlnet_dict = {k.lower():v for k,v in controlnet_dict.items()}  # 转小写
        params_dict["controlnets"].append(controlnet_dict)

    # 添加Lora hashes, TI hashes
    if len(lora_hashes) > 0:
        params_dict["Lora hashes"] = lora_hashes[0]
    if len(ti_hashes) > 0:
        params_dict["TI hashes"] = ti_hashes[0]


    return {
        "positive_text": positive_text,
        "negative_text": negative_text,
        "params": params_dict,
        "loras" : loras,
        "controlnets": params_dict["controlnets"]
    }


def webui_info_2_zjxz(webui_info:str, image_url:str, width:int, height:int):
    """将webui生成的img信息转换为建筑学长格式"""

    # info_dict, 从webui生成的img中提取信息
    # {
    #     "positive_text": positive_text,
    #     "negative_text": negative_text,
    #     "params": params_dict,
    #     "loras" : loras,
    #     "controlnets": params_dict["controlnets"]
    # }

    # jzxz_info  建筑学长的post请求参数
    # "generateImageParams": {
    #     "prompt": positive_prompt,
    #     "modelName": "jianzhuw_jzxz",
    #     "negativePrompt": negative_prompt if negative_prompt else "NSFW, (worst quality:1.5), (low quality:1.5), (normal quality:1.5), lowres, blur, signature, drawing, sketch, text, word, logo, cropped, out of frame, wormline,(low quality),drawing,painting,crayon,sketch,graphite,soft,deformed,ugly,(soft line)",
    #     "schedulerName": "DPM++ 2M Karras",
    #     "guidanceScale": 7.5,
    #     "imageGeneratedNum": batch_size,
    #     "numInferenceSteps": 30,
    #     "seed": 77567595,
    #     "loraParamses": [
    #         {
    #             # "loraName": "XSArchi_124",
    #             "loraName": "xsarchitectural-9JapanesewabiSabi3",
    #             "weight": 0.6
    #         }
    #     ],
    #     "controlNetInfoList": [
    #         {
    #             "model": "control_v11p_sd15_lineart [43d4be0d]",
    #             "module": "lineart_standard",
    #             "mode": 1,
    #             "url": image_url, 
    #             "controlGuidanceStart": 0,
    #             "controlGuidanceEnd": 1,
    #             "controlnetConditioningScale": 1.4
    #         }
    #     ],
    #     "width": width,
    #     "height": height
    #     }

    # 提取webui信息
    info_dict = extract_info_from_webui_img(webui_info)
    params = info_dict["params"]
    loras = info_dict["loras"]
    controlnets = info_dict["controlnets"]
    negative_text = info_dict["negative_text"]
    seed = params["seed"]

    

    # ------------------- 构建jzxz_info
    jzxz_info_loras = []
    jzxz_info_controlnets = []
    for lora in loras:
        _tmp_lora = {
            "loraName": lora["lora_name"],
            "weight": lora["lora_weight"]
        }
        jzxz_info_loras.append(_tmp_lora)
    
    for controlnet in controlnets:
        # controlnets [{'module': 'lineart_standard', 'model': 'control_v11p_sd15_lineart [43d4be0d]', 'weight': '1.4', 'resize mode': 'ResizeMode.INNER_FIT', 'low vram': 'False', 'processor res': '512', 'guidance start': '0', 'guidance end': '1', 'pixel perfect': 'True', 'control mode': '1', 'save detected map': 'True'}]
        _tmp_controlnet = {
            "model": controlnet["model"],
            "module": controlnet["module"],
            "mode": int(controlnet["control mode"]),
            "url": image_url,
            "controlGuidanceStart": float(controlnet["guidance start"]),
            "controlGuidanceEnd": float(controlnet["guidance end"]),
            "controlnetConditioningScale": float(controlnet["weight"])
        }
        jzxz_info_controlnets.append(_tmp_controlnet)

    generateImageParams = {
        "prompt": "",
        "modelName": "jianzhuw_jzxz",
        "negativePrompt": negative_text,
        "schedulerName": "DPM++ 2M Karras",
        "guidanceScale": 7.5,
        "imageGeneratedNum": 1,
        "numInferenceSteps": 30,
        "loraParamses": jzxz_info_loras,
        "controlNetInfoList": jzxz_info_controlnets,
        "width": width,
        "height": height
    }
    
    return generateImageParams


class Malio_BBOXES:
    """使用python的eval函数生成bboxes"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input":("STRING",),
            }
        }

    RETURN_TYPES = ("BBOXES",)
    RETURN_NAMES = ("bboxs",)
    FUNCTION = "eval_function"

    def eval_function(self, input):
        """使用python的eval函数生成bboxes"""
        print(f"使用eval函数，获得输入：{input}")
        output = None
        try:
            output = eval(input)
            print(f"使用eval函数，输入：{input} 生成数据: {output}")
        except Exception as e:
            print(f"出错了，eval_function error: {e}")
            output = None
        return (output,)


# if __name__ == "__main__":
#     eval_node = Malio_BBOXES()
#     res = eval_node.eval_function("[[1,2,3,4],[5,6,7,8]]")
#     print(res)
#     print(type(res))

#     res = eval_node.eval_function("[[0,10, 1100, 200]]")
#     print(res[0])
#     print(type(res))

