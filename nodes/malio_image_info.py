import json
import re
import folder_paths
from .controlnet import apply_preprocessor
from .constant.preprocess import WEBUI_2_COMFYUI_PREPROCESS

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
        key = key.strip()
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

class Malio_Webui_Info_Params:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "webui_params_info": ("STRING", {"forceInput": True})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "CONTROL_INFOS", "LORA_INFOS")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "params", "seed", "CONTROLNET_INFOS", "LORA_INFOS")

    FUNCTION = "get_webui_params_info"

    # CATEGORY = "🐼malio/webui/info"

    def get_webui_params_info(self, webui_params_info:str):
        """提取webui生成的图片信息"""

        embeddings_file_paths = folder_paths.get_filename_list("embeddings")  # 本地的controlnet文件
        # 去除后缀
        embeddings_name_list = [item.split(".")[0] for item in embeddings_file_paths]

        # return {
        #     "positive_text": positive_text,
        #     "negative_text": negative_text,
        #     "params": params_dict,
        #     "loras" : loras,
        #     "controlnets": params_dict["controlnets"]
        # }
        try:
            info_dict = extract_info_from_webui_img(webui_params_info)
            positive_text = info_dict["positive_text"]
            negative_text = info_dict["negative_text"]
            params_dict = info_dict["params"]
            loras = info_dict["loras"]
            controlnets = info_dict["controlnets"]

            # 替换embeddings, webui中的embeddings是不带embedding:前缀的, comfyui需要带前缀
            for embed_name in embeddings_name_list:
                if embed_name in negative_text:
                    print(f"发现embeddings: {embed_name}")
                    negative_text = negative_text.replace(embed_name, f"embedding:{embed_name}")
        except Exception as e:
            print(f"提取webui信息出错: {e}")
            return (None, None, None, None)
       
        return (positive_text, negative_text, json.dumps(params_dict), int(params_dict["Seed"]), controlnets, loras)


class Maliooo_Get_Controlnet_Stack:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                {
                    "image": ("IMAGE",),
                    "controlnet_infos": ("CONTROL_INFOS",), 
                }
        }
    RETURN_TYPES = ("CONTROL_NET_STACK", "STRING", "IMAGE")
    RETURN_NAMES = ("controlnet_stack", "show_help", "image")
    OUTPUT_IS_LIST = (False,False,True)
    FUNCTION = "get_controlnet_stacks"

    CATEGORY = "conditioning"

    def get_controlnet_stacks(self, image, controlnet_infos):
        controlnet_file_paths = folder_paths.get_filename_list("controlnet")  # 本地的controlnet文件
        controlnet_list = []
        imgs = []
        for index, controlnet in enumerate(controlnet_infos):
            if index >= 3:  # 最多只能有3个controlnet
                break

            # ============= 提取controlnet的预处理器 =============
            preprocessor = controlnet["module"]  # 预处理器
            if preprocessor in WEBUI_2_COMFYUI_PREPROCESS:
                preprocessor = WEBUI_2_COMFYUI_PREPROCESS[preprocessor]
            else:
                flag = False
                for key in WEBUI_2_COMFYUI_PREPROCESS.keys():
                    if key in preprocessor:
                        preprocessor = WEBUI_2_COMFYUI_PREPROCESS[key]
                        flag = True
                        break
                if not flag:
                    print(f"webui中controlnet预处理器找不到，不支持的预处理器: {preprocessor}")
                    continue

            if preprocessor not in WEBUI_2_COMFYUI_PREPROCESS.values():
                print(f"webui中controlnet预处理器找不到，不支持的预处理器: {preprocessor}")
                continue  # 跳过不支持的预处理器
            
            controlnet_model_name = controlnet["model"].lower()
            if "ip-adapter" in controlnet_model_name:
                continue # 跳过ip-adapter
            pixel_perfect = controlnet["pixel perfect"]  # 是否完美像素
            if pixel_perfect.lower() == "false":
                resolution = 512
            
            controlnet_model_name = controlnet["model"].lower()
            if "[" in controlnet_model_name:
                controlnet_model_name = controlnet_model_name.split("[")[0].strip()

            # 遍历从信息中提取得到的controlnet
            for controlnet_file_path in controlnet_file_paths:    # 遍历本地的controlnet文件


                if controlnet_model_name in controlnet_file_path.lower():
                    # 获得controlnet的预处理器得到的预处理图片
                    preprocess_image = apply_preprocessor(image=image, preprocessor=preprocessor, resolution=resolution)
                    imgs.append(preprocess_image)
                    
                    # controlnet_list.extend([(controlnet_3, image_3, controlnet_strength_3, start_percent_3, end_percent_3)])
                    controlnet_list.append(
                        (controlnet_file_path, 
                         preprocess_image, 
                         float(controlnet["weight"]), 
                         float(controlnet["guidance start"]), 
                         float(controlnet["guidance end"])
                         )
                    )
                    break

        show_help = "显示帮助信息"

        #  return (controlnet_list) 如果只返回一个会出错，不知道为啥
        return (controlnet_list, show_help, imgs)  
    

class Maliooo_Get_Lora_Stack:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                {
                    "lora_infos": ("LORA_INFOS",), 
                }
        }
    RETURN_TYPES = ("LORA_STACK", "STRING")
    RETURN_NAMES = ("loras_stack", "show_help")
    # OUTPUT_IS_LIST = (False,False)
    FUNCTION = "get_lora_stacks"

    CATEGORY = "conditioning"

    def get_lora_stacks(self, lora_infos):
        loras_file_paths = folder_paths.get_filename_list("loras")  # 本地的controlnet文件
        
        loras_list = []
        for index, lora_item in enumerate(lora_infos):
            lora_clip_weight = 1.0
            lora_name, lora_weight = lora_item["lora_name"], float(lora_item["lora_weight"])
            
            if f"{lora_name}.safetensors" in loras_file_paths:
                lora_name = f"{lora_name}.safetensors"
                loras_list.append((lora_name, lora_weight, lora_clip_weight))
            else:
                print(f"未找到webui生成信息中的 lora: {lora_name}")

    
        show_help = "显示帮助信息"

        #  return (controlnet_list) 如果只返回一个会出错，不知道为啥
        return (loras_list, show_help)  