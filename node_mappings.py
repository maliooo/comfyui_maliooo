from .nodes.malio_controlnet import Malio_Get_Controlnet_Name, Malio_ControlNetStack_By_Name
from .nodes.malio_lora import Malio_Get_Lora_Name, Malio_LoRAStack_By_Name, Malio_Get_Lora_Name_And_Keyword
from .nodes.malio_image import Malio_ImageScale_Side, Malio_ImageScale_Side_Cascade, Maliooo_LoadImageFromUrl, Maliooo_LoadImageByPathSequence, Maliooo_LoadImageByCsv, Maliooo_LoadImageByCsv_V2
from .nodes.malio_image import Maliooo_LoadImageByPath, Malio_Repeat_and_Tile_Image, Malio_SaveImage, Malio_SD35_Image_Resize
from .nodes.malio_nsfw import Malio_NSFWDetection
from .nodes.malio_image_info import Malio_Webui_Info_Params, Maliooo_Get_Controlnet_Stack,  Maliooo_Get_Lora_Stack
from .nodes.malio_comfy import Malio_CheckpointLoaderSimple, Malio_LoadImage
from .nodes.malio_image_mask import Malio_Image_Watermark_Mask_v0, Malio_Image_Watermark_Mask_yolov8, Malio_Image_Watermark_EasyOCR
from .nodes.malio_image_jingpin import Maliooo_Get_SuappImage, Maliooo_Get_ShengJingKeJi, Maliooo_Get_JianZhuXueZhang
from .nodes.malio_sdxl import Malio_SDXL_Image_Resize, Malio_SDXL_Image_Resize_64, Malio_SDXL_Image_Resize_1536
from .nodes.malio_florence import Maliooo_Florence_Str_Postprocess, Maliooo_Florence_Str_Color_Postprocess
from .nodes.malio_photoshop import Malio_ImageSmartSharpen
from .nodes.malio_controlnet import Malio_ControlNet_Names_Selector
from .nodes.malio_ipadapter import Malio_Ipadapter_Selector
from .nodes.malio_aux import Malio_OneFormer_ADE20K_SemSegPreprocessor
from .nodes.malio_tags import Malio_Merge_Tags_By_Onerformer_ADE20K, Malio_LLM_By_Qwen_VL, Malio_LLM_Answer
from .nodes.malio_llm import Malio_ARK_LLM_Answer
from .nodes.malio_bbox import Malio_BBOXES
from .nodes.malio_image_info import Malio_Comfy_Info_Params


NODE_CLASS_MAPPINGS = {
    "Maliooo_Get_Controlnet_Name": Malio_Get_Controlnet_Name,
    "Maliooo_ControlNetStack_By_Name": Malio_ControlNetStack_By_Name,
    "Maliooo_Get_Lora_Name": Malio_Get_Lora_Name,
    "Maliooo_Get_Lora_Name_And_Keyword": Malio_Get_Lora_Name_And_Keyword,
    "Maliooo_LoRAStack_By_Name": Malio_LoRAStack_By_Name,
    "Maliooo_ImageScale_Side": Malio_ImageScale_Side,
    "Maliooo_ImageScale_Side_Cascade": Malio_ImageScale_Side_Cascade,
    "Maliooo_NSFWDetection": Malio_NSFWDetection,
    "Maliooo_LoadImageFromUrl": Maliooo_LoadImageFromUrl,
    "Maliooo_Webui_ImageInfo": Malio_Webui_Info_Params,
    "Maliooo_Get_Controlnet_Stack": Maliooo_Get_Controlnet_Stack,
    "Maliooo_Get_Lora_Stack": Maliooo_Get_Lora_Stack,
    "Maliooo_CheckpointLoaderSimple": Malio_CheckpointLoaderSimple,
    "Maliooo_LoadImage": Malio_LoadImage,
    "Maliooo_LoadImageByPathSequence": Maliooo_LoadImageByPathSequence,
    "Maliooo_LoadImageByCsv": Maliooo_LoadImageByCsv,
    "Maliooo_LoadImageByCsv_V2": Maliooo_LoadImageByCsv_V2,
    "Maliooo_LoadImageByPath": Maliooo_LoadImageByPath,
    "Maliooo_Repeat_and_Tile_Image": Malio_Repeat_and_Tile_Image,
    "Malio_Image_Watermark_Mask_V0": Malio_Image_Watermark_Mask_v0,
    "Malio_Image_Watermark_Mask_yolov8": Malio_Image_Watermark_Mask_yolov8,
    "Malio_Image_Watermark_EasyOCR": Malio_Image_Watermark_EasyOCR,
    "Malio_SaveImage": Malio_SaveImage,
    "Maliooo_Get_SuappImage": Maliooo_Get_SuappImage,
    "Maliooo_Get_ShengJingKeJi": Maliooo_Get_ShengJingKeJi,
    "Maliooo_Get_JianZhuXueZhang": Maliooo_Get_JianZhuXueZhang,
    "Malio_SDXL_Image_Resize": Malio_SDXL_Image_Resize,
    "Malio_SDXL_Image_Resize_64": Malio_SDXL_Image_Resize_64,
    "Malio_SDXL_Image_Resize_1536": Malio_SDXL_Image_Resize_1536,
    "Malio_SD35_Image_Resize": Malio_SD35_Image_Resize,
    "Maliooo_Florence_Str_Postprocess": Maliooo_Florence_Str_Postprocess,
    "Maliooo_Florence_Str_Color_Postprocess": Maliooo_Florence_Str_Color_Postprocess,
    "Malio_ImageSmartSharpen": Malio_ImageSmartSharpen,
    "Malio_ControlNet_Names_Selector": Malio_ControlNet_Names_Selector,
    "Malio_Ipadapter_Selector": Malio_Ipadapter_Selector,
    "Malio_OneFormer_ADE20K_SemSegPreprocessor": Malio_OneFormer_ADE20K_SemSegPreprocessor,
    "Malio_Merge_Tags_By_Onerformer_ADE20K": Malio_Merge_Tags_By_Onerformer_ADE20K,
    "Malio_LLM_By_Qwen_VL": Malio_LLM_By_Qwen_VL,
    "Malio_LLM_Answer": Malio_LLM_Answer,
    "Malio_ARK_LLM_Answer": Malio_ARK_LLM_Answer,
    "Malio_BBOXES": Malio_BBOXES,
    "Malio_Comfy_Info_Params": Malio_Comfy_Info_Params,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Maliooo_Get_Controlnet_Name": "Malio Get Controlnet Name",
    "Maliooo_ControlNetStack_By_Name": "Malio ControlNet Stack By Name",
    "Maliooo_Get_Lora_Name": "Malio Get Lora Name 获取lora名称",
    "Maliooo_Get_Lora_Name_And_Keyword": "Malio Get Lora Name And Keyword 获取lora名称和触发词",
    "Maliooo_LoRAStack_By_Name": "Malio LoRA Stack By Name",
    "Maliooo_ImageScale_Side": "Malio Image Scale Side 按边缩放图片",
    "Maliooo_ImageScale_Side_Cascade": "Malio Image Scale Side 缩放到Cascade比例",
    "Maliooo_NSFWDetection": "Malio NSFW Detection",
    "Maliooo_LoadImageFromUrl": "从URL加载图片, 输出info信息",
    "Maliooo_Webui_ImageInfo": "输入webui的info信息, 提取参数",
    "Maliooo_Get_Controlnet_Stack": "根据webui的生成信息，得到controlnet_stack列表",
    "Maliooo_Get_Lora_Stack": "根据webui的生成信息，得到lora_stack列表",
    "Maliooo_CheckpointLoaderSimple": "Malio load_checkpoint 模型加载器",
    "Maliooo_LoadImage": "Malio Load Image加载图片",
    "Maliooo_LoadImageByPathSequence": "Malio LoadImageByPathSequence 根据文件夹顺序加载图片",
    "Maliooo_LoadImageByCsv": "Malio LoadImageByCsv 根据csv加载图片",
    "Maliooo_LoadImageByCsv_V2": "Malio LoadImageByCsv_V2 根据csv加载图片, 获取同一个task_id的其他生成图片",
    "Maliooo_LoadImageByPath": "Malio LoadImageByPath 根据路径加载单张图片",
    "Maliooo_Repeat_and_Tile_Image": "Malio Repeat and Tile Image 重复平铺图片",
    "Malio_Image_Watermark_Mask_V0": "Malio Image Watermark Mask 获取输入图片的水印遮罩_mask版本",
    "Malio_Image_Watermark_Mask_yolov8": "Malio Image Watermark Mask yolov8 获取输入图片的水印遮罩_yolov8版本",
    "Malio_Image_Watermark_EasyOCR": "Malio Image Watermark EasyOCR 获取输入图片的文字遮罩",
    "Malio_SaveImage": "Malio Save Image 保存图片",
    "Maliooo_Get_SuappImage": "Malio Get Suapp Image 获取suapp图片",
    "Maliooo_Get_ShengJingKeJi": "Malio Get ShengJingKeJi 获取生境科技图片",
    "Maliooo_Get_JianZhuXueZhang": "Malio Get JianZhuXueZhang 获取建筑学长图片",
    "Malio_SDXL_Image_Resize": "Malio SDXL Image Resize 缩放到SDXL比例",
    "Malio_SDXL_Image_Resize_64": "Malio SDXL 64 Image Resize 缩放到SDXL 64比例",
    "Malio_SDXL_Image_Resize_1536": "Malio SDXL 1536 Image Resize 缩放到SDXL 1536",
    "Malio_SD35_Image_Resize": "Malio SD35 Image Resize 缩放到SD35比例",
    "Maliooo_Florence_Str_Postprocess": "Malio Florence 字符串后处理",
    "Maliooo_Florence_Str_Color_Postprocess": "Malio Florence 字符串后处理, 处理标注和颜色信息",
    "Malio_ImageSmartSharpen": "Malio Image Smart Sharpen 自动图片锐化",
    "Malio_ControlNet_Names_Selector": "Malio ControlNet Names Selector 选择controlnet名称",
    "Malio_Ipadapter_Selector": "Malio Ipadapter Selector 选择ipadapter参数",
    "Malio_OneFormer_ADE20K_SemSegPreprocessor": "Malio OneFormer ADE20K Segmentation 获取分割图和标签",
    "Malio_Merge_Tags_By_Onerformer_ADE20K": "Malio_Merge_Tags_By_Onerformer_ADE20K 合并SEG的标签结果",
    "Malio_LLM_By_Qwen_VL": "Malio_LLM_By_Qwen_VL 通过Qwen-VL 得到描述",
    "Malio_LLM_Answer": "Malio_LLM_Answer 通过Qwen LLM得到描述",
    "Malio_ARK_LLM_Answer": "Malio_ARK_LLM_Answer 通过火山ARK LLM得到描述",
    "Malio_BBOXES": "Malio_BBOXES 使用python的eval函数生成bboxes",
    "Malio_Comfy_Info_Params": "Malio_Comfy_Info_Params 根据url提取comfy生成的图片信息",
}