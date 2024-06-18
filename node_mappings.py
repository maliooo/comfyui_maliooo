from .nodes.malio_controlnet import Malio_Get_Controlnet_Name, Malio_ControlNetStack_By_Name
from .nodes.malio_lora import Malio_Get_Lora_Name, Malio_LoRAStack_By_Name
from .nodes.malio_image import Malio_ImageScale_Side, Malio_ImageScale_Side_Cascade, Maliooo_LoadImageFromUrl, Maliooo_LoadImageByPathSequence
from .nodes.malio_nsfw import Malio_NSFWDetection
from .nodes.malio_image_info import Malio_Webui_Info_Params, Maliooo_Get_Controlnet_Stack,  Maliooo_Get_Lora_Stack
from .nodes.malio_comfy import Malio_CheckpointLoaderSimple, Malio_LoadImage


NODE_CLASS_MAPPINGS = {
    "Maliooo_Get_Controlnet_Name": Malio_Get_Controlnet_Name,
    "Maliooo_ControlNetStack_By_Name": Malio_ControlNetStack_By_Name,
    "Maliooo_Get_Lora_Name": Malio_Get_Lora_Name,
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
    "Maliooo_LoadImageByPathSequence": Maliooo_LoadImageByPathSequence

}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Maliooo_Get_Controlnet_Name": "Malio Get Controlnet Name",
    "Maliooo_ControlNetStack_By_Name": "Malio ControlNet Stack By Name",
    "Maliooo_Get_Lora_Name": "Malio Get Lora Name",
    "Maliooo_LoRAStack_By_Name": "Malio LoRA Stack By Name",
    "Maliooo_ImageScale_Side": "Malio Image Scale Side 按边缩放图片",
    "Maliooo_ImageScale_Side_Cascade": "Malio Image Scale Side 缩放到Cascade比例",
    "Maliooo_NSFWDetection": "Malio NSFW Detection",
    "Maliooo_LoadImageFromUrl": "从URL加载图片, 输出info信息",
    "Maliooo_Webui_ImageInfo": "输入webui的info信息, 提取参数",
    "Maliooo_Get_Controlnet_Stack": "根据webui的生成信息，得到controlnet_stack列表",
    "Maliooo_Get_Lora_Stack": "根据webui的生成信息，得到lora_stack列表",
    "Maliooo_CheckpointLoaderSimple": "Malio load_checkpoint模型加载器",
    "Maliooo_LoadImage": "Malio Load Image加载图片",
    "Maliooo_LoadImageByPathSequence": "Malio Load Image By Path Sequence根据文件夹顺序加载图片"
}