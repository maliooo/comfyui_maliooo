from .nodes.malio_controlnet import Malio_Get_Controlnet_Name, Malio_ControlNetStack_By_Name
from .nodes.malio_lora import Malio_Get_Lora_Name, Malio_LoRAStack_By_Name
from .nodes.malio_image import Malio_ImageScale_Side
from .nodes.malio_nsfw import Malio_NSFWDetection

NODE_CLASS_MAPPINGS = {
    "Maliooo_Get_Controlnet_Name": Malio_Get_Controlnet_Name,
    "Maliooo_ControlNetStack_By_Name": Malio_ControlNetStack_By_Name,
    "Maliooo_Get_Lora_Name": Malio_Get_Lora_Name,
    "Maliooo_LoRAStack_By_Name": Malio_LoRAStack_By_Name,
    "Maliooo_ImageScale_Side": Malio_ImageScale_Side,
    "Maliooo_NSFWDetection": Malio_NSFWDetection
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Maliooo_Get_Controlnet_Name": "Get Controlnet Name",
    "Maliooo_ControlNetStack_By_Name": "ControlNet Stack By Name",
    "Maliooo_Get_Lora_Name": "Get Lora Name",
    "Maliooo_LoRAStack_By_Name": "LoRA Stack By Name",
    "Maliooo_ImageScale_Side": "Image Scale Side 按边缩放图片",
    "Maliooo_NSFWDetection": "NSFW Detection"
}