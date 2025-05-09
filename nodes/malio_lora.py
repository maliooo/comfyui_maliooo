import folder_paths
import os
import sys
import comfy.controlnet
import comfy.sd
import folder_paths
"""
一些lora的操作
"""


class Malio_Get_Lora_Name:
    """从lora文件夹中获取选择的lora的名称"""

    # lora_names = ["None"] + folder_paths.get_filename_list("loras")

    @classmethod
    def INPUT_TYPES(cls):
        #controlnets = ["None"]
        return {
            "required": {
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
            }
        }
    
    RETURN_TYPES = ("STRING",)

    
    FUNCTION = "get_lora_name"

    #OUTPUT_NODE = False

    CATEGORY = "🐼malio/lora/get_lora_name"

    def get_lora_name(self, lora_name):
        if isinstance(lora_name, str):
            lora_name = lora_name.strip()
        return (lora_name, )


class Malio_Get_Lora_Name_And_Keyword:
    """从lora文件夹中获取选择的lora的名称, 并且可以设置触发词"""

    @classmethod
    def INPUT_TYPES(cls):
        # 得到controlnet文件夹中所有的controlnet文件的名称
        # lora_names = ["None"] + folder_paths.get_filename_list("loras")
        #controlnets = ["None"]
        return {
            "required": {
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "触发词": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_name", "触发词")

    
    FUNCTION = "get_lora_name"

    #OUTPUT_NODE = False

    CATEGORY = "🐼malio/lora/get_lora_name_and_keyword"

    def get_lora_name(self, lora_name, 触发词):
        if isinstance(lora_name, str):
            lora_name = lora_name.strip()
        return (lora_name, 触发词)


class Malio_LoRAStack_By_Name:

    @classmethod
    def INPUT_TYPES(cls):
    
        lora_names = ["None"] + folder_paths.get_filename_list("loras")
        
        return {"required": {
                    "switch_1": ("BOOLEAN", {"default": False}),
                    "lora_name_1": ("STRING",{"default": "None"}),
                    "model_weight_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "clip_weight_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "switch_2": ("BOOLEAN", {"default": False}),
                    "lora_name_2": ("STRING",{"default": "None"}),
                    "model_weight_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "clip_weight_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "switch_3": ("BOOLEAN", {"default": False}),
                    "lora_name_3": ("STRING",{"default": "None"}),
                    "model_weight_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "clip_weight_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                },
                "optional": {"lora_stack": ("LORA_STACK",)
                },
        }

    RETURN_TYPES = ("LORA_STACK", "STRING", )
    RETURN_NAMES = ("LORA_STACK", "show_help", )
    FUNCTION = "lora_stacker"
    CATEGORY = "🐼malio/lora/lora_stack"

    def lora_stacker(self, lora_name_1, model_weight_1, clip_weight_1, switch_1, lora_name_2, model_weight_2, clip_weight_2, switch_2, lora_name_3, model_weight_3, clip_weight_3, switch_3, lora_stack=None):

        # Initialise the list
        lora_list=list()
        
        if lora_stack is not None:
            lora_list.extend([l for l in lora_stack if l[0] != "None"])
        
        if lora_name_1 != "None" and  switch_1:
            lora_list.extend([(lora_name_1, model_weight_1, clip_weight_1)]),

        if lora_name_2 != "None" and  switch_2:
            lora_list.extend([(lora_name_2, model_weight_2, clip_weight_2)]),

        if lora_name_3 != "None" and  switch_3:
            lora_list.extend([(lora_name_3, model_weight_3, clip_weight_3)]),
           
        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/LoRA-Nodes#cr-lora-stack"           

        return (lora_list, show_help, )