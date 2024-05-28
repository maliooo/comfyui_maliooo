import folder_paths
import os
import sys
import comfy.controlnet
import comfy.sd
import folder_paths
"""
一些controlnet的操作
"""


class Malio_Get_Controlnet_Name:
    """从controlnet文件夹中获取选择的controlnet的名称"""

    # 得到controlnet文件夹中所有的controlnet文件的名称
    controlnets = ["None"] + folder_paths.get_filename_list("controlnet")

    @classmethod
    def INPUT_TYPES(cls):
        #controlnets = ["None"]
        return {
            "required": {
                "controlnet": (cls.controlnets,),
            }
        }
    
    RETURN_TYPES = ("STRING",)

    
    FUNCTION = "get_controlnet_name"

    #OUTPUT_NODE = False

    CATEGORY = "malio/controlnet_name"

    def get_controlnet_name(self, controlnet):
        if isinstance(controlnet, str):
            controlnet = controlnet.strip()
        return (controlnet, )

class Malio_ControlNetStack_By_Name:

    @classmethod
    def INPUT_TYPES(cls):
        #controlnets = ["None"]
        return {"required": {
                },
                "optional": {
                    "switch_1": ("BOOLEAN", {"default": False}),
                    "controlnet_1_name": ("STRING",{"default": "None"}),
                    "controlnet_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "start_percent_1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_percent_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    #
                    "switch_2": ("BOOLEAN", {"default": True}),
                    "controlnet_2_name": ("STRING",{"default": "None"}),
                    "controlnet_strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "start_percent_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_percent_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    #
                    "switch_3": ("BOOLEAN", {"default": True}),
                    "controlnet_3_name": ("STRING",{"default": "None"}),
                    "controlnet_strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "start_percent_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_percent_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    #
                    "image_1": ("IMAGE",),
                    "image_2": ("IMAGE",),
                    "image_3": ("IMAGE",),
                    "controlnet_stack": ("CONTROL_NET_STACK",)
                },
        }

    RETURN_TYPES = ("CONTROL_NET_STACK", "STRING", )
    RETURN_NAMES = ("CONTROLNET_STACK", "show_help", )
    FUNCTION = "controlnet_stacker"
    CATEGORY = "malio/controlnet"

    def controlnet_stacker(self, 
                           switch_1, controlnet_1_name, controlnet_strength_1, start_percent_1, end_percent_1,
                           switch_2, controlnet_2_name, controlnet_strength_2, start_percent_2, end_percent_2,
                           switch_3, controlnet_3_name, controlnet_strength_3, start_percent_3, end_percent_3,
                           image_1=None, image_2=None, image_3=None, controlnet_stack=None):

        # Initialise the list
        controlnet_list= []
        
        if controlnet_stack is not None:
            controlnet_list.extend([l for l in controlnet_stack if l[0] != "None"])
        
        if controlnet_1_name != "None" and  switch_1 and image_1 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet_1_name)
            controlnet_1 = comfy.controlnet.load_controlnet(controlnet_path)
            controlnet_list.extend([(controlnet_1, image_1, controlnet_strength_1, start_percent_1, end_percent_1)]),

        if controlnet_2_name != "None" and  switch_2 and image_2 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet_2_name)
            controlnet_2 = comfy.controlnet.load_controlnet(controlnet_path)
            controlnet_list.extend([(controlnet_2, image_2, controlnet_strength_2, start_percent_2, end_percent_2)]),

        if controlnet_3_name != "None" and  switch_3 and image_3 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet_3_name)
            controlnet_3 = comfy.controlnet.load_controlnet(controlnet_path)
            controlnet_list.extend([(controlnet_3, image_3, controlnet_strength_3, start_percent_3, end_percent_3)]),

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/ControlNet-Nodes#cr-multi-controlnet-stack"

        return (controlnet_list, show_help, )

