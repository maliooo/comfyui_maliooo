import folder_paths
import comfy.controlnet


class Malio_ControlNet_Names_Selector:
    """从controlnet文件夹中获取选择的controlnet的名称"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "control_net_name": (folder_paths.get_filename_list("controlnet"), )}}

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("control_net_name", )
    FUNCTION = "load_controlnet"

    CATEGORY = "🐼malio/loader/controlnet_select"

    def load_controlnet(self, control_net_name):
        return (control_net_name, )