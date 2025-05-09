
WEIGHT_TYPES = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer', 'composition', 'strong style transfer', 'style and composition', 'style transfer precise', 'composition precise']

class Malio_Ipadapter_Selector:
    """malio Ipadapter 参数选择器"""

    # 参考：ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                {
                    "loader_preset": (['LIGHT - SD1.5 only (low strength)', 'STANDARD (medium strength)', 'VIT-G (medium strength)', 'PLUS (high strength)', 'PLUS FACE (portraits)', 'FULL FACE - SD1.5 only (portraits stronger)'], ),
                    "weight_type_adv": (WEIGHT_TYPES, ),
                    "embeds_scaling_adv": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
                }
        }

    RETURN_TYPES = ("STRING", "STRING","STRING", "LIST", "LIST", "LIST")
    RETURN_NAMES = ("loader_preset", "weight_type_adv", "embeds_scaling", "loader_preset_list", "weight_type_adv_list", "embeds_scaling_list")
    FUNCTION = "ipadapter_select"

    CATEGORY = "🐼malio/ipadapter/select"

    def ipadapter_select(self, loader_preset, weight_type_adv, embeds_scaling_adv):
        loader_preset_list = ['LIGHT - SD1.5 only (low strength)', 'STANDARD (medium strength)', 'VIT-G (medium strength)', 'PLUS (high strength)', 'PLUS FACE (portraits)', 'FULL FACE - SD1.5 only (portraits stronger)']
        weight_type_adv_list = WEIGHT_TYPES
        embeds_scaling_list = ['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty']
        return (loader_preset, weight_type_adv, embeds_scaling_adv, loader_preset_list, weight_type_adv_list, embeds_scaling_list, )