# 参考 custom_nodes\comfyui-art-venture\modules\controlnet\preprocessors.py,  结点：AV_ControlNetPreprocessor
import os
import math
from typing import Dict

import folder_paths

def load_module(module_path, module_name=None):
    import importlib.util

    if module_name is None:
        module_name = os.path.basename(module_path)
        if os.path.isdir(module_path):
            module_path = os.path.join(module_path, "__init__.py")

    module_spec = importlib.util.spec_from_file_location(module_name, module_path)

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module


custom_nodes = folder_paths.get_folder_paths("custom_nodes")
preprocessors_dir_names = ["ControlNetPreprocessors", "comfyui_controlnet_aux"]

# 得到custom_nodes 下comfyui_controlnet_aux 的 __init__.py 文件 中的 NODE_CLASS_MAPPINGS
control_net_preprocessors = {}

try:
    module_path = None

    for custom_node in custom_nodes:
        custom_node = (
            custom_node if not os.path.islink(custom_node) else os.readlink(custom_node)
        )
        for module_dir in preprocessors_dir_names:
            if module_dir in os.listdir(custom_node):
                module_path = os.path.abspath(os.path.join(custom_node, module_dir))
                break

    if module_path is None:
        raise Exception("Could not find ControlNetPreprocessors nodes")

    module = load_module(module_path)  # 得到custom_nodes 下comfyui_controlnet_aux 的 __init__.py 文件
    print("Loaded ControlNetPreprocessors nodes from", module_path)


    # 得到custom_nodes 下comfyui_controlnet_aux 的 __init__.py 文件 中的 NODE_CLASS_MAPPINGS
    nodes: Dict = getattr(module, "NODE_CLASS_MAPPINGS")

    if "CannyEdgePreprocessor" in nodes:
        control_net_preprocessors["canny"] = (
            nodes["CannyEdgePreprocessor"],
            [100, 200],  # 这里是除了image 和 resolusion 之外的其他参数
        )
    if "LineArtPreprocessor" in nodes:
        control_net_preprocessors["lineart"] = (
            nodes["LineArtPreprocessor"],
            ["disable"],
        )
        control_net_preprocessors["lineart_coarse"] = (
            nodes["LineArtPreprocessor"],
            ["enable"],
        )

    # 添加的
    if "LineartStandardPreprocessor" in nodes:
        control_net_preprocessors["lineart_standard"] = (
            nodes["LineartStandardPreprocessor"],
            [6.0, 8],  # guassian_sigma, intensity_threshold
        )
    if "AnimeLineArtPreprocessor" in nodes:
        control_net_preprocessors["lineart_anime"] = (
            nodes["AnimeLineArtPreprocessor"],
            [],
        )
    if "Manga2Anime_LineArt_Preprocessor" in nodes:
        control_net_preprocessors["lineart_manga"] = (
            nodes["Manga2Anime_LineArt_Preprocessor"],
            [],
        )
    if "ScribblePreprocessor" in nodes:
        control_net_preprocessors["scribble"] = (nodes["ScribblePreprocessor"], [])
    if "FakeScribblePreprocessor" in nodes:
        control_net_preprocessors["scribble_hed"] = (
            nodes["FakeScribblePreprocessor"],
            ["enable"],
        )
    if "HEDPreprocessor" in nodes:
        control_net_preprocessors["hed"] = (nodes["HEDPreprocessor"], ["disable"])
        control_net_preprocessors["hed_safe"] = (nodes["HEDPreprocessor"], ["enable"])
    if "PiDiNetPreprocessor" in nodes:
        control_net_preprocessors["pidi"] = (
            nodes["PiDiNetPreprocessor"],
            ["disable"],
        )
        control_net_preprocessors["pidi_safe"] = (
            nodes["PiDiNetPreprocessor"],
            ["enable"],
        )
    if "M-LSDPreprocessor" in nodes:
        control_net_preprocessors["mlsd"] = (nodes["M-LSDPreprocessor"], [0.1, 0.1])
    if "OpenposePreprocessor" in nodes:
        control_net_preprocessors["openpose"] = (
            nodes["OpenposePreprocessor"],
            ["enable", "enable", "enable"],
        )
        control_net_preprocessors["pose"] = control_net_preprocessors["openpose"]
    if "DWPreprocessor" in nodes:
        control_net_preprocessors["dwpose"] = (
            nodes["DWPreprocessor"],
            ["enable", "enable", "enable", "yolox_l.onnx", "dw-ll_ucoco_384.onnx"],
        )
        # use DWPreprocessor for pose by default if available
        control_net_preprocessors["pose"] = control_net_preprocessors["dwpose"]
    if "BAE-NormalMapPreprocessor" in nodes:
        control_net_preprocessors["normalmap_bae"] = (
            nodes["BAE-NormalMapPreprocessor"],
            [],
        )
    if "MiDaS-NormalMapPreprocessor" in nodes:
        control_net_preprocessors["normalmap_midas"] = (
            nodes["MiDaS-NormalMapPreprocessor"],
            [math.pi * 2.0, 0.1],
        )
    if "MiDaS-DepthMapPreprocessor" in nodes:
        control_net_preprocessors["depth_midas"] = (
            nodes["MiDaS-DepthMapPreprocessor"],
            [math.pi * 2.0, 0.4],
        )
    if "Zoe-DepthMapPreprocessor" in nodes:
        control_net_preprocessors["depth"] = (nodes["Zoe-DepthMapPreprocessor"], [])
        control_net_preprocessors["depth_zoe"] = (nodes["Zoe-DepthMapPreprocessor"], [])
    

    # 添加的
    if "LeReS-DepthMapPreprocessor" in nodes:
        control_net_preprocessors["depth_leres"] = (
            nodes["LeReS-DepthMapPreprocessor"],
            [0.0, 0.0, "disable"],  # rm_nearest, rm_background, boost
        )

    if "OneFormer-COCO-SemSegPreprocessor" in nodes:
        control_net_preprocessors["seg_ofcoco"] = (
            nodes["OneFormer-COCO-SemSegPreprocessor"],
            [],
        )
    if "OneFormer-ADE20K-SemSegPreprocessor" in nodes:
        control_net_preprocessors["seg_ofade20k"] = (
            nodes["OneFormer-ADE20K-SemSegPreprocessor"],
            [],
        )
    if "UniFormer-SemSegPreprocessor" in nodes:
        control_net_preprocessors["seg_ufade20k"] = (
            nodes["UniFormer-SemSegPreprocessor"],
            [],
        )

except Exception as e:
    print(e)


class DummyPreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }
    
    FUNCTION = "process"

    def process(self, image):
        return (image,)
