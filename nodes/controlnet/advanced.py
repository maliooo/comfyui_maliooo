import os
from typing import Dict

import folder_paths
import comfy.sd
import comfy.controlnet

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
advanced_cnet_dir_names = ["AdvancedControlNet", "ComfyUI-Advanced-ControlNet"]


def comfy_load_controlnet(control_net_name: str, **_):
    controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
    return comfy.controlnet.load_controlnet(controlnet_path)


try:
    module_path = None

    for custom_node in custom_nodes:
        custom_node = custom_node if not os.path.islink(custom_node) else os.readlink(custom_node)
        for module_dir in advanced_cnet_dir_names:
            if module_dir in os.listdir(custom_node):
                module_path = os.path.abspath(os.path.join(custom_node, module_dir))
                break

    if module_path is None:
        raise Exception("Could not find AdvancedControlNet nodes")

    module = load_module(module_path)
    print("Loaded AdvancedControlNet nodes from", module_path)

    nodes: Dict = getattr(module, "NODE_CLASS_MAPPINGS")
    ControlNetLoaderAdvanced = nodes["ControlNetLoaderAdvanced"]

    loader = ControlNetLoaderAdvanced()

    def comfy_load_controlnet(control_net_name: str, timestep_keyframe=None):
        return loader.load_controlnet(control_net_name, timestep_keyframe=timestep_keyframe)[0]

except Exception as e:
    print(e)
