import folder_paths
import os
import sys
import comfy.controlnet
import comfy.sd
import folder_paths

class Example:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "int_field": ("INT", {
                    "default": 0, 
                    "min": 0, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "float_field": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number"}),
                "print_to_screen": (["enable", "disable"],),
                "string_field": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Hello World!"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(self, image, string_field, int_field, float_field, print_to_screen):
        if print_to_screen == "enable":
            print(f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {int_field}
                float_field: {float_field}
            """)
        #do some processing on the image, in this example I just invert it
        image = 1.0 - image
        return (image,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique

class Maliooo_Get_Controlnet_Name:
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

    CATEGORY = "maliooo/controlnet_name"

    def get_controlnet_name(self, controlnet):
        if controlnet == "None":
            return ("None",)
        else:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet)
        return (controlnet_path,)

class Maliooo_ControlNetStack_By_Name:

    controlnets = ["None"] + folder_paths.get_filename_list("controlnet")
    
    @classmethod
    def INPUT_TYPES(cls):
        #controlnets = ["None"]
        return {"required": {
                },
                "optional": {
                    "switch_1": ("BOOLEAN", {"default": False}),
                    "controlnet_1_path": ("STRING",{"default": "None"}),
                    "controlnet_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "start_percent_1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_percent_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    #
                    "switch_2": ("BOOLEAN", {"default": True}),
                    "controlnet_2_path": ("STRING",{"default": "None"}),
                    "controlnet_strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "start_percent_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_percent_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    #
                    "switch_3": ("BOOLEAN", {"default": True}),
                    "controlnet_3_path": ("STRING",{"default": "None"}),
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
    CATEGORY = "maliooo/controlnet"

    def controlnet_stacker(self, switch_1, controlnet_1_path, controlnet_strength_1, start_percent_1, end_percent_1,
                           switch_2, controlnet_2_path, controlnet_strength_2, start_percent_2, end_percent_2,
                           switch_3, controlnet_3_path, controlnet_strength_3, start_percent_3, end_percent_3,
                           image_1=None, image_2=None, image_3=None, controlnet_stack=None):

        # Initialise the list
        controlnet_list= []
        
        if controlnet_stack is not None:
            controlnet_list.extend([l for l in controlnet_stack if l[0] != "None"])
        
        if controlnet_1_path != "None" and  switch_1 and image_1 is not None:
            # controlnet_path = folder_paths.get_full_path("controlnet", controlnet_1)
            controlnet_1 = comfy.controlnet.load_controlnet(controlnet_1_path)
            controlnet_list.extend([(controlnet_1, image_1, controlnet_strength_1, start_percent_1, end_percent_1)]),

        if controlnet_2_path != "None" and  switch_2 and image_2 is not None:
            # controlnet_path = folder_paths.get_full_path("controlnet", controlnet_2)
            controlnet_2 = comfy.controlnet.load_controlnet(controlnet_2_path)
            controlnet_list.extend([(controlnet_2, image_2, controlnet_strength_2, start_percent_2, end_percent_2)]),

        if controlnet_3_path != "None" and  switch_3 and image_3 is not None:
            # controlnet_path = folder_paths.get_full_path("controlnet", controlnet_3)
            controlnet_3 = comfy.controlnet.load_controlnet(controlnet_3_path)
            controlnet_list.extend([(controlnet_3, image_3, controlnet_strength_3, start_percent_3, end_percent_3)]),

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/ControlNet-Nodes#cr-multi-controlnet-stack"

        return (controlnet_list, show_help, )



class Maliooo_ControlNetStack:

    controlnets = ["None"] + folder_paths.get_filename_list("controlnet")
    
    @classmethod
    def INPUT_TYPES(cls):
        #controlnets = ["None"]
        return {"required": {
                },
                "optional": {
                    "switch_1": ("BOOLEAN", {"default": False}),
                    "controlnet_1": (cls.controlnets,),
                    "controlnet_strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "start_percent_1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_percent_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    #
                    "switch_2": ("BOOLEAN", {"default": True}),
                    "controlnet_2": (cls.controlnets,),
                    "controlnet_strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "start_percent_2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_percent_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    #
                    "switch_3": ("BOOLEAN", {"default": True}),
                    "controlnet_3": (cls.controlnets,),
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
    CATEGORY = "maliooo/controlnet"

    def controlnet_stacker(self, switch_1, controlnet_1, controlnet_strength_1, start_percent_1, end_percent_1,
                           switch_2, controlnet_2, controlnet_strength_2, start_percent_2, end_percent_2,
                           switch_3, controlnet_3, controlnet_strength_3, start_percent_3, end_percent_3,
                           image_1=None, image_2=None, image_3=None, controlnet_stack=None):

        # Initialise the list
        controlnet_list= []
        
        if controlnet_stack is not None:
            controlnet_list.extend([l for l in controlnet_stack if l[0] != "None"])
        
        if controlnet_1 != "None" and  switch_1 and image_1 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet_1)
            controlnet_1 = comfy.controlnet.load_controlnet(controlnet_path)
            controlnet_list.extend([(controlnet_1, image_1, controlnet_strength_1, start_percent_1, end_percent_1)]),

        if controlnet_2 != "None" and  switch_2 and image_2 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet_2)
            controlnet_2 = comfy.controlnet.load_controlnet(controlnet_path)
            controlnet_list.extend([(controlnet_2, image_2, controlnet_strength_2, start_percent_2, end_percent_2)]),

        if controlnet_3 != "None" and  switch_3 and image_3 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet_3)
            controlnet_3 = comfy.controlnet.load_controlnet(controlnet_path)
            controlnet_list.extend([(controlnet_3, image_3, controlnet_strength_3, start_percent_3, end_percent_3)]),

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/ControlNet-Nodes#cr-multi-controlnet-stack"

        return (controlnet_list, show_help, )


NODE_CLASS_MAPPINGS = {
    "Example": Example,
    "Maliooo_Get_Controlnet_Name": Maliooo_Get_Controlnet_Name,
    "Maliooo_ControlNetStack_By_Name": Maliooo_ControlNetStack_By_Name
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Example": "Example Node",
    "Maliooo_Get_Controlnet_Name": "Get Controlnet Name",
    "Maliooo_ControlNetStack_By_Name": "ControlNet Stack By Name"

}