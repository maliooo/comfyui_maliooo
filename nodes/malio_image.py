import math
from .components.fields import Field
from .components.sizes import get_image_size
from comfy.utils import common_upscale

scale_methods = ["nearest-exact", "bilinear", "bicubic", "bislerp", "area", "lanczos"]

class Malio_ImageScale_Side:
    scale_methods = scale_methods
    crop_methods = ["disabled", "center"]

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": Field.image(),
                "side_length": Field.int(),
                "side": Field.combo(["Longest", "Shortest", "Width", "Height"]),
                "upscale_method": Field.combo(cls.scale_methods),
                "crop": Field.combo(cls.crop_methods),
                # 整除数字
                "divisible_num": ("INT", {"default": 1, "min": 1, "max": 32}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "malio/image_scale_side"

    def upscale(self, image, upscale_method, side_length: int, side: str, crop, divisible_num: int):
        samples = image.movedim(-1, 1)

        size = get_image_size(image)

        width_B = int(size[0])
        height_B = int(size[1])

        width = width_B
        height = height_B

        def determineSide(_side: str) -> tuple[int, int]:
            width, height = 0, 0
            if _side == "Width":
                heigh_ratio = height_B / width_B
                width = side_length
                height = heigh_ratio * width
            elif _side == "Height":
                width_ratio = width_B / height_B
                height = side_length
                width = width_ratio * height
            return width, height

        if side == "Longest":
            if width > height:
                width, height = determineSide("Width")
            else:
                width, height = determineSide("Height")
        elif side == "Shortest":
            if width < height:
                width, height = determineSide("Width")
            else:
                width, height = determineSide("Height")
        else:
            width, height = determineSide(side)

        width = math.ceil(width)
        height = math.ceil(height)

        # 整除数字
        width = width // divisible_num * divisible_num
        height = height // divisible_num * divisible_num


        res = common_upscale(samples, width, height, upscale_method, crop)
        res = res.movedim(1, -1)  # 移动维度
        return (res,)
