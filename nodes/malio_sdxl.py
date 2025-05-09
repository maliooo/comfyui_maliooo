
from .components.fields import Field
from .components.sizes import get_image_size
from comfy.utils import common_upscale

scale_methods = ["nearest-exact", "bilinear", "bicubic", "bislerp", "area", "lanczos"]

class Malio_SDXL_Image_Resize:
    """Image Resize 缩放到SDXL比例
    
    参考：https://www.cnblogs.com/bossma/p/17615201.html
    """

    scale_methods = scale_methods
    crop_methods = ["disabled", "center"]

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": Field.image(),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "🐼malio/image/sdxl_resize"

    def upscale(
        self,
        image,
    ):
        samples = image.movedim(-1, 1)

        size = get_image_size(image)
        size_tuple_width = (
            (1024, 1024),
            (1152, 896),
            (1216, 832),
            (1344, 768),
            (1536, 640),
        )
        size_tuple_height = (
            (1024, 1024),
            (896, 1152),
            (832, 1216),
            (768, 1344),
            (640, 1536),
        )


        width_B = int(size[0])
        height_B = int(size[1])
        
        if width_B == height_B:
            new_width, new_height = 1024, 1024
        elif width_B > height_B:
            # 选择宽高比例最接近的
            new_width, new_height = min(size_tuple_width, key=lambda x: abs(x[0] / x[1] - width_B / height_B))
        else:
            new_width, new_height = min(size_tuple_height, key=lambda x: abs(x[0] / x[1] - width_B / height_B))
        

        res = common_upscale(samples, new_width, new_height, upscale_method="lanczos", crop="disabled")
        res = res.movedim(1, -1)  # 移动维度
        return (res,)


class Malio_SD35_Image_Resize:
    """Image Resize 缩放到SD35比例
    
    参考：https://www.cnblogs.com/bossma/p/17615201.html
    """

    scale_methods = scale_methods
    crop_methods = ["disabled", "center"]

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": Field.image(),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "🐼malio/image/sdxl_resize"

    def upscale(
        self,
        image,
    ):
        samples = image.movedim(-1, 1)

        size = get_image_size(image)
        size_tuple_width = (
            (1024, 1024),
            (1152, 896),
            (1216, 832),
            (1344, 768),
            # (1536, 640),
        )
        size_tuple_height = (
            (1024, 1024),
            (896, 1152),
            (832, 1216),
            (768, 1344),
            (640, 1536),
        )


        width_B = int(size[0])
        height_B = int(size[1])
        
        if width_B == height_B:
            new_width, new_height = 1024, 1024
        elif width_B > height_B:
            # 选择宽高比例最接近的
            new_width, new_height = min(size_tuple_width, key=lambda x: abs(x[0] / x[1] - width_B / height_B))
        else:
            new_width, new_height = min(size_tuple_height, key=lambda x: abs(x[0] / x[1] - width_B / height_B))
        

        res = common_upscale(samples, new_width, new_height, upscale_method="lanczos", crop="disabled")
        res = res.movedim(1, -1)  # 移动维度
        return (res,)
    

class Malio_SDXL_Image_Resize_64:
    """Image Resize 缩放到SDXL比例
    
    参考：https://www.cnblogs.com/bossma/p/17615201.html
    """

    scale_methods = scale_methods
    crop_methods = ["disabled", "center"]

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": Field.image(),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "🐼malio/image/sdxl_resize"

    def upscale(
        self,
        image,
    ):
        samples = image.movedim(-1, 1)

        size = get_image_size(image)
        size_tuple = (
            (512, 2048),
            (512, 1984),
            (512, 1920),
            (512, 1856),
            (576, 1792),
            (576, 1728),
            (576, 1664),
            (640, 1600),
            (640, 1536),
            (704, 1472),
            (704, 1408),
            (704, 1344),
            (768, 1344),
            (768, 1280),
            (832, 1216),
            (832, 1152),
            (896, 1152),
            (896, 1088),
            (960, 1088),
            (960, 1024),
            (1024, 1024),
            (1024, 960),
            (1088, 960),
            (1088, 896),
            (1152, 896),
            (1152, 832),
            (1216, 832),
            (1280, 768),
            (1344, 768),
            (1408, 704),
            (1472, 704),
            (1536, 640),
            (1600, 640),
            (1664, 576),
            (1728, 576),
            (1792, 576),
            (1856, 512),
            (1920, 512),
            (1984, 512),
            (2048, 512),
        )


        width_B = int(size[0])
        height_B = int(size[1])
        
        if width_B == height_B:
            new_width, new_height = 1024, 1024
        else:
            new_width, new_height = min(size_tuple, key=lambda x: abs(x[0] / x[1] - width_B / height_B))
        

        res = common_upscale(samples, new_width, new_height, upscale_method="lanczos", crop="disabled")
        res = res.movedim(1, -1)  # 移动维度
        return (res,)
    

class Malio_SDXL_Image_Resize_1536:
    """Image Resize 缩放到SDXL比例
    
    参考：https://www.cnblogs.com/bossma/p/17615201.html
    """

    scale_methods = scale_methods
    crop_methods = ["disabled", "center"]

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": Field.image(),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "🐼malio/image/sdxl_resize"

    def upscale(
        self,
        image,
    ):
        samples = image.movedim(-1, 1)

        size = get_image_size(image)
        size_tuple = (
            (768, 3072),
            (768, 2976),
            (768, 2880),
            (768, 2784),
            (864, 2688),
            (864, 2592),
            (864, 2496),
            (960, 2400),
            (960, 2304),
            (1056, 2208),
            (1056, 2112),
            (1056, 2016),
            (1152, 2016),
            (1152, 1920),
            (1248, 1824),
            (1248, 1728),
            (1344, 1728),
            (1344, 1632),
            (1440, 1632),
            (1440, 1536),
            (1536, 1536),
            (1536, 1440),
            (1632, 1440),
            (1632, 1344),
            (1728, 1344),
            (1728, 1248),
            (1824, 1248),
            (1920, 1152),
            (2016, 1152),
            (2112, 1056),
            (2208, 1056),
            (2304, 960),
            (2400, 960),
            (2496, 864),
            (2592, 864),
            (2688, 864),
            (2784, 768),
            (2880, 768),
            (2976, 768),
            (3072, 768),
        )


        width_B = int(size[0])
        height_B = int(size[1])
        
        if width_B == height_B:
            new_width, new_height = 1536, 1536
        else:
            new_width, new_height = min(size_tuple, key=lambda x: abs(x[0] / x[1] - width_B / height_B))
        

        res = common_upscale(samples, new_width, new_height, upscale_method="lanczos", crop="disabled")
        res = res.movedim(1, -1)  # 移动维度
        return (res,)