import torch
import kornia


# 参考来自：/ComfyUI/custom_nodes/ComfyUI_essentials/image.py 的 ImageSmartSharpen
class Malio_ImageSmartSharpen:
    """自动图片锐化"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_radius": ("INT", { "default": 7, "min": 1, "max": 25, "step": 1, }),
                "preserve_edges": ("FLOAT", { "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05 }),
                "sharpen": ("FLOAT", { "default": 5.0, "min": 0.0, "max": 25.0, "step": 0.5 }),
                "ratio": ("FLOAT", { "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1 }),
        }}

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "malio/photoshop/自动图片锐化"
    FUNCTION = "execute"

    def execute(self, image, noise_radius, preserve_edges, sharpen, ratio):
        import cv2

        output = []
        #diagonal = np.sqrt(image.shape[1]**2 + image.shape[2]**2)
        if preserve_edges > 0:
            preserve_edges = max(1 - preserve_edges, 0.05)

        for img in image:
            if noise_radius > 1:
                sigma = 0.3 * ((noise_radius - 1) * 0.5 - 1) + 0.8 # this is what pytorch uses for blur
                #sigma_color = preserve_edges * (diagonal / 2048)
                blurred = cv2.bilateralFilter(img.cpu().numpy(), noise_radius, preserve_edges, sigma)
                blurred = torch.from_numpy(blurred)
            else:
                blurred = img

            if sharpen > 0:
                sharpened = kornia.enhance.sharpness(img.permute(2,0,1), sharpen).permute(1,2,0)
            else:
                sharpened = img

            img = ratio * sharpened + (1 - ratio) * blurred
            img = torch.clamp(img, 0, 1)
            output.append(img)
        
        del blurred, sharpened
        output = torch.stack(output)

        return (output,)