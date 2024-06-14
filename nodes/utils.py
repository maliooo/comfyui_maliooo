import torch
from PIL import Image
import numpy as np
from typing import List, Union

def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(
        np.array(image).astype(np.float32) / 255.0
    ).unsqueeze(0)