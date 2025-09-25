import torch
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from controlnet_aux import PidiNetDetector, HEDdetector

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler, UniPCMultistepScheduler
)

from PIL import Image, ImageOps, ImageFilter
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import numpy as np
 
def _sauvola_mask(g: np.ndarray, window: int = 31, k: float = 0.3, R: float = 128.0):
    """
    g: 2D grayscale float32 in [0,255]
    window: odd window size ~ 1–2× stroke width
    k: sensitivity (0.2–0.5). Higher keeps fainter strokes.
    R: dynamic range (128 for 8-bit)
    """
    win = int(window) | 1                       # ensure odd
    pad = win // 2
    gp = np.pad(g, ((pad, pad), (pad, pad)), mode="reflect")
 
    # integral images (summed area tables)
    I  = np.pad(gp, ((1,0),(1,0)), mode="constant").cumsum(0).cumsum(1)
    I2 = np.pad(gp*gp, ((1,0),(1,0)), mode="constant").cumsum(0).cumsum(1)
 
    S  = I[win:, win:] - I[:-win, win:] - I[win:, :-win] + I[:-win, :-win]
    S2 = I2[win:, win:] - I2[:-win, win:] - I2[win:, :-win] + I2[:-win, :-win]
 
    area = float(win * win)
    m = S / area
    var = S2 / area - m * m
    std = np.sqrt(np.clip(var, 0, None))
 
    T = m * (1.0 + k * (std / R - 1.0))        # Sauvola threshold
    return g < T                                # True = ink
 
def scribble_to_black_on_white(img: Image.Image,
                               invert: bool = False,
                               window: int | None = None,
                               k: float = 0.3) -> Image.Image:
    # 1) grayscale + light cleanup
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
 
    # choose a reasonable default window from image size if not provided
    if window is None:
        h, w = gray.size[1], gray.size[0]
        window = max(15, (min(h, w) // 20) | 1)   # e.g., 31–61 on ~1–2k px images
 
    # 2) Local adaptive (Sauvola) threshold
    g = np.asarray(gray, dtype=np.float32)
    ink = _sauvola_mask(g, window=window, k=k)
 
    # 3) Output black ink on white
    out = np.where(ink, 0, 255).astype(np.uint8)
    if invert:
        out = 255 - out
    return Image.fromarray(out, mode="L")

'''
def scribble_to_black_on_white(img: Image.Image, invert: bool = False) -> Image.Image:
    # 1) grayscale + light cleanup
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
 
    # 2) Otsu threshold
    g = np.asarray(gray)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    total = g.size
    sum_total = np.dot(np.arange(256), hist)
 
    sumB = wB = 0.0
    varMax, thresh = -1.0, 127
    for t in range(256):
        wB += hist[t]
        if wB == 0: 
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > varMax:
            varMax, thresh = var_between, t
 
    ink = g < thresh                      # darker = ink
    out = np.where(ink, 0, 255).astype(np.uint8)  # black on white
    if invert:
        out = 255 - out                   # white on black
 
    return Image.fromarray(out, mode="L")


image = load_image(
    "C:\\Projects\\CSR_Education\\Webcam_drawing\\canny_results\\canny_lower30_upper40.png"
)

#hed = HEDdetector.from_pretrained('lllyasviel/Annotators')  #Does not exists
#image = hed(image, scribble=True)
image=scribble_to_black_on_white(image,True)
image.save("HED_processed_image.png")

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None).to('cuda')
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

image = pipe(
        prompt="House in the mountains with a lake infront ", 
        #negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy",
        torch_dtype=torch.float16, image=image, num_inference_steps=20, 
        #guidance_scale=6.0,
        #controlnet_conditioning_scale=2.0
        ).images[0]
image.save("guess_mode_generated.png")
'''