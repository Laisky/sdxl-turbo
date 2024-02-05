import base64
import io
import tempfile
import random
from typing import List

import torch
from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
)
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image

from .base import logger

logger = logger.getChild("sdxl_turbo")


DEVICE: str
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

pipes = {
    "txt2img": AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    ).to(DEVICE),
    "img2img": AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    ).to(DEVICE),
    "img2video": StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    ),
}


def txt2img(prompt: str, negative_prompt: str, n_images: int = 1) -> List[Image.Image]:
    """draw image with text by sdxl-turbo

    Args:
        prompt (str): prompt text
        negative_prompt (str): negative prompt text
        n_images (int, optional): number of images to generate. Defaults to 1.

    Returns:
        List[Image.Image]: list of PIL image
    """
    logger.debug(f"txt2img with {prompt=}, {negative_prompt=}, {n_images=}")
    generator = torch.Generator(DEVICE).manual_seed(random.randint(0, 1000000))
    return pipes["txt2img"](
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        height=512,
        width=512,
        guidance_scale=0.5,
        strength=0.1,
        num_inference_steps=6,
        num_images_per_prompt=n_images,
    ).images


def img2img(
    b64img: str, prompt: str, negative_prompt: str, n_images: int = 1
) -> List[Image.Image]:
    """draw image with text by sdxl-turbo

    Args:
        b64img (str): image data encoded in base64
        prompt (str): prompt text
        negative_prompt (str): negative prompt text
        n_images (int, optional): number of images to generate. Defaults to 1.

    Returns:
        List[Image.Image]: list of PIL image
    """
    logger.debug(f"img2img with {prompt=}, {negative_prompt=}, {n_images=}")
    generator = torch.Generator(DEVICE).manual_seed(random.randint(0, 1000000))
    b64img = b64img.removeprefix("data:image/png;base64,")
    image_bytes = base64.b64decode(b64img)
    src_image = load_image(Image.open(io.BytesIO(image_bytes)))
    return pipes["img2img"](
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=src_image,
        generator=generator,
        height=1024,
        width=1024,
        guidance_scale=0.5,
        strength=0.8,
        num_inference_steps=6,
        num_images_per_prompt=n_images,
    ).images


def img2video(b64img: str, prompt: str, negative_prompt: str) -> bytes:
    """draw image with text by sdxl-turbo

    Args:
        b64img (str): image data encoded in base64
        prompt (str): prompt text
        negative_prompt (str): negative prompt text

    Returns:
        bytes: video data encoded in bytes
    """
    logger.debug(f"img2video with {prompt=}, {negative_prompt=}")
    b64img = b64img.removeprefix("data:image/png;base64,")
    image_bytes = base64.b64decode(b64img)
    src_image = load_image(Image.open(io.BytesIO(image_bytes)))

    generator = torch.manual_seed(42)
    frames = pipes["img2video"](
        src_image, decode_chunk_size=8, generator=generator
    ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        export_to_video(frames, f.name, fps=7)
        return f.read()
