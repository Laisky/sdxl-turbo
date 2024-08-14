import base64
import io
import tempfile
import random
from typing import List
import gc
import os

import torch
from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
)
from diffusers import StableVideoDiffusionPipeline, DiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
from kipp.decorator import timer


from .base import logger

logger = logger.getChild("sdxl_turbo")


# https://huggingface.co/settings/tokens
# from huggingface_hub import login
# login()

DEVICE: str
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

pipes = {
    "img2video": DiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
    ).to(DEVICE),
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
}


@timer
def txt2img(prompt: str, negative_prompt: str, n_images: int = 1) -> List[Image.Image]:
    """draw image with text by sdxl-turbo

    Args:
        prompt (str): prompt text
        negative_prompt (str): negative prompt text
        n_images (int, optional): number of images to generate. Defaults to 1.

    Returns:
        List[Image.Image]: list of PIL image
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("prompt should be a non-empty string")

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


@timer
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
    if not prompt or not isinstance(prompt, str):
        raise ValueError("prompt should be a non-empty string")

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


@timer
def img2video(b64img: str) -> bytes:
    """draw image with text by sdxl-turbo

    Args:
        b64img (str): image data encoded in base64

    Returns:
        bytes: video data encoded in bytes
    """
    logger.debug(f"img2video")
    b64img = b64img.removeprefix("data:image/png;base64,")
    image_bytes = base64.b64decode(b64img)
    image = load_image(Image.open(io.BytesIO(image_bytes)))
    # image = image.resize((1024, 576))

    generator = torch.manual_seed(42)

    # Perform GPU memory cleanup
    gc.collect()
    torch.cuda.empty_cache()

    frames = pipes["img2video"](
        image=image,
        decode_chunk_size=3,
        generator=generator,
        # num_frames=10,
        num_inference_steps=10,
        motion_bucket_id=180,
        noise_aug_strength=0.3,
    ).frames[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "video.mp4")
        export_to_video(frames, fpath, fps=7)
        with open(fpath, "rb") as f:
            content = f.read()

        logger.info(f"generate video size: {len(content)} bytes")
        return content
