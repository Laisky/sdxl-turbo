import os

os.environ["HTTP_PROXY"] = "http://100.122.41.16:17777"
os.environ["HTTPS_PROXY"] = "http://100.122.41.16:17777"
os.environ[
    "NO_PROXY"
] = "localhost,127.0.0.1,100.64.0.0/10,192.168.0.0/16,10.0.0.0/8,127.0.0.0/8,snake-carp.ts.net"
os.environ["http_proxy"] = "http://100.122.41.16:17777"
os.environ["https_proxy"] = "http://100.122.41.16:17777"
os.environ[
    "no_proxy"
] = "localhost,127.0.0.1,100.64.0.0/10,192.168.0.0/16,10.0.0.0/8,127.0.0.0/8,snake-carp.ts.net"


from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
)
from diffusers.utils.loading_utils import load_image
from PIL import Image
import base64
import io
import torch
import aiohttp.web
import random

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
        requires_safety_checker=False,
    ).to(DEVICE),
    "img2img": AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        requires_safety_checker=False,
    ).to(DEVICE),
}


async def handler(request: aiohttp.web.Request) -> aiohttp.web.Response:
    """draw image with text or image by sdxl-turbo

    HTTP POST:
    ::
        {
            "text": "panda"
            "image": "data:image/png;base64,xxxxx"
        }

    Returns:
        image bytes in png format
    """
    data = await request.json()
    assert data["text"], "text is required"
    prompt = data["text"]

    generator = torch.Generator(DEVICE).manual_seed(random.randint(0, 1000000))

    resp: aiohttp.web.Response
    if data.get("image"):
        base64_image: str = data["image"]
        base64_image = base64_image.removeprefix("data:image/png;base64,")
        image_bytes = base64.b64decode(base64_image)
        src_image = load_image(Image.open(io.BytesIO(image_bytes)))

        # draw image
        result_pil_image = pipes["img2img"](
            prompt=prompt,
            image=src_image,
            generator=generator,
            height=512,
            width=512,
            guidance_scale=0.5,
            num_inference_steps=4,
        ).images[0]

        # convert to png
        byte_arr = io.BytesIO()
        result_pil_image.save(byte_arr, format="PNG")
        resp = aiohttp.web.Response(body=byte_arr.getvalue(), content_type="image/png")
    else:
        result_pil_image = pipes["txt2img"](
            prompt=prompt,
            generator=generator,
            height=512,
            width=512,
            guidance_scale=0.5,
            num_inference_steps=4,
        ).images[0]
        # convert to png
        byte_arr = io.BytesIO()
        result_pil_image.save(byte_arr, format="PNG")
        resp = aiohttp.web.Response(body=byte_arr.getvalue(), content_type="image/png")

    return resp


if __name__ == "__main__":
    app = aiohttp.web.Application()
    app.add_routes([aiohttp.web.post("/predict", handler)])
    aiohttp.web.run_app(app, host="100.102.187.66", port=7861)
