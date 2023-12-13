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


import base64
import io
import json
from typing import Dict, List

import aiohttp.web
import torch
from PIL import Image

from models import sdxlturbo

DEVICE: str
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


async def handler(request: aiohttp.web.Request) -> aiohttp.web.Response:
    """draw image with prompt or image by sdxl-turbo

    HTTP POST:
    ::
        {
            "n": 1,
            "model": "sdxl-turbo",
            "text": "panda"
            "image": "data:image/png;base64,xxxxx"
        }

    Returns:
        image bytes in png format
    """
    data = await request.json()
    assert data["text"], "prompt is required"
    assert isinstance(data["text"], str), "prompt must be string"

    prompt = data["text"]
    n_images = data.get("n", 1)

    images: List[Image.Image] = []
    model = data.get("model", "sdxl-turbo")
    if model == "sdxl-turbo":
        if data.get("image"):
            images = sdxlturbo.img2img(
                prompt=prompt,
                negative_prompt=data.get("negative_prompt"),
                b64img=data["image"],
                n_images=n_images,
            )
        else:
            images = sdxlturbo.txt2img(
                prompt=prompt,
                negative_prompt=data.get("negative_prompt"),
                n_images=n_images,
            )

    response: Dict[str, List] = {
        "images": [],
    }

    # convert to png
    for img in images:
        byte_arr = io.BytesIO()
        img.save(byte_arr, format="PNG")
        response["images"].append(
            f"data:image/png;base64,{base64.b64encode(byte_arr.getvalue()).decode()}"
        )

    resp = aiohttp.web.Response(
        body=json.dumps(response), content_type="application/json"
    )
    return resp


if __name__ == "__main__":
    app = aiohttp.web.Application(client_max_size=1024**2 * 30)
    app.add_routes([aiohttp.web.post("/predict", handler)])
    aiohttp.web.run_app(app, host="0.0.0.0", port=7861)
