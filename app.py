import os

os.environ["HTTP_PROXY"] = "http://100.122.41.16:17777"
os.environ["HTTPS_PROXY"] = "http://100.122.41.16:17777"
os.environ["NO_PROXY"] = (
    "localhost,127.0.0.1,100.64.0.0/10,192.168.0.0/16,10.0.0.0/8,127.0.0.0/8,snake-carp.ts.net"
)
os.environ["http_proxy"] = "http://100.122.41.16:17777"
os.environ["https_proxy"] = "http://100.122.41.16:17777"
os.environ["no_proxy"] = (
    "localhost,127.0.0.1,100.64.0.0/10,192.168.0.0/16,10.0.0.0/8,127.0.0.0/8,snake-carp.ts.net"
)


import base64
import io
import json
from typing import Dict, List
import asyncio


import aiohttp.web
import torch
from PIL import Image
from kipp.utils import ThreadPoolExecutor

# from models import sdxlturbo
from models.gemma import predict as gemma_completions

DEVICE: str
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


executor = ThreadPoolExecutor(max_workers=10)

# async def handler(request: aiohttp.web.Request) -> aiohttp.web.Response:
#     """draw image with prompt or image by sdxl-turbo

#     HTTP POST:
#     ::
#         {
#             "n": 1,
#             "model": "sdxl-turbo",
#             "text": "panda"
#             "image": "data:image/png;base64,xxxxx"
#         }

#     Returns:
#         image bytes in png format
#     """
#     data = await request.json()

#     prompt = data.get("text")
#     n_images = data.get("n", 1)

#     images: List[Image.Image] = []
#     videos: List[bytes] = []
#     model = data.get("model", "sdxl-turbo")
#     if model == "sdxl-turbo":
#         if data.get("image"):
#             images = sdxlturbo.img2img(
#                 prompt=prompt,
#                 negative_prompt=data.get("negative_prompt"),
#                 b64img=data["image"],
#                 n_images=n_images,
#             )
#         else:
#             images = sdxlturbo.txt2img(
#                 prompt=prompt,
#                 negative_prompt=data.get("negative_prompt"),
#                 n_images=n_images,
#             )
#     elif model == "svd-xt":
#         video = sdxlturbo.img2video(
#             b64img=data["image"],
#         )
#         videos = [video]
#     else:
#         raise ValueError(f"there is no model named {model}")

#     response: Dict[str, List] = {}

#     # convert to png
#     for img in images:
#         response.setdefault("images", [])
#         byte_arr = io.BytesIO()
#         img.save(byte_arr, format="PNG")
#         response["images"].append(
#             f"data:image/png;base64,{base64.b64encode(byte_arr.getvalue()).decode()}"
#         )

#     for video in videos:
#         response.setdefault("videos", [])
#         response["videos"].append(
#             f"data:video/mp4;base64,{base64.b64encode(video).decode()}",
#         )

#     resp = aiohttp.web.Response(
#         body=json.dumps(response), content_type="application/json"
#     )
#     return resp


async def text_predict(request: aiohttp.web.Request) -> aiohttp.web.Response:
    """predict text completions by gemma
    """
    data = await request.json()

    async with asyncio.Lock():
        ioloop = asyncio.get_event_loop()
        completion = await ioloop.run_in_executor(executor, gemma_completions, data)

    response = {
        "completion": completion,
    }
    resp = aiohttp.web.Response(
        body=json.dumps(response), content_type="application/json"
    )
    return resp


if __name__ == "__main__":
    app = aiohttp.web.Application(client_max_size=1024**2 * 30)

    # app.add_routes([aiohttp.web.post("/predict", handler)])
    app.add_routes([aiohttp.web.post("/chat/completions", text_predict)])

    aiohttp.web.run_app(app, host="0.0.0.0", port=7861)
