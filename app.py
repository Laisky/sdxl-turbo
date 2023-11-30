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


from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
import aiohttp.web

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


pipes = {
    "txt2img": AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    ).to(device),
    "img2img": AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    ).to(device),
}


async def handler(request: aiohttp.web.Request) -> aiohttp.web.Response:
    data = await request.json()
    pipe = pipes[data["type"]]
    resp: aiohttp.web.Response
    if data["pipe"] == "txt2img":
        text = data["text"]
        image = pipe(text)
        resp = aiohttp.web.Response(body=image, content_type="image/png")
    elif data["pipe"] == "img2img":
        image = load_image(data["image"])
        image = pipe(image)
        resp = aiohttp.web.Response(body=image, content_type="image/png")
    else:
        resp = aiohttp.web.HTTPBadRequest(reason="Invalid pipe")

    return resp


if __name__ == "__main__":
    app = aiohttp.web.Application()
    app.add_routes([aiohttp.web.post("/", handler)])
    aiohttp.web.run_app(app, host="0.0.0.0", port=7860)
