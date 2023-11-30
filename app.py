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
    if data["pipe"] == "txt2img":
        text = data["text"]
        image = pipe(text)
        return aiohttp.web.Response(body=image, content_type="image/png")
    elif data["pipe"] == "img2img":
        image = load_image(data["image"])
        image = pipe(image)
        return aiohttp.web.Response(body=image, content_type="image/png")


if __name__ == "__main__":
    app = aiohttp.web.Application()
    app.add_routes([aiohttp.web.post("/", handler)])
    aiohttp.web.run_app(app, host="0.0.0.0", port=8000)
