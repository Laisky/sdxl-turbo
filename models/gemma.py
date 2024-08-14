from typing import Dict
from textwrap import dedent
import asyncio

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from kipp.decorator import timer

from .env import TOKEN

model_id = "gg-hf/gemma-2b-it"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id, token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype,
    token=TOKEN,
)


@timer
def predict(data: Dict) -> str:
    chat = data["messages"]
    input_ids = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(input_ids, return_tensors="pt").to("cuda")

    outputs = model.generate(max_new_tokens=data["max_tokens"], **input_ids)
    return tokenizer.decode(outputs[0])


if __name__ == "__main__":
    system_prompt = dedent(
        """
        You play the role of a tokenization engine. Execute the following instructions step by step and provide the output of each step.

        1. Read the user's input, extract the most important keywords based on your understanding, Output these words separated by commas.

        2. Create an expanded JSON array based on the original one. Keep the original array intact and translate any non-English words into English, adding those translations to the end of the original array.

        3. Present the final result as a JSON array, Please strictly output in the following format, without adding any other characters:

            result: ["word-1", "word-2"].
        """
    )
    user_prompt = "焦らし寸止め絶頂セックス あやみ史上1番エロいです！あやみはまだまだ進化しています！ ACT.03 あやみ旬果"
    body = {
        "model": "gemma-2b-it",
        "max_tokens": 3000,
        "stream": False,
        "messages": [
            {"role": "user", "content": f"{system_prompt}\n>>\n{user_prompt}"},
        ],
    }
    print(predict(body))
