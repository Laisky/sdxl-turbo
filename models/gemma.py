# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")


def predict(propmt: str) -> str:
    input_ids = tokenizer(propmt, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids)
    return tokenizer.decode(outputs[0])
