"""Test Qwen3-VL on a real fashion product image."""

import torch
from PIL import Image
from unsloth import FastVisionModel

# Load model (cached from first run, no re-download)
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

# Load your image from E:
img = Image.open("/mnt/e/vlm_images/nillab-it-042148-75067d72.webp").convert("RGB")

SYSTEM_PROMPT = (
    "You are a fashion product description specialist. Given an image of a "
    "fashion item, provide a detailed, accurate description covering: "
    "silhouette and shape, color, material/texture, construction details "
    "(closures, seams, embellishments), heel/sole type if footwear, and any "
    "distinctive design features. Be precise and use standard fashion "
    "terminology. Do not speculate about brand or price."
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this fashion product in detail."},
    ]},
]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=True,
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs, max_new_tokens=512, temperature=0.3, top_p=0.9, do_sample=True,
    )

generated = output_ids[0, inputs["input_ids"].shape[1]:]
result = tokenizer.decode(generated, skip_special_tokens=True)

print("── Qwen3-VL output (zero-shot) ──")
print(result)
print("\n── Reference description ──")
print(
    "Wedge-heel round-toe sling-back in bright red. Sling-back strap features "
    "adjustable buckle closure. Toe vamp decorated with crystal embellishments "
    "in a curved pattern. Wedge heel appears to have textured woven finish. "
    "Beige footbed visible through open heel. Light-colored sole provides "
    "contrast to upper and heel."
)