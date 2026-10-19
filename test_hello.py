"""Quick smoke test: load Qwen3-VL-8B in 4-bit and describe an image."""

import torch
from PIL import Image, ImageDraw

# Create a simple test image (red rectangle on white background)
img = Image.new("RGB", (512, 512), "white")
draw = ImageDraw.Draw(img)
draw.rectangle([100, 100, 400, 400], fill="red")
img.save("test_image.png")
print("✓ Test image created")

# Check GPU
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load model (this will download ~5GB on first run)
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)
print("✓ Model loaded")

# Run inference
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe what you see in this image."},
    ]},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.3)

generated = output_ids[0, inputs["input_ids"].shape[1]:]
result = tokenizer.decode(generated, skip_special_tokens=True)

print(f"\n── Model output ──\n{result}")