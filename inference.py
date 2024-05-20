
from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, BeitImageProcessor
import requests
import matplotlib.pyplot as plt
from PIL import Image


model_path = "./final_model_beit_gpt"

# Perform inference on an image
model_2 = VisionEncoderDecoderModel.from_pretrained(model_path)
tokenizer_2 = GPT2TokenizerFast.from_pretrained(model_path)
image_processor_2 = BeitImageProcessor.from_pretrained(model_path)

# Let's perform inference on an image
url = 'https://img-cdn.pixlr.com/image-generator/history/65bb506dcb310754719cf81f/ede935de-1138-4f66-8ed7-44bd16efc709/medium.webp'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

pixel_values = image_processor_2(image, return_tensors="pt").pixel_values

# Autoregressively generate caption (uses greedy decoding by default)
generated_ids = model_2.generate(pixel_values, max_length=16)
generated_text = tokenizer_2.batch_decode(generated_ids, skip_special_tokens=True)[0]
plt.imshow(image)
print("generated_text:", generated_text)