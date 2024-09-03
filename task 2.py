# Step 1: Install the required libraries (if not already installed)
# !pip install diffusers transformers torch torchvision

# Step 2: Import necessary libraries
from diffusers import StableDiffusionPipeline
import torch

# Step 3: Load the pre-trained Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)  # Remove torch_dtype argument

# Use GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Step 4: Generate an image from a text prompt
prompt = "A futuristic cityscape with flying cars"  # Customize your prompt
image = pipe(prompt).images[0]

# Step 5: Save the generated image to a file
image.save("generated_image.png")

# Optional: Display the generated image
image.show()

print("Image generated and saved as 'generated_image.png'")