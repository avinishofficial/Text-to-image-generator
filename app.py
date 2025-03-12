import torch
from diffusers import DiffusionPipeline
import gradio as gr

def generate_image(prompt, model_name="stabilityai/stable-diffusion-2-1"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Stable Diffusion model
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to(device)
    
    # Generate image
    image = pipe(prompt).images[0]
    
    return image

# Now launch Gradio again
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Image(label="Generated Image"),
    title="AI Text-to-Image Generator",
    description="Generate high-quality images from text using Stable Diffusion."
)

interface.launch(share=True)
