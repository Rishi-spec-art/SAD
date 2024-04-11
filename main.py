from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import torch
import cv2
import streamlit as st
from huggingface_hub import login

tkn = st.secrets["H_TOKEN"]
login(token = tkn)

class CFG:
    device = "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400,400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12
    
token = st.secrets["H_TOKEN"]
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float32,
    revision="fp16", guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image


st.title("Stable Diffusion AI Image Generator")
st.markdown("You can generate any number of images by giving prompts...Hope you have a happy day :blush::ok_hand: !")
prompt = st.text_input("Enter prompt", max_chars=256)

img = generate_image(prompt, image_gen_model)

st.image(img)
