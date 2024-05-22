import os
import random
import uuid

import gradio as gr
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

DESCRIPTION = "🦊 Generative-Gallery Web App :"

# Load the model and scheduler based on device availability
if torch.cuda.is_available():
    device = "cuda"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "fluently/Fluently-XL-v4",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Load LoRA weights and set adapters
    pipe.load_lora_weights("artificialguybr/3DRedmond-V1", weight_name="3DRedmond-3DRenderStyle-3DRenderAF.safetensors", adapter_name="lora")
    pipe.set_adapters("lora")

    pipe.to(device)
else:
    device = "cpu"
    DESCRIPTION += "\n<p>🛑 Running on CPU. This demo may take time to generate image.</p>"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "fluently/Fluently-XL-v4",
        torch_dtype=torch.float32,  # Use float32 for CPU
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Load LoRA weights and set adapters
    pipe.load_lora_weights("artificialguybr/3DRedmond-V1", weight_name="3DRedmond-3DRenderStyle-3DRenderAF.safetensors", adapter_name="lora")
    pipe.set_adapters("lora")

    pipe.to(device)

# Define a function to save images with unique names
def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

# Define a function to handle seed randomization
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

# Define the maximum seed value
MAX_SEED = np.iinfo(np.int32).max

# Define the generation function with GPU support
def generate(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    progress=gr.Progress(track_tqdm=True),
):

    seed = int(randomize_seed_fn(seed, randomize_seed))

    if not use_negative_prompt:
        negative_prompt = ""

    # Generate images using the Stable Diffusion pipeline
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=25,
        num_images_per_prompt=1,
        cross_attention_kwargs={"scale": 0.65},
        output_type="pil",
    ).images

    # Save the generated images and return their paths
    image_paths = [save_image(img) for img in images]
    print(image_paths)
    return image_paths, seed

examples = [
    "A whimsical treehouse nestled among vibrant autumn leaves",
    "A futuristic city skyline with flying cars and neon lights",
    "A curious puppy exploring a field of wildflowers",
    "A cozy cabin in a snowy forest, with smoke billowing from the chimney",
    "A surreal landscape with melting clocks and distorted reality",
    "A majestic dragon soaring above a medieval castle",
    "A cute robot walking a dog on a city street",
    "A serene beach scene with palm trees and a colorful sunset",
    "A whimsical bakery with pastries that come to life",
    "A mystical forest with glowing mushrooms and ethereal creatures"
]

css = '''
.gradio-container{max-width: 560px !important}
h1{text-align:center}
footer {
    visibility: hidden
}
'''

# Create the Gradio interface
with gr.Blocks(css=css, theme="HaleyCH/HaleyCH_Theme") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(label="Result", columns=1, preview=True, show_label=False)

    # Advanced options section with an accordion
    with gr.Accordion("Advanced options", open=False):
        use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=True)
        negative_prompt = gr.Text(
            label="Negative prompt",
            lines=4,
            max_lines=6,
            value=""bad proportions, low resolution, bad, ugly, terrible, watermark, signature, worst quality, low quality, inaccurate limb, extra fingers, fewer fingers, missing fingers, inaccurate eyes, bad composition, bad anatomy, cropped, blurry, deformed, malformed, deformed face, duplicate body parts, disfigured, extra limbs, fused fingers, twisted, distorted, missing limbs, mutation""",
            placeholder="Enter a negative prompt",
            visible=True,
        )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
            visible=True
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row(visible=True):
            width = gr.Slider(
                label="Width",
                minimum=512,
                maximum=2048,
                step=8,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=512,
                maximum=2048,
                step=8,
                value=1024,
            )
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=0.1,
                maximum=20.0,
                step=0.1,
                value=7.5,
            )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate,
        cache_examples=False,
    )

    # Functionality to toggle the visibility of the negative prompt field
    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    # Trigger the generation function on prompt submission, negative prompt submission, and Run button click
    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            use_negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
        ],
        outputs=[result, seed],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(show_api=False, debug=False)
