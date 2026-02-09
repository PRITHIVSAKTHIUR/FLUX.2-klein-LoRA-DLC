import os
import gc
import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
from typing import Iterable

from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red", c50="#FFF0E5", c100="#FFE0CC", c200="#FFC299", c300="#FFA366",
    c400="#FF8533", c500="#FF4500", c600="#E63E00", c700="#CC3700", c800="#B33000",
    c900="#992900", c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self, *, primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate, text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue, secondary_hue=secondary_hue, neutral_hue=neutral_hue,
            text_size=text_size, font=font, font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600", block_border_width="3px",
            block_shadow="*shadow_drop_lg", button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px", color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()
MAX_SEED = np.iinfo(np.int32).max

# Face Swap Prompt Template
FACE_SWAP_PROMPT = """head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. Remove the head from Picture 1 completely and replace it with the head from Picture 2.
FROM PICTURE 1 (strictly preserve):
- Scene: lighting conditions, shadows, highlights, color temperature, environment, background
- Head positioning: exact rotation angle, tilt, direction the head is facing
- Expression: facial expression, micro-expressions, eye gaze direction, mouth position, emotion
FROM PICTURE 2 (strictly preserve identity):
- Facial structure: face shape, bone structure, jawline, chin
- All facial features: eye color, eye shape, nose structure, lip shape and fullness, eyebrows
- Hair: color, style, texture, hairline
- Skin: texture, tone, complexion
The replaced head must seamlessly match Picture 1's lighting and expression while maintaining the complete identity from Picture 2. High quality, photorealistic, sharp details, 4k."""

LORA_STYLES = [
    {
        "image": "https://huggingface.co/spaces/prithivMLmods/FLUX.2-Klein-LoRA-Studio/resolve/main/examples/image.webp",
        "title": "None",
        "adapter_name": None,
        "repo": None,
        "weights": None,
        "default_prompt": None
    },
    {
        "image": "https://huggingface.co/linoyts/Flux2-Klein-Delight-LoRA/resolve/main/image_3.png",
        "title": "Klein-Delight-Style",
        "adapter_name": "klein-delight",
        "repo": "linoyts/Flux2-Klein-Delight-LoRA",
        "weights": "pytorch_lora_weights.safetensors",
        "default_prompt": None
    },
    {
        "image": "https://huggingface.co/spaces/prithivMLmods/FLUX.2-Klein-LoRA-Studio/resolve/main/examples/face-swap.jpg",
        "title": "Best-Face-Swap",
        "adapter_name": "face-swap",
        "repo": "Alissonerdx/BFS-Best-Face-Swap",
        "weights": "bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors",
        "default_prompt": FACE_SWAP_PROMPT
    },
    {
        "image": "https://huggingface.co/spaces/prithivMLmods/FLUX.2-Klein-LoRA-Studio/resolve/main/examples/mc.png",
        "title": "Ghost-Mannequin",
        "adapter_name": "ghost-mannequin",
        "repo": "nhathoangfoto/FLUX.2-klein-ghost-mannequin",
        "weights": "3D-GhosMannequinRank-256_000005000.safetensors",
        "default_prompt": None
    },
]

LOADED_ADAPTERS = set()

print("Loading FLUX.2 Klein 9B model base...")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B",
    torch_dtype=torch.bfloat16,
).to(device)
print("Base Model loaded successfully.")

def update_dimensions_on_upload(image):
    """Resizes image to be divisible by 16 to avoid tensor mismatch errors in FLUX."""
    if image is None:
        return 1024, 1024
    
    original_width, original_height = image.size
    
    scale = min(1024 / original_width, 1024 / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16
    
    return new_width, new_height

def process_gallery_images(images):
    """Process images from gallery input and return list of PIL images."""
    if not images:
        return []
    
    pil_images = []
    for item in images:
        try:
            if isinstance(item, tuple) or isinstance(item, list):
                path_or_img = item[0]
            else:
                path_or_img = item

            if isinstance(path_or_img, str):
                pil_images.append(Image.open(path_or_img).convert("RGB"))
            elif isinstance(path_or_img, Image.Image):
                pil_images.append(path_or_img.convert("RGB"))
            else:
                pil_images.append(Image.open(path_or_img.name).convert("RGB"))
        except Exception as e:
            print(f"Skipping invalid image item: {e}")
            continue
    
    return pil_images

def get_style_by_name(name):
    """Retrieve the style dictionary by its title."""
    for style in LORA_STYLES:
        if style["title"] == name:
            return style
    return LORA_STYLES[0]  # Default to None

def update_style_selection(evt: gr.SelectData):
    """Update selected style based on gallery click."""
    selected_style = LORA_STYLES[evt.index]
    default_prompt = selected_style.get("default_prompt", None)
    # Return the title string and optional prompt update
    return selected_style["title"], default_prompt if default_prompt else gr.update()

def update_style_info(style_name):
    """Update the info text based on the selected style name."""
    return f"### Selected: {style_name} ✅"

def get_image_count_info(images):
    """Return info about uploaded images"""
    if not images:
        return "📷 No images uploaded"
    
    count = len(images)
    if count == 1:
        return "📷 1 image uploaded (Picture 1 - Base)"
    elif count == 2:
        return "📷 2 images uploaded (Picture 1 - Base, Picture 2 - Face Source)"
    else:
        return f"📷 {count} images uploaded"

@spaces.GPU
def infer(
    input_images, 
    prompt, 
    style_name,
    seed=42, 
    randomize_seed=True, 
    guidance_scale=1.0, 
    steps=4, 
    progress=gr.Progress(track_tqdm=True)
):
    gc.collect()
    torch.cuda.empty_cache()

    if not input_images:
        raise gr.Error("Please upload at least one image to apply a style to.")

    # Process gallery images
    pil_images = process_gallery_images(input_images)
    
    if not pil_images:
        raise gr.Error("Could not process uploaded images.")

    # Find the selected style configuration
    selected_style = get_style_by_name(style_name)
    
    # Check if Face Swap is selected and validate image count
    if selected_style["adapter_name"] == "face-swap":
        if len(pil_images) < 2:
            raise gr.Error("Face Swap requires exactly 2 images: Picture 1 (base/body) and Picture 2 (face source). Please upload 2 images.")
        elif len(pil_images) > 2:
            gr.Warning("Face Swap uses only the first 2 images. Additional images will be ignored.")
            pil_images = pil_images[:2]
    
    if selected_style["adapter_name"] is None:
        print("Selection is None. Disabling LoRA adapters.")
        pipe.disable_lora()
    else:
        adapter_name = selected_style["adapter_name"]
        
        if adapter_name not in LOADED_ADAPTERS:
            print(f"--- Downloading and Loading Adapter: {selected_style['title']} ---")
            try:
                pipe.load_lora_weights(
                    selected_style["repo"], 
                    weight_name=selected_style["weights"], 
                    adapter_name=adapter_name
                )
                LOADED_ADAPTERS.add(adapter_name)
            except Exception as e:
                raise gr.Error(f"Failed to load adapter {selected_style['title']}: {e}")
        else:
            print(f"--- Adapter {selected_style['title']} is already loaded. ---")
            
        print(f"Activating LoRA: {adapter_name}")
        pipe.set_adapters([adapter_name], adapter_weights=[1.0])

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    # Get dimensions from first image
    width, height = update_dimensions_on_upload(pil_images[0])
    
    # Process all images to the same dimensions
    processed_images = [
        img.resize((width, height), Image.LANCZOS).convert("RGB") 
        for img in pil_images
    ]
    
    try:
        # Pass single image or list based on count
        image_input = processed_images if len(processed_images) > 1 else processed_images[0]
        
        image = pipe(
            image=image_input, 
            prompt=prompt,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).images[0]
        
        return image, seed

    except Exception as e:
        raise gr.Error(f"Inference failed: {e}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()

@spaces.GPU
def infer_example(input_images, prompt, style_name):
    if not input_images: 
        return None, 0
    
    # Handle examples where inputs might be paths
    if isinstance(input_images, str):
        input_images = [input_images]
    
    image, seed = infer(
        input_images=input_images, 
        prompt=prompt, 
        style_name=style_name, 
        seed=0, 
        randomize_seed=True,
        guidance_scale=1.0,
        steps=4
    )
    return image, seed

css = """
#col-container { margin: 0 auto; max-width: 960px; }
#main-title h1 { font-size: 2.4em !important; }
#style_gallery .grid-wrap { height: 10vh }
#input_gallery .grid-wrap { min-height: 200px }
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **FLUX.2-Klein-LoRA-Studio**", elem_id="main-title")
        gr.Markdown("Perform diverse image edits using specialized [LoRAs](https://huggingface.co/models?other=base_model:adapter:black-forest-labs/FLUX.2-klein-9B) adapters for the [FLUX.2-Klein-Distilled](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) model.")
        
        selected_style_name = gr.Textbox(value="None", visible=False, label="Selected Style Name")
        
        with gr.Row(equal_height=True):
            with gr.Column():
                input_images = gr.Gallery(
                    label="Upload Images", 
                    type="filepath", 
                    columns=2, 
                    rows=1, 
                    height=290,
                    allow_preview=True,
                    elem_id="input_gallery"
                )
                
                with gr.Row():
                    prompt = gr.Text(
                        label="Edit Prompt", 
                        max_lines=1,
                        show_label=True, 
                        placeholder="e.g., a man with a red superhero mask"
                    )
                    
                run_button = gr.Button("Apply Style", variant="primary")

                with gr.Accordion("Advanced Settings", open=False, visible=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=0.0, maximum=10.0, step=0.1, value=1.0)        
                    steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=4, step=1)
                    
            with gr.Column():
                output_image = gr.Image(label="Output Image", interactive=False, format="png", height=358)
                used_seed = gr.Textbox(label="Used Seed", interactive=False, visible=False)

        selected_style_info = gr.Markdown("### Selected: None (FLUX.2-klein-9B) ✅")
        
        style_gallery = gr.Gallery(
            [(item["image"], item["title"]) for item in LORA_STYLES],
            label="Edit Style Gallery",
            allow_preview=False,
            columns=3,
            elem_id="style_gallery",
        )
                            
        gr.Examples(
            examples=[
                [
                    ["examples/2.jpg"], 
                    "Relight the image to remove all existing lighting conditions and replace them with neutral, uniform illumination. Apply soft, evenly distributed lighting with no directional shadows, no harsh highlights, and no dramatic contrast. Maintain the original identity of all subjects exactly—preserve facial structure, skin tone, proportions, expressions, hair, clothing, and textures. Do not alter pose, camera angle, background geometry, or image composition. Lighting should appear balanced, and studio-neutral, similar to diffuse overcast or a soft lightbox setup. Ensure consistent exposure across the entire image with realistic depth and subtle shading only where necessary for form.", 
                    "Klein-Delight-Style"
                ],
                [
                    ["examples/1.jpg", "examples/2.jpg"], 
                    FACE_SWAP_PROMPT, 
                    "Best-Face-Swap"
                ],
                [
                    ["examples/1.jpg"], 
                    "cinematic polaroid with soft grain subtle vignette gentle lighting white frame handwritten photographed by prithivMLmods preserving realistic texture and details", 
                    "None"
                ],
                [
                    ["examples/cloth.jpg"], 
                    "3Dghostmannequin", 
                    "Ghost-Mannequin"
                ],
            ],
            inputs=[input_images, prompt, selected_style_name],
            outputs=[output_image, used_seed],
            fn=infer_example,
            cache_examples=False,
            label="Examples"
        )
        
        gr.Markdown("[*](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)This is still an experimental Space for FLUX.2-Klein-9B. More adapters will be added soon.")
    
    input_images.change(
        fn=get_image_count_info,
        inputs=[input_images],
    )
    
    style_gallery.select(
        fn=update_style_selection,
        outputs=[selected_style_name, prompt]
    )

    selected_style_name.change(
        fn=update_style_info,
        inputs=[selected_style_name],
        outputs=[selected_style_info]
    )
    
    run_button.click(
        fn=infer,
        inputs=[input_images, prompt, selected_style_name, seed, randomize_seed, guidance_scale, steps],
        outputs=[output_image, used_seed]
    )

if __name__ == "__main__":
    demo.queue().launch(css=css, theme=orange_red_theme, mcp_server=True, ssr_mode=False, show_error=True)
