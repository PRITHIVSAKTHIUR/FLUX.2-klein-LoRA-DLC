import os
import gc
import gradio as gr
import numpy as np
import random
import spaces
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
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
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

REPO_ID_REGULAR = "black-forest-labs/FLUX.2-klein-base-9B"
REPO_ID_DISTILLED = "black-forest-labs/FLUX.2-klein-9B"

ADAPTER_SPECS = {
    "Outpaint": {
        "repo": "fal/flux-2-klein-4B-outpaint-lora",
        "weights": "flux-outpaint-lora.safetensors",
        "adapter_name": "outpaint"
    },  
     "Zoom": {
        "repo": "fal/flux-2-klein-4B-zoom-lora",
        "weights": "flux-red-zoom-lora.safetensors",
        "adapter_name": "zoom"
    }, 
    "Background-Remove": {
        "repo": "fal/flux-2-klein-4B-background-remove-lora",
        "weights": "flux-background-remove-lora.safetensors",
        "adapter_name": "rmbg"
    },
     "Object-Remove": {
        "repo": "fal/flux-2-klein-4B-object-remove-lora",
        "weights": "flux-object-remove-lora.safetensors",
        "adapter_name": "object-remove"
    }, 
    "Sprite-Sheet": {
        "repo": "fal/flux-2-klein-4b-spritesheet-lora",
        "weights": "flux-spritesheet-lora.safetensors",
        "adapter_name": "spritesheet"
    },
}

current_loaded_lora = "None"

print(f"Device: {device}")
print("Loading 9B Regular model...")
pipe_regular = Flux2KleinPipeline.from_pretrained(REPO_ID_REGULAR, torch_dtype=dtype)
pipe_regular.to(device)

print("Loading 9B Distilled model...")
pipe_distilled = Flux2KleinPipeline.from_pretrained(REPO_ID_DISTILLED, torch_dtype=dtype)
pipe_distilled.to(device)

pipes = {
    "Distilled (4 steps)": pipe_distilled,
    "Base (50 steps)": pipe_regular,
}

DEFAULT_STEPS = {
    "Distilled (4 steps)": 4,
    "Base (50 steps)": 50,
}

DEFAULT_CFG = {
    "Distilled (4 steps)": 1.0,
    "Base (50 steps)": 4.0,
}

def update_dimensions_from_image(image_list):
    """
    Update width/height based on uploaded image aspect ratio.
    """
    if image_list is None or len(image_list) == 0:
        return 1024, 1024
    
    img = image_list[0][0]
    img_width, img_height = img.size
    
    aspect_ratio = img_width / img_height
    
    if aspect_ratio >= 1:
        new_width = 1024
        new_height = int(1024 / aspect_ratio)
    else:
        new_height = 1024
        new_width = int(1024 * aspect_ratio)
    
    new_width = round(new_width / 8) * 8
    new_height = round(new_height / 8) * 8
    
    new_width = max(256, min(1024, new_width))
    new_height = max(256, min(1024, new_height))
    
    return new_width, new_height

def update_steps_from_mode(mode_choice):
    """
    Update inference steps and guidance scale based on the selected mode.
    """
    return DEFAULT_STEPS[mode_choice], DEFAULT_CFG[mode_choice]

def manage_lora_state(pipe, target_lora):
    """
    Manages loading/unloading of LoRAs to ensure only the requested one is active.
    """
    global current_loaded_lora
    
    if current_loaded_lora == target_lora:
        return

    print(f"Switching LoRA: {current_loaded_lora} -> {target_lora}")

    if current_loaded_lora != "None":
        try:
            print("Unloading existing LoRA weights...")
            pipe.unload_lora_weights()
        except Exception as e:
            print(f"Warning during unload: {e}")
        current_loaded_lora = "None"

    if target_lora != "None" and target_lora in ADAPTER_SPECS:
        spec = ADAPTER_SPECS[target_lora]
        print(f"Downloading/Loading {target_lora} from {spec['repo']}...")
        try:
            pipe.load_lora_weights(
                spec["repo"], 
                weight_name=spec["weights"], 
                adapter_name=spec["adapter_name"]
            )
            current_loaded_lora = target_lora
        except Exception as e:
            print(f"Error loading LoRA {target_lora}: {e}")
            raise gr.Error(f"Failed to load LoRA: {e}")

@spaces.GPU
def run_distilled(prompt, image_list, lora_style, seed, width, height, num_inference_steps, guidance_scale, progress=gr.Progress(track_tqdm=True)):
    gc.collect()
    torch.cuda.empty_cache()
    
    pipe = pipes["Distilled (4 steps)"]
    
    # Handle LoRA loading only for distilled model
    progress(0.1, desc="Checking LoRA status...")
    manage_lora_state(pipe, lora_style)
    
    progress(0.2, desc="Generating with Distilled...")
    generator = torch.Generator(device=device).manual_seed(seed)
    
    pipe_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    
    if image_list:
        pipe_kwargs["image"] = image_list
        
    try:
        image = pipe(**pipe_kwargs).images[0]
        return image, seed
    except Exception as e:
        raise gr.Error(f"Generation failed: {e}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()

@spaces.GPU(duration=120)
def run_base(prompt, image_list, seed, width, height, num_inference_steps, guidance_scale, progress=gr.Progress(track_tqdm=True)):
    gc.collect()
    torch.cuda.empty_cache()
    
    pipe = pipes["Base (50 steps)"]
    
    progress(0.2, desc="Generating with Base...")
    generator = torch.Generator(device=device).manual_seed(seed)
    
    pipe_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    
    if image_list:
        pipe_kwargs["image"] = image_list
        
    try:
        image = pipe(**pipe_kwargs).images[0]
        return image, seed
    except Exception as e:
        raise gr.Error(f"Generation failed: {e}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()

def infer(
    prompt: str,
    input_images=None,
    mode_choice: str = "Distilled (4 steps)",
    lora_style: str = "None",
    seed: int = 42,
    randomize_seed: bool = False,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 4,
    guidance_scale: float = 4.0,
    progress=gr.Progress(track_tqdm=True)
):
    if isinstance(seed, str): seed = int(seed)
    if isinstance(randomize_seed, str): randomize_seed = randomize_seed.lower() == "true"
    if isinstance(width, str): width = int(width)
    if isinstance(height, str): height = int(height)
    if isinstance(num_inference_steps, str): num_inference_steps = int(num_inference_steps)
    if isinstance(guidance_scale, str): guidance_scale = float(guidance_scale)
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    image_list = None
    if input_images is not None and len(input_images) > 0:
        image_list = []
        for item in input_images:
            image_list.append(item[0])

    if mode_choice == "Distilled (4 steps)":
        return run_distilled(prompt, image_list, lora_style, seed, width, height, num_inference_steps, guidance_scale, progress)
    else:
        return run_base(prompt, image_list, seed, width, height, num_inference_steps, guidance_scale, progress)

examples = [
    ["Create a vase on a table in living room, the color of the vase is a gradient of color, starting with #02eb3c color and finishing with #edfa3c. The flowers inside the vase have the color #ff0088"],
    ["Photorealistic infographic showing the complete Berlin TV Tower (Fernsehturm) from ground base to antenna tip, full vertical view with entire structure visible including concrete shaft, metallic sphere, and antenna spire. Slight upward perspective angle looking up toward the iconic sphere, perfectly centered on clean white background. Left side labels with thin horizontal connector lines: the text '368m' in extra large bold dark grey numerals (#2D3748) positioned at exactly the antenna tip with 'TOTAL HEIGHT' in small caps below. The text '207m' in extra large bold with 'TELECAFÉ' in small caps below, with connector line touching the sphere precisely at the window level. Right side label with horizontal connector line touching the sphere's equator: the text '32m' in extra large bold dark grey numerals with 'SPHERE DIAMETER' in small caps below. Bottom section arranged in three balanced columns: Left - Large text '986' in extra bold dark grey with 'STEPS' in caps below. Center - 'BERLIN TV TOWER' in bold caps with 'FERNSEHTURM' in lighter weight below. Right - 'INAUGURATED' in bold caps with 'OCTOBER 3, 1969' below. All typography in modern sans-serif font (such as Inter or Helvetica), color #2D3748, clean minimal technical diagram style. Horizontal connector lines are thin, precise, and clearly visible, touching the tower structure at exact corresponding measurement points. Professional architectural elevation drawing aesthetic with dynamic low angle perspective creating sense of height and grandeur, poster-ready infographic design with perfect visual hierarchy."],
    ["Soaking wet capybara taking shelter under a banana leaf in the rainy jungle, close up photo"],
    ["A kawaii die-cut sticker of a chubby orange cat, featuring big sparkly eyes and a happy smile with paws raised in greeting and a heart-shaped pink nose. The design should have smooth rounded lines with black outlines and soft gradient shading with pink cheeks."],
]

examples_images = [
    ["The person from image 1 is petting the cat from image 2, the bird from image 3 is next to them", ["examples/woman1.webp", "examples/cat_window.webp", "examples/bird.webp"]],
    ["Make it b&w.", ["examples/woman3.jpg"]],
    ["Zoom into the red highlighted area.", ["examples/example3_input.jpg"]],
    ["Remove the highlighted object from the scene.", ["examples/example4_input.jpg"]],
    ["Remove the background from the image.", ["examples/example1_input.jpg"]],
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#main-title h1 {font-size: 2.3em !important;}
.gallery-container img{
    object-fit: contain;
}
"""

with gr.Blocks(theme=orange_red_theme, css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **FLUX.2-klein-LoRA-Studio**", elem_id="main-title")
        gr.Markdown("Perform diverse image edits using specialized LoRA adapters for [FLUX.2-klein](https://huggingface.co/collections/black-forest-labs/flux2) models. Upload one or more images.")

        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    prompt = gr.Text(
                        label="Prompt",
                        show_label=False,
                        max_lines=2,
                        placeholder="Enter your prompt",
                        container=False,
                        scale=3
                    )
                    
                    run_button = gr.Button("Run", scale=1, variant="primary")
                    
                with gr.Accordion("Input image(s) (optional)", open=True):
                    input_images = gr.Gallery(
                        label="Input Image(s)",
                        type="pil",
                        columns=3,
                        rows=1,
                        height=280,
                        allow_preview=True
                    )

                with gr.Row():
                    mode_choice = gr.Radio(
                        label="Choose Mode",
                        choices=["Distilled (4 steps)", "Base (50 steps)"],
                        value="Distilled (4 steps)",
                        scale=1
                    )
                
                with gr.Accordion("Advanced Settings", open=False):
                    
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    
                    with gr.Row():
                        width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=MAX_IMAGE_SIZE,
                            step=8,
                            value=1024,
                        )
                        
                        height = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=MAX_IMAGE_SIZE,
                            step=8,
                            value=1024,
                        )
                    
                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            label="Number of inference steps",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=4,
                        )
                        
                        guidance_scale = gr.Slider(
                            label="Guidance scale",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                            value=1.0,
                        )
                
            with gr.Column():
                result = gr.Image(label="Output Image", interactive=False, format="png", height=463)

                with gr.Row():
                    lora_style = gr.Dropdown(
                        label="Editing Style (Distilled Only)",
                        choices=["None"] + list(ADAPTER_SPECS.keys()),
                        value="None",
                        scale=1
                    )
        
        gr.Examples(
            examples=examples,
            fn=infer,
            inputs=[prompt],
            outputs=[result, seed],
            cache_examples=False,
            label="Text Generation Examples"
        )
        
        gr.Examples(
            examples=examples_images,
            fn=infer,
            inputs=[prompt, input_images],
            outputs=[result, seed],
            cache_examples=False,
            label="Image Editing Examples"
            )


        gr.Markdown("Note: This is an experimental space for FLUX.2-klein model [LoRAs](https://huggingface.co/models?other=base_model:adapter:black-forest-labs/FLUX.2-klein-4B). The Editing Style: **None** option represents the klein distilled (4B) or klein (9B) base model that is live. This app will receive an *update* soon.")
        
    input_images.upload(
        fn=update_dimensions_from_image,
        inputs=[input_images],
        outputs=[width, height]
    )
    
    mode_choice.change(
        fn=update_steps_from_mode,
        inputs=[mode_choice],
        outputs=[num_inference_steps, guidance_scale]
    )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[prompt, input_images, mode_choice, lora_style, seed, randomize_seed, width, height, num_inference_steps, guidance_scale],
        outputs=[result, seed],
        api_name="generate"
    )

if __name__ == "__main__":
    demo.queue(max_size=30).launch(mcp_server=True, ssr_mode=False, show_error=True)