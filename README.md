# **FLUX.2-Klein-LoRA-Studio**

> A Gradio-based demonstration for the black-forest-labs/FLUX.2-klein-9B model with lazy-loaded LoRA adapters for advanced image editing and style application. Supports specialized LoRAs like Klein-Delight-Style, with fast inference using bfloat16 and dynamic adapter loading to optimize memory. Features auto-resizing to multiples of 16, seed randomization, and guidance scale adjustments.

<img width="1401" height="818" alt="Screenshot 2026-02-02 100856" src="https://github.com/user-attachments/assets/0e92b34c-c26d-42be-9d9d-759e446b69fb" />
<img width="1368" height="649" alt="Screenshot 2026-02-02 100917" src="https://github.com/user-attachments/assets/84485865-3ed7-4ff4-886f-db0756ba73e4" />

## Features
- **Image Upload Support**: Upload via file, webcam, or clipboard; auto-resizes to fit model requirements (up to 1024px, divisible by 16).
- **Lazy LoRA Loading**: Adapters load on-demand only when selected, minimizing VRAM usage.
- **Advanced Editing Tasks**:
  - Base Model: Default FLUX.2-klein-9B for general edits.
  - Klein-Delight-Style: Apply delightful, stylized transformations.
- **Rapid Inference**: 4-step default with bfloat16 on CUDA.
- **Custom Theme**: OrangeRedTheme with clean, responsive layout.
- **Style Gallery**: Interactive gallery for selecting styles with previews.
- **Examples**: Curated scenarios with prompts and images.
- **Queueing**: Up to default Gradio queue limits for concurrent jobs.

## Prerequisites
- Python 3.10 or higher.
- CUDA-compatible GPU (required for bfloat16 and efficient inference).
- Stable internet for initial model/LoRA downloads.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/FLUX.2-Klein-LoRA-Studio.git
   cd FLUX.2-Klein-LoRA-Studio
   ```
2. Install dependencies:
   First, install pre-requirements:
   ```
   pip install -r pre-requirements.txt
   ```
   Then, install main requirements:
   ```
   pip install -r requirements.txt
   ```
   **pre-requirements.txt content:**
   ```
   pip>=23.0.0
   ```
   **requirements.txt content:**
   ```
   git+https://github.com/huggingface/diffusers.git
   transformers==4.57.3
   huggingface_hub
   spaces==0.43.0
   sentencepiece
   torch==2.8.0
   bitsandbytes
   torchvision
   accelerate
   torchao
   hf_xet
   gradio #gradio@6
   numpy
   peft
   av
   ```
3. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860`.

## Usage
1. **Upload Image**: Add an image via the input component.
2. **Select Style**: Click on a style from the gallery (e.g., None or Klein-Delight-Style).
3. **Enter Prompt**: Describe the edit (e.g., "a man with a red superhero mask").
4. **Configure (Optional)**: Adjust seed, guidance scale, steps in Advanced Settings.
5. **Apply Style**: Click "Apply Style" to generate output.

### Supported Styles
| Style              | Use Case                          |
|--------------------|-----------------------------------|
| None              | Base FLUX.2-klein-9B edits       |
| Klein-Delight-Style | Delightful stylized transformations |

## Examples
| Input Image    | Prompt Example | Style              |
|----------------|---------------|--------------------|
| examples/2.jpg | "Relight the image to remove all existing lighting conditions and replace them with neutral, uniform illumination. Apply soft, evenly distributed lighting with no directional shadows, no harsh highlights, and no dramatic contrast. Maintain the original identity of all subjects exactly—preserve facial structure, skin tone, proportions, expressions, hair, clothing, and textures. Do not alter pose, camera angle, background geometry, or image composition. Lighting should appear balanced, and studio-neutral, similar to diffuse overcast or a soft lightbox setup. Ensure consistent exposure across the entire image with realistic depth and subtle shading only where necessary for form." | None (or selected) |
| examples/1.jpg | "cinematic polaroid with soft grain subtle vignette gentle lighting white frame handwritten photographed by prithivMLmods preserving realistic texture and details" | None (or selected) |

## Troubleshooting
- **Adapter Loading**: First selection downloads LoRA; monitor console.
- **OOM**: Reduce steps/resolution; clear cache with `torch.cuda.empty_cache()`.
- **No Output**: Ensure valid image and prompt; check console for errors.
- **Tensor Mismatch**: Auto-resizing handles this; verify image dimensions.

## Contributing
Contributions welcome! Add new styles to `LORA_STYLES`, improve resizing logic, or enhance prompts. Submit pull requests via the repository.

Repository: [https://github.com/PRITHIVSAKTHIUR/FLUX.2-Klein-LoRA-Studio.git](https://github.com/PRITHIVSAKTHIUR/FLUX.2-Klein-LoRA-Studio.git)

## License
Apache License 2.0. See [LICENSE](LICENSE) for details.
Built by Prithiv Sakthi. Report issues via the repository.
