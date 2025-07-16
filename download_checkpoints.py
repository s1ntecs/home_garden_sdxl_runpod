import os
import torch

from diffusers import (
    ControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionXLControlNetInpaintPipeline,
    AutoencoderKL,
    StableDiffusionXLImg2ImgPipeline
)

from controlnet_aux import MLSDdetector, HEDdetector
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

LORA_NAMES = [
]


# ------------------------- загрузка весов -------------------------
def fetch_checkpoints() -> None:
    """Скачиваем SD-чекпойнт, LoRA-файлы и все внешние зависимости."""

    for fname in LORA_NAMES:
        hf_hub_download(
            repo_id="sintecs/interior",
            filename=fname,
            local_dir="loras",
            local_dir_use_symlinks=False,
        )


# ------------------------- пайплайн -------------------------
def get_pipeline():
    controlnet = [
        ControlNetModel.from_pretrained(
            "SargeZT/sdxl-controlnet-seg",
            torch_dtype=DTYPE,
        ),
        ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=DTYPE,
        )
    ]
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                        torch_dtype=torch.float16,
                                        use_safetensors=True)

    PIPELINE = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "rubbrband/albedobaseXL_v21",
        torch_dtype=torch.float16,
        add_watermarker=False,
        controlnet=controlnet,
        vae=vae,
    ).to(DEVICE)
    PIPELINE.scheduler = UniPCMultistepScheduler.from_config(
        PIPELINE.scheduler.config)

    StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=DTYPE,
        variant="fp16" if DTYPE == torch.float16 else None,
        safety_checker=None,
    )

    AutoImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    MLSDdetector.from_pretrained("lllyasviel/Annotators")
    HEDdetector.from_pretrained("lllyasviel/Annotators")
    return


if __name__ == "__main__":
    fetch_checkpoints()
    get_pipeline()
