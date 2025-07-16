import base64
import io
import random
import time
from typing import Any, Dict, Tuple, Union, List, Optional

import numpy as np
import torch
from PIL import Image, ImageFilter
from diffusers import (StableDiffusionImg2ImgPipeline,
                       DDIMScheduler)

from diffusers.pipelines.controlnet import \
    StableDiffusionControlNetInpaintPipeline
from diffusers import ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import MLSDdetector, HEDdetector
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from colors import ade_palette
from utils import map_colors_rgb

import runpod
from runpod.serverless.utils.rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

# --------------------------------------------------------------------------- #
#                               КОНСТАНТЫ                                     #
# --------------------------------------------------------------------------- #
MAX_SEED: int = np.iinfo(np.int32).max
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_STEPS: int = 250
LORA_DIR = "./loras"
LORA_LIST = [
    "lora_garden_architecture_Exterior_SDlife_Chiasedamme_V1.0.safetensors",
    "TS_ChineseTraditionGarden_V10.safetensors",
    "别墅的后花园_V1.safetensors",
]

logger = RunPodLogger()


# --------------------------------------------------------------------------- #
#                               ЗАГРУЗКА МОДЕЛИ                               #
# --------------------------------------------------------------------------- #
def filter_items(
    colors_list: Union[List, np.ndarray],
    items_list: Union[List, np.ndarray],
    items_to_remove: Union[List, np.ndarray],
) -> Tuple[Union[List, np.ndarray], Union[List, np.ndarray]]:
    """
    Filters items and their corresponding colors from given lists, excluding
    specified items.

    Args:
        colors_list: A list or numpy array of colors corresponding to items.
        items_list: A list or numpy array of items.
        items_to_remove: A list or numpy array of items to be removed.

    Returns:
        A tuple of two lists or numpy arrays: filtered colors and filtered
        items.
    """
    filtered_colors = []
    filtered_items = []
    for color, item in zip(colors_list, items_list):
        if item not in items_to_remove:
            filtered_colors.append(color)
            filtered_items.append(item)

    return filtered_colors, filtered_items


controlnet = [
    ControlNetModel.from_pretrained(
        "BertChristiaens/controlnet-seg-room", torch_dtype=DTYPE
    ),
    ControlNetModel.from_pretrained(
        # "lllyasviel/sd-controlnet-mlsd", torch_dtype=DTYPE
        "lllyasviel/sd-controlnet-canny",  torch_dtype=torch.float16

    ),
]

PIPELINE = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V3.0_VAE",
    # "hafsa000/interior-design",
    # "checkpoints/ruyiGardenLandscapeDesign_v10.safetensors",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=DTYPE,
    requires_safety_checker=False,
)

PIPELINE.scheduler = UniPCMultistepScheduler.from_config(
    PIPELINE.scheduler.config
)
PIPELINE.enable_xformers_memory_efficient_attention()
PIPELINE.to(DEVICE)

REFINER = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=DTYPE,
    safety_checker=None,
)
REFINER.scheduler = DDIMScheduler.from_config(REFINER.scheduler.config)
# REFINER.enable_attention_slicing()
# REFINER.enable_sequential_cpu_offload()
REFINER.to(DEVICE)


seg_image_processor = AutoImageProcessor.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)
image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)
mlsd_processor = MLSDdetector.from_pretrained("lllyasviel/Annotators")
hed_processor = HEDdetector.from_pretrained("lllyasviel/Annotators")

CURRENT_LORA: str = "None"


@torch.inference_mode()
@torch.autocast(DEVICE)
def segment_image(image):
    """
    Segments an image using a semantic segmentation model.

    Args:
        image (PIL.Image): The input image to be segmented.
        image_processor (AutoImageProcessor): The processor to prepare the
            image for segmentation.
        image_segmentor (SegformerForSemanticSegmentation): The semantic
            segmentation model used to identify different segments in the img.

    Returns:
        Image: The segmented image with each segment colored differently based
            on its identified class.
    """
    pixel_values = seg_image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = seg_image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    palette = np.array(ade_palette())

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)
    seg_image = Image.fromarray(color_seg).convert("RGB")

    return seg_image


def resize_dimensions(dimensions, target_size):
    """
    Resize PIL to target size while maintaining aspect ratio
    If smaller than target size leave it as is
    """
    width, height = dimensions

    # Check if both dimensions are smaller than the target size
    if width < target_size and height < target_size:
        return dimensions

    # Determine the larger side
    if width > height:
        # Calculate the aspect ratio
        aspect_ratio = height / width
        # Resize dimensions
        return (target_size, int(target_size * aspect_ratio))
    else:
        # Calculate the aspect ratio
        aspect_ratio = width / height
        # Resize dimensions
        return (int(target_size * aspect_ratio), target_size)


# --------------------------------------------------------------------------- #
#                           LOADING / UNLOADING LoRA                          #
# --------------------------------------------------------------------------- #
def _switch_lora(lora_name: Optional[str],
                 lora_scale: float) -> Optional[str]:
    """Load new LoRA or unload if lora_name is None. Return err str or None."""
    global CURRENT_LORA

    # -------- unload current LoRA -------- #
    if lora_name is None and CURRENT_LORA != "None":
        if hasattr(PIPELINE, "unfuse_lora"):
            PIPELINE.unfuse_lora()
        if hasattr(PIPELINE, "unload_lora_weights"):
            PIPELINE.unload_lora_weights()
        CURRENT_LORA = "None"
        return None

    # ----- nothing to do / unsupported --- #
    if lora_name is None or lora_name == CURRENT_LORA:
        return None
    if lora_name not in LORA_LIST:
        return f"Unknown LoRA '{lora_name}'."

    # --------- load new LoRA ------------- #
    try:
        if CURRENT_LORA != "None":
            if hasattr(PIPELINE, "unfuse_lora"):
                PIPELINE.unfuse_lora()
            if hasattr(PIPELINE, "unload_lora_weights"):
                PIPELINE.unload_lora_weights()

        PIPELINE.load_lora_weights(f"{LORA_DIR}/{lora_name}",
                                   use_peft_backend=True)
        if hasattr(PIPELINE, "fuse_lora"):
            PIPELINE.fuse_lora(lora_scale=lora_scale)

        CURRENT_LORA = lora_name
        return None
    except Exception as err:  # noqa: BLE001
        return f"Failed to load LoRA '{lora_name}': {err}"


# --------------------------------------------------------------------------- #
#                                ВСПОМОГАТЕЛЬНЫЕ                              #
# --------------------------------------------------------------------------- #
def url_to_pil(url: str) -> Image.Image:
    info = rp_file(url)
    return Image.open(info["file_path"]).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# --------------------------------------------------------------------------- #
#                                HANDLER                                      #
# --------------------------------------------------------------------------- #
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler."""
    try:
        payload: Dict[str, Any] = job.get("input", {})
        image_url: Optional[str] = payload.get("image_url")

        if not image_url:
            return {"error": "'image_url' is required"}
        prompt = payload.get("prompt")
        if not prompt:
            return {"error": "'prompt' is required"}
        negative_prompt = payload.get("negative_prompt", "")

        # ----------------- parameters ------------------ #
        num_images = int(payload.get("num_images", 1))
        if num_images < 1 or num_images > 8:
            return {"error": "'num_images' must be between 1 and 8."}

        guidance_scale = float(payload.get("guidance_scale", 7.5))
        prompt_strength = float(payload.get("prompt_strength", 0.8))
        steps = min(int(payload.get("steps", MAX_STEPS)), MAX_STEPS)

        DEFAULT_SEED = random.randint(0, MAX_SEED)
        seed = int(payload.get("seed", DEFAULT_SEED))
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        refiner_strength = float(payload.get("refiner_strength", 0.2))
        refiner_steps = int(payload.get("refiner_steps", 15))
        refiner_scale = float(payload.get("refiner_scale", 7.5))

        segment_conditioning_scale = float(
            payload.get("segment_conditioning_scale", 0.4))
        segment_guidance_start = float(
            payload.get("segment_guidance_start", 0.0))
        segment_guidance_end = float(payload.get("segment_guidance_end", 0.5))
        mlsd_conditioning_scale = float(
            payload.get("mlsd_conditioning_scale", 0.2))
        mlsd_guidance_start = float(payload.get("mlsd_guidance_start", 0.1))
        mlsd_guidance_end = float(payload.get("mlsd_guidance_end", 0.25))

        # ----------------- segment items ----------------- #
        mask_items_raw = payload.get("mask_items", [])
        if isinstance(mask_items_raw, str):
            mask_items = [
                s.strip() for s in mask_items_raw.split(",") if s.strip()]
        elif isinstance(mask_items_raw, list):
            mask_items = [str(s) for s in mask_items_raw]
        else:
            mask_items = ["windowpane;window",
                          "column;pillar",
                          "door;double;door"]

        # ------------------ MASK BLUR ----------------- #
        mask_blur_radius = float(payload.get("mask_blur_radius", 3))

        # ----------------- handle LoRA ----------------- #
        error = _switch_lora(payload.get("lora"),
                             payload.get("lora_scale", 1.0))
        if error:
            return {"error": error}

        image = url_to_pil(image_url)
        start = time.time()

        orig_w, orig_h = image.size
        new_width, new_height = resize_dimensions(image.size, 768)
        input_image = image.resize((new_width, new_height))

        # preprocess for segmentation controlnet
        real_seg = np.array(
            segment_image(input_image)
        )
        unique_colors = np.unique(real_seg.reshape(-1,
                                                   real_seg.shape[2]),
                                  axis=0)
        unique_colors = [tuple(color) for color in unique_colors]
        segment_items = [map_colors_rgb(i) for i in unique_colors]
        logger.info(f"Segmented items: {segment_items},"
                    f"mask items: {mask_items}")

        chosen_colors, segment_items = filter_items(
            colors_list=unique_colors,
            items_list=segment_items,
            items_to_remove=mask_items,
        )
        mask = np.zeros_like(real_seg)
        for color in chosen_colors:
            color_matches = (real_seg == color).all(axis=2)
            mask[color_matches] = 1

        image_np = np.array(input_image)
        image = Image.fromarray(image_np).convert("RGB")
        segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
        mask_image = Image.fromarray(
            (mask * 255).astype(np.uint8)).convert("RGB")
        mask_image = mask_image.filter(
            ImageFilter.GaussianBlur(radius=mask_blur_radius))

        # preprocess for mlsd controlnet
        mlsd_img = mlsd_processor(input_image)
        mlsd_img = mlsd_img.resize(image.size)

        hed_img = hed_processor(input_image)
        hed_img = hed_img.resize(image.size)
        control_images = [segmentation_cond_image, mlsd_img]
        # ------------------- generation -------------------- #
        images = PIPELINE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            strength=prompt_strength,
            guidance_scale=guidance_scale,
            generator=generator,
            image=image,
            mask_image=mask_image,
            control_image=control_images,
            controlnet_conditioning_scale=[segment_conditioning_scale,
                                           mlsd_conditioning_scale],
            control_guidance_start=[segment_guidance_start,
                                    mlsd_guidance_start],
            control_guidance_end=[segment_guidance_end,
                                  mlsd_guidance_end],
            num_images_per_prompt=num_images
        ).images

        done_images = []
        for image in images:
            image = image.resize((orig_w, orig_h),
                                 Image.Resampling.LANCZOS)
            image = image.convert("RGB")
            done_images.append(image)

        final_images = []
        # масштабируем каждую картинку
        torch.cuda.empty_cache()
        for img in done_images:
            ups = REFINER(
                prompt=prompt,
                image=img,
                strength=refiner_strength,
                num_inference_steps=refiner_steps,
                guidance_scale=refiner_scale,
            ).images[0]
            final_images.append(ups)

        elapsed = round(time.time() - start, 2)

        return {
            "images_base64": [pil_to_b64(img) for img in final_images],
            "time": elapsed,
            "steps": steps,
            "seed": seed,
            "lora": CURRENT_LORA if CURRENT_LORA != "None" else None,
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "CUDA out of memory" in str(exc):
            return {"error": "CUDA out of memo — reduce 'steps' or img size."}
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc(limit=5)}


# --------------------------------------------------------------------------- #
#                               RUN WORKER                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
