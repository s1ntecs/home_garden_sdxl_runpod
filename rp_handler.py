# import cv2
import base64, io, random, time, numpy as np, torch
from typing import Any, Dict
from PIL import Image, ImageFilter

from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel, UniPCMultistepScheduler, DDIMScheduler
)

from controlnet_aux import MidasDetector
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from colors import ade_palette
from utils import map_colors_rgb
import runpod
from runpod.serverless.utils.rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

# --------------------------- КОНСТАНТЫ ----------------------------------- #
MAX_SEED = np.iinfo(np.int32).max
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_STEPS = 250
TARGET_RES = 1024  # SDXL рекомендует 1024×1024

POOL_GROUND_ALLOW = [
    "grass", "lawn", "field", "earth", "dirt", "soil",
    "ground", "sand", "gravel",
    "floor", "floor-wood", "floor-tile", "floor-marble", "floor-other",
    "pavement", "sidewalk", "path", "trail", "courtyard", "deck", "patio"
]


logger = RunPodLogger()


# ------------------------- ФУНКЦИИ-ПОМОЩНИКИ ----------------------------- #
def filter_items(colors_list, items_list, items_to_remove):
    keep_c, keep_i = [], []
    for c, it in zip(colors_list, items_list):
        if it not in items_to_remove:
            keep_c.append(c)
            keep_i.append(it)
    return keep_c, keep_i


def resize_dimensions(dimensions, target_size):
    w, h = dimensions
    if w < target_size and h < target_size:
        return dimensions
    if w > h:
        ar = h / w
        return target_size, int(target_size * ar)
    ar = w / h
    return int(target_size * ar), target_size


def url_to_pil(url: str) -> Image.Image:
    info = rp_file(url)
    return Image.open(info["file_path"]).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ------------------------- ЗАГРУЗКА МОДЕЛЕЙ ------------------------------ #
controlnet = [
    ControlNetModel.from_pretrained(
        "SargeZT/sdxl-controlnet-seg",
        torch_dtype=DTYPE
    ),
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=DTYPE,
        use_safetensors=True
    )
]

PIPELINE = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    # "RunDiffusion/Juggernaut-XL-v9",
    # "SG161222/RealVisXL_V5.0",
    # "misri/cyberrealisticPony_v90Alt1",
    "John6666/epicrealism-xl-vxvii-crystal-clear-realism-sdxl",
    controlnet=controlnet,
    torch_dtype=DTYPE,
    # variant="fp16" if DTYPE == torch.float16 else None,
    safety_checker=None,
    requires_safety_checker=False,
    add_watermarker=False,
    use_safetensors=True,
    resume_download=True,
)
PIPELINE.scheduler = UniPCMultistepScheduler.from_config(
    PIPELINE.scheduler.config)
PIPELINE.enable_xformers_memory_efficient_attention()
PIPELINE.to(DEVICE)

REFINER = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=DTYPE,
    variant="fp16" if DTYPE == torch.float16 else None,
    safety_checker=None,
)
REFINER.scheduler = DDIMScheduler.from_config(REFINER.scheduler.config)
REFINER.to(DEVICE)

# --- детекторы / сегментатор --- #
seg_image_processor = AutoImageProcessor.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)
image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)
# canny_detector = CannyDetector()
# hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")

midas = MidasDetector.from_pretrained("lllyasviel/ControlNet")

CURRENT_LORA = "None"


# ------------------------- ОСНОВНОЙ HANDLER ------------------------------ #
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = job.get("input", {})
        image_url = payload.get("image_url")
        if not image_url:
            return {"error": "'image_url' is required"}

        prompt = payload.get("prompt")
        if not prompt:
            return {"error": "'prompt' is required"}

        negative_prompt = payload.get("negative_prompt", "")
        num_images = max(1, min(int(payload.get("num_images", 1)), 8))
        guidance_scale = float(payload.get("guidance_scale", 7.5))
        prompt_strength = float(payload.get("prompt_strength", 0.8))
        steps = min(int(payload.get("steps", MAX_STEPS)), MAX_STEPS)

        seed = int(payload.get("seed", random.randint(0, MAX_SEED)))
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # refiner
        refiner_strength = float(payload.get("refiner_strength", 0.2))
        refiner_steps = int(payload.get("refiner_steps", 15))
        refiner_scale = float(payload.get("refiner_scale", 7.5))

        # control scales
        seg_scale = float(payload.get("segment_conditioning_scale", 0.4))
        seg_g_start = float(payload.get("segment_guidance_start", 0.0))
        seg_g_end = float(payload.get("segment_guidance_end", 0.8))
        depth_scale = float(payload.get("depth_conditioning_scale", 0.4))
        depth_g_start = float(payload.get("depth_guidance_start", 0.0))
        depth_g_end = float(payload.get("depth_guidance_end", 0.8))

        mask_items_raw = payload.get("mask_items", [])
        if isinstance(mask_items_raw, str):
            mask_items = [s.strip() for s in mask_items_raw.split(",") if s.strip()]
        elif isinstance(mask_items_raw, list):
            mask_items = [str(s) for s in mask_items_raw]
        else:
            mask_items = POOL_GROUND_ALLOW

        mask_blur_radius = float(payload.get("mask_blur_radius", 3))

        # ---------- препроцессинг входа ------------
        image_pil = url_to_pil(image_url)
        orig_w, orig_h = image_pil.size

        # input_image = image_pil.resize((new_w, new_h))
        input_image = image_pil

        # ---- сегментация ----
        with torch.inference_mode(), torch.autocast(DEVICE):
            pixel = seg_image_processor(
                input_image,
                return_tensors="pt").pixel_values
            outputs = image_segmentor(pixel)
        seg = seg_image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[input_image.size[::-1]])[0]
        palette = np.array(ade_palette())
        color_seg = np.zeros((*seg.shape, 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label] = color
        seg_pil = Image.fromarray(color_seg).convert("RGB")

        # ---- маска из сегментов ----
        unique_colors = [tuple(c) for c in np.unique(color_seg.reshape(-1, 3), axis=0)]
        seg_items = [map_colors_rgb(c) for c in unique_colors]
        logger.info(f"[MASK] unique segments: {seg_items}")
        logger.info(f"[MASK] items to remove: {mask_items}")

        # chosen, _ = filter_items(unique_colors, seg_items, mask_items)
        # выбираем только сегменты указанные в mask_items_raw
        chosen = [c for c, name in zip(unique_colors,
                                       seg_items) if name in mask_items]
        logger.info(f"[MASK] chosen segment colors: {chosen}")
        mask_np = np.zeros_like(color_seg)

        for c in chosen:
            mask_np[(color_seg == c).all(axis=2)] = 1
        masked_pixels = int(mask_np.sum())
        logger.info(f"[MASK] masked pixels count: {masked_pixels}")

        mask_pil = Image.fromarray(
            (mask_np * 255).astype(np.uint8)).convert("RGB")

        mask_pil = mask_pil.filter(
            ImageFilter.GaussianBlur(radius=mask_blur_radius))

        # ---- canny ----
        # canny_pil = canny_detector(input_image)
        # logger.info(f"[CANNY] size: {canny_pil.size}")  

        # ---- depth --------------------------------------------------------------
        depth_cond = midas(image_pil)
        # ------------------ генерация ---------------- #
        images = PIPELINE(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            mask_image=mask_pil,
            control_image=[seg_pil, depth_cond],
            controlnet_conditioning_scale=[seg_scale, depth_scale],
            control_guidance_start=[seg_g_start, depth_g_start],
            control_guidance_end=[seg_g_end, depth_g_end],
            strength=prompt_strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=1,
            height=input_image.height, width=input_image.width
        ).images

        # ---- up-scale через рефайнер ----
        final = []
        torch.cuda.empty_cache()
        for im in images:
            im = im.resize((orig_w, orig_h), Image.Resampling.LANCZOS).convert("RGB")
            ref = REFINER(
                prompt=prompt, image=im, strength=refiner_strength,
                num_inference_steps=refiner_steps, guidance_scale=refiner_scale
            ).images[0]
            final.append(ref)

        return {
            "images_base64": [pil_to_b64(i) for i in final],
            "time": round(time.time() - job["created"], 2) if "created" in job else None,
            "steps": steps, "seed": seed,
            "lora": CURRENT_LORA if CURRENT_LORA != "None" else None,
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "CUDA out of memory" in str(exc):
            return {"error": "CUDA OOM — уменьшите 'steps' или размер изображения."}
        return {"error": str(exc)}
    except Exception as exc:
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc(limit=5)}


# ------------------------- RUN WORKER ------------------------------------ #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
