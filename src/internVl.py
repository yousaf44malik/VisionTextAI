import logging
import torch
from PIL import Image as PILImage
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import gradio as gr
import numpy as np
from configs import load_config
from configs.prompts import system_messages
from typing import List, Tuple, Dict, Any, Union
from torchvision.transforms import Compose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VisionTextAI")

config = load_config()
model_path: str = config["model_path"]
device: str = config["device"]
input_size: int = config.get("input_size", 448)
generation_config: Dict[str, Any] = config.get(
    "generation_config", {"max_new_tokens": 1024, "do_sample": True}
)
imagenet_mean: Tuple[float, float, float] = tuple(
    config.get("imagenet_mean", [0.485, 0.456, 0.406])
)
imagenet_std: Tuple[float, float, float] = tuple(
    config.get("imagenet_std", [0.229, 0.224, 0.225])
)

logger.info(f"Usingm model: {model_path}")
logger.info(f"Running on device: {device}")


def build_transform(size: int) -> Compose:
    """
    Builds an image preprocessing pipeline with resizing, normalization, and RGB conversion.

    Args:
        input_size (int): Target width and height for resizing the image.

    Returns:
        torchvision.transforms.Compose: Composed image transformation pipeline.
    """
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def preprocess_single_image(pil_img: PILImage.Image, size: int = 448) -> torch.Tensor:
    """
    Preprocesses a single PIL image for model inference.

    Args:
        pil_img (PIL.Image.Image): The image to be preprocessed.
        input_size (int, optional): Size to which the image is resized. Defaults to 448.

    Returns:
        torch.Tensor: The preprocessed image tensor of shape (1, 3, H, W).
    """
    transform = build_transform(size)
    pil_img = pil_img.resize((size, size))
    tensor_img = transform(pil_img).unsqueeze(0)
    return tensor_img


logger.info(f"Loading model on {device}...")
model = (
    AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        use_flash_attn=torch.cuda.is_available(),
        trust_remote_code=True,
    )
    .eval()
    .to(device)
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, use_fast=False
)
logger.info("Model loaded successfully!")


def pairs_to_messages(history_pairs: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """
    Converts list of (user, assistant) message pairs into a flat list of dicts for Gradio Chatbot.
    """
    messages = []
    for user_msg, assistant_msg in history_pairs:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    return messages


def chat_interface(
    query: str,
    image: Union[PILImage.Image, np.ndarray, None],
    history_pairs: List[Tuple[str, str]],
    prompt_type: str,
) -> Tuple[List[Dict[str, str]], List[Tuple[str, str]]]:
    """
    Processes a user query and optional image, returns updated Gradio chat messages and raw history.
    """
    system_prompt = system_messages.get(prompt_type, system_messages["default"])
    full_query = f"{system_prompt}\n{query}"

    pixel_values = None
    if image is not None:
        if not isinstance(image, PILImage.Image):
            image = PILImage.fromarray(image)
        try:
            pixel_values = preprocess_single_image(image, input_size)
            pixel_values = pixel_values.to(
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            ).to(device)
            if "<image>" not in full_query:
                full_query = "<image>\n" + full_query
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            pixel_values = None

    try:
        response, updated_history = model.chat(
            tokenizer,
            pixel_values,
            full_query,
            generation_config,
            history=history_pairs,
            return_history=True,
        )
    except Exception as e:
        logger.error(f"Error during model.chat: {e}", exc_info=True)
        updated_history = history_pairs or []
        updated_history.append(
            (query, "I'm sorry, I encountered an error processing your request.")
        )
        response = "I'm sorry, I encountered an error processing your request."

    return pairs_to_messages(updated_history), updated_history


def reset_history() -> Tuple[List[Any], List[Tuple[str, str]]]:
    """Resets chat history."""
    return [], []


with gr.Blocks() as demo:
    gr.Markdown("# InternVL3 Chat Interface")

    chatbot = gr.Chatbot(label="Conversation", type="messages")
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(placeholder="Enter your message here", label="Message")
        img_input = gr.Image(label="Optional Image Input", type="pil")
        prompt_selector = gr.Dropdown(
            choices=list(system_messages.keys()), value="default", label="Prompt Type"
        )

    with gr.Row():
        send_btn = gr.Button("Send")
        reset_btn = gr.Button("Reset Conversation")

    send_btn.click(
        fn=chat_interface,
        inputs=[txt, img_input, state, prompt_selector],
        outputs=[chatbot, state],
        scroll_to_output=True,
    )

    reset_btn.click(fn=reset_history, outputs=[chatbot, state])
