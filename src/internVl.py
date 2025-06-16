import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import gradio as gr
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.prompts import system_messages

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """
    Builds an image preprocessing pipeline with resizing, normalization, and RGB conversion.

    Args:
        input_size (int): Target width and height for resizing the image.

    Returns:
        torchvision.transforms.Compose: Composed image transformation.
    """
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_single_image(pil_img, input_size=448):
    """
    Preprocesses a single PIL image for model inference.

    Args:
        pil_img (PIL.Image.Image): The image to be preprocessed.
        input_size (int, optional): Size to which the image is resized. Defaults to 448.

    Returns:
        torch.Tensor: The preprocessed image tensor of shape (1, 3, H, W).
    """
    transform = build_transform(input_size)
    pil_img = pil_img.resize((input_size, input_size))
    tensor_img = transform(pil_img).unsqueeze(0)
    return tensor_img


model_path = "OpenGVLab/InternVL3-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")
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
print("Model loaded successfully!")


def pairs_to_messages(history_pairs):
    """
    Converts a list of (user, assistant) message tuples into a list of message dictionaries.

    This format is required by Gradio's Chatbot component, where each message has a 'role' and 'content'.

    Args:
        history_pairs (list of tuple): A list of (user_message, assistant_message) pairs.

    Returns:
        list of dict: A flat list of message dictionaries formatted as {'role': ..., 'content': ...}.
    """  
    messages = []
    for user_msg, assistant_msg in history_pairs:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    return messages


def chat_interface(query, image, history_pairs, prompt_type):
    """
    Handles a chat turn by sending user input and optional image to the model, with dynamic system prompts.

    Preprocesses the image (if any), formats the prompt with a system message, sends it to the model,
    and returns the updated chat history.

    Args:
        query (str): The user's text input.
        image (PIL.Image.Image or np.ndarray or None): Optional image to include in the prompt.
        history_pairs (list of tuple): Previous (user, assistant) message pairs.
        prompt_type (str): The system prompt type key to select a system message from.

    Returns:
        tuple:
            - list of dict: Updated conversation formatted for Gradio's Chatbot component.
            - list of tuple: Updated raw (user, assistant) history pairs for future turns.
    """
    generation_config = {"max_new_tokens": 1024, "do_sample": True}

    system_prompt = system_messages.get(prompt_type, system_messages["default"])
    full_query = f"{system_prompt}\n{query}"

    pixel_values = None
    if image is not None:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        try:
            pixel_values = preprocess_single_image(image, 448)
            pixel_values = pixel_values.to(
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            ).to(device)
            if "<image>" not in full_query:
                full_query = "<image>\n" + full_query
        except Exception as e:
            print(f"Error processing image: {e}")
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
        print(f"Error during model.chat: {e}")
        if history_pairs is None:
            history_pairs = []
        updated_history = history_pairs + [
            (query, "I'm sorry, I encountered an error processing your request.")
        ]
        response = "I'm sorry, I encountered an error processing your request."

    return pairs_to_messages(updated_history), updated_history


def reset_history():
    """
    Resets the chat history to an empty state.

    This function clears both the display messages and the internal history tracking,
    typically used when the user clicks a "Reset" button in the UI.

    Returns:
        tuple:
            - list: An empty list for Gradio Chatbot display.
            - list: An empty list for internal state tracking.
    """
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

if __name__ == "__main__":
    print("Starting Gradio interface...")
    demo.launch()
