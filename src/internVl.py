import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import gradio as gr
import sys
import os

# Add root to path for loading config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.prompts import system_messages

# --------------------
# Image Preprocessing
# --------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def preprocess_single_image(pil_img, input_size=448):
    transform = build_transform(input_size)
    pil_img = pil_img.resize((input_size, input_size))
    tensor_img = transform(pil_img).unsqueeze(0)  # shape: (1, 3, H, W)
    return tensor_img

# --------------------
# Model Loading
# --------------------
model_path = "OpenGVLab/InternVL3-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    use_flash_attn=torch.cuda.is_available(),
    trust_remote_code=True
).eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=False
)
print("Model loaded successfully!")

# --------------------
# Conversation Helpers
# --------------------
def pairs_to_messages(history_pairs):
    messages = []
    for user_msg, assistant_msg in history_pairs:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    return messages

def chat_interface(query, image, history_pairs, prompt_type):
    generation_config = {"max_new_tokens": 1024, "do_sample": True}

    # Load system prompt
    system_prompt = system_messages.get(prompt_type, system_messages["default"])
    full_query = f"{system_prompt}\n{query}"

    pixel_values = None
    if image is not None:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        try:
            pixel_values = preprocess_single_image(image, 448)
            pixel_values = pixel_values.to(torch.bfloat16 if torch.cuda.is_available() else torch.float32).to(device)
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
            return_history=True
        )
    except Exception as e:
        print(f"Error during model.chat: {e}")
        if history_pairs is None:
            history_pairs = []
        updated_history = history_pairs + [(query, "I'm sorry, I encountered an error processing your request.")]
        response = "I'm sorry, I encountered an error processing your request."

    return pairs_to_messages(updated_history), updated_history

def reset_history():
    return [], []

# --------------------
# Gradio Interface
# --------------------
with gr.Blocks() as demo:
    gr.Markdown("# InternVL3 Chat Interface")

    chatbot = gr.Chatbot(label="Conversation", type="messages")
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(placeholder="Enter your message here", label="Message")
        img_input = gr.Image(label="Optional Image Input", type="pil")
        prompt_selector = gr.Dropdown(
            choices=list(system_messages.keys()),
            value="default",
            label="Prompt Type"
        )

    with gr.Row():
        send_btn = gr.Button("Send")
        reset_btn = gr.Button("Reset Conversation")

    send_btn.click(
        fn=chat_interface,
        inputs=[txt, img_input, state, prompt_selector],
        outputs=[chatbot, state],
        scroll_to_output=True
    )

    reset_btn.click(fn=reset_history, outputs=[chatbot, state])

if __name__ == "__main__":
    print("Starting Gradio interface...")
    demo.launch()
