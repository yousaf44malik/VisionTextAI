# 🧠 VisionText AI — InternVL3 Chat Interface

**VisionText AI** is a multimodal AI assistant prototype powered by [InternVL3-1B](https://huggingface.co/OpenGVLab/InternVL3-1B). It lets users engage in image + text conversations via a clean Gradio interface, ideal for testing vision-language models locally.

> 🚧 This is an early prototype. Future phases will add FastAPI, NLP models, Docker support, and full API endpoints.

---

## 🌟 Features

- 🖼️ Chat with images and text using InternVL3
- 🗨️ Multi-turn conversation memory
- ⚙️ Modular configuration via `config.yml`
- 🎛️ Choose from predefined system prompts (e.g., medical, default, reasoning)

---

## 📁 Project Structure

visiontextai/
├── configs/
│ ├── config.yml # Device/model settings
│ └── prompts.py # Customizable prompt types
│
├── src/
│ └── internVl.py # Gradio + inference logic
│
├── main.py # Entry point to launch the app
└── README.md # You're here!
---

## 🚀 Getting Started

**1. Clone the repository and set up the environment:**
git clone https://github.com/yousaf44malik/visiontextai.git
cd visiontextai
conda create -n visiontextai python=3.11 -y
conda activate visiontextai
**2. Install dependencies:**
pip install torch torchvision transformers gradio pillow numpy einops timm

**3. Configure the model:**

Edit `configs/config.yml` as needed:
model_path: OpenGVLab/InternVL3-1B # Use HuggingFace path or local folder
device: cuda # "cuda" for GPU, "cpu" for CPU


**4. Run the app:**

python main.py

After startup, open your browser to:

http://127.0.0.1:7860

---

## 🧪 Example Usage

- Enter a prompt like:  
  *"Describe this image in detail."*
- Optionally upload an image and ask:  
  *"What animal is this?"*
- Use the dropdown to switch between system prompt types:  
  `default`, `medical`, `reasoning`, etc.

---

## 🔜 Roadmap

- ✅ InternVL3 chat UI with image + text input
- 🔜 FastAPI support
- 🔜 LLaVA and BLIP-2 integrations
- 🔜 Dockerized deployment
- 🔜 NLP-only endpoints (`/process-nlp`)
- 🔜 Vision-only endpoints (`/process-image`)

---

## 🤝 Contributing

Pull requests, issues, and discussions are welcome!

---

## 📄 License

This project is under the MIT License. See `LICENSE` for details.

---

*Let me know if you'd like to include a screenshot, demo video link, or CI badge later on.*
