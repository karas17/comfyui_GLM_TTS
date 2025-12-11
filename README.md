# ComfyUI-GLM-TTS

ComfyUI nodes for [GLM-TTS](https://github.com/zai-org/GLM-TTS), a high-quality text-to-speech system supporting zero-shot voice cloning.

## Installation

1.  Clone this repository into your `ComfyUI/custom_nodes/` directory.
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/karas17/ComfyUI-GLM-TTS.git
    ```
    *(If you downloaded this folder directly, just rename it to `ComfyUI-GLM-TTS`)*

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    Note: 
    - `torch`, `torchaudio`, and `transformers` are required but expected to be in your ComfyUI environment.
    - `onnxruntime` (or `onnxruntime-gpu`) is required for the frontend. If you encounter issues, install `onnxruntime-gpu`.

3.  Models:
    By default, models are loaded from `ComfyUI/models/GLM-TTS`. If missing, they will be auto-downloaded from HuggingFace `zai-org/GLM-TTS` to that location.
    Windows 示例: `<ComfyUI 安装目录>\models\GLM-TTS`
    
    Structure expected inside the model path:
    - `speech_tokenizer/`
    - `vq32k-phoneme-tokenizer/`
    - `llm/`
    - `flow/`
    - `frontend/` (contains `campplus.onnx`, `spk2info.pt`)
    
    You can also provide an absolute `model_path` to a prepared directory with the above structure.

## Usage

1.  **GLM-TTS Loader**:
    -   `model_path`: Path to your models. Default `GLM-TTS` resolves to `ComfyUI/models/GLM-TTS`. Absolute paths are supported（Windows 示例：`<ComfyUI 安装目录>\models\GLM-TTS`）.
    -   `sample_rate`: 24000 (default) or 32000.
    
2.  **GLM-TTS Sampler**:
    -   `model`: Connect from Loader.
    -   `text`: Text to speak.
    -   `reference_audio` (Optional): Audio for voice cloning.
    -   `reference_text` (Optional): Transcript of the reference audio (improves cloning).

## Notes
-   The first run might be slow as it loads models.
-   Ensure your `reference_audio` is clear.
