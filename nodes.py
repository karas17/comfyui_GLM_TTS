import os
import sys
import glob
import torch
import torchaudio
import logging
import importlib.util
import folder_paths
from functools import partial
from transformers import AutoTokenizer, LlamaForCausalLM

current_dir = os.path.dirname(os.path.abspath(__file__))
glmtts_src_candidates = [
    os.path.join(current_dir, "glmtts_src"),
    os.path.join(current_dir, "GLM-TTS"),
]
for _p in glmtts_src_candidates:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LLM_SEQ_INP_LEN = 750

def set_seed(seed):
    import random
    import numpy as np
    try:
        seed_int = int(seed)
    except Exception:
        seed_int = 0
    max_seed = 2**32
    seed_int = seed_int % max_seed
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)
    random.seed(seed_int)
    np.random.seed(seed_int)

ASR_MODEL = None

def auto_transcribe_reference_audio(file_path, language="zh"):
    global ASR_MODEL
    try:
        import whisper
    except Exception:
        print("GLM-TTS: openai-whisper not available, skip auto transcription.")
        return ""
    if ASR_MODEL is None:
        asr_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            ASR_MODEL = whisper.load_model("small", device=asr_device)
        except Exception as e:
            print(f"GLM-TTS: failed to load Whisper model: {e}")
            return ""
    lang_arg = None if language == "auto" else language
    try:
        if lang_arg is None:
            result = ASR_MODEL.transcribe(file_path)
        else:
            result = ASR_MODEL.transcribe(file_path, language=lang_arg)
    except Exception as e:
        print(f"GLM-TTS: Whisper transcription error: {e}")
        return ""
    text = result.get("text", "")
    if not isinstance(text, str):
        return ""
    return text.strip()

def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25, temperature=1.0):
    prob, indices = [], []
    cum_prob = 0.0
    scaled_scores = weighted_scores / temperature
    sorted_value, sorted_idx = scaled_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)]
    return top_ids

def random_sampling(weighted_scores):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
    return top_ids

def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1, temperature=1.0):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k, temperature=temperature)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores)
    return top_ids

class LLMWrapper(torch.nn.Module):
    def __init__(self, llama_cfg_path, llama_path):
        super().__init__()
        from transformers import LlamaConfig
        config = LlamaConfig.from_json_file(llama_cfg_path)
        self.llama = LlamaForCausalLM(config)
        self.llama = LlamaForCausalLM.from_pretrained(llama_path, dtype=torch.float32).to(DEVICE)
        self.llama_embedding = self.llama.model.embed_tokens
        self.special_token_ids = None
        self.ats = None
        self.ate = None
        self.boa = None
        self.eoa = None
        self.pad = None
        self.mode = "PRETRAIN"

    def set_runtime_vars(self, special_token_ids):
        required_keys = ['ats', 'ate', 'boa', 'eoa', 'pad']
        assert all(k in special_token_ids for k in required_keys)
        self.special_token_ids = special_token_ids
        self.ats = special_token_ids['ats']
        self.ate = special_token_ids['ate']
        self.boa = special_token_ids['boa']
        self.eoa = special_token_ids['eoa']
        self.pad = special_token_ids['pad']

    @torch.inference_mode()
    def inference(self, text, text_len, prompt_text, prompt_text_len, prompt_speech_token, prompt_speech_token_len, beam_size=1, sampling=25, max_token_text_ratio=20, min_token_text_ratio=2, sample_method="ras", spk=None):
        device = text.device
        if prompt_speech_token_len != 0 and prompt_text_len != 0:
            prompt_speech_token = prompt_speech_token + self.ats
        boa_tensor = torch.tensor([self.boa], device=device).unsqueeze(0)
        input_ids = torch.cat([prompt_text, text, boa_tensor, prompt_speech_token], dim=1).to(torch.long)
        inputs_embeds = self.llama_embedding(input_ids)
        min_len = int(text_len * min_token_text_ratio)
        max_len = int(text_len * max_token_text_ratio)
        out_tokens = []
        past_key_values = None
        for i in range(max_len):
            model_input = {
                "inputs_embeds": inputs_embeds,
                "output_hidden_states": True,
                "return_dict": True,
                "use_cache": True,
                "past_key_values": past_key_values
            }
            outputs = self.llama(**model_input)
            past_key_values = outputs['past_key_values']
            logp = outputs['logits'][:, -1].log_softmax(dim=-1)
            if sample_method == "ras":
                if i < min_len:
                    logp[:, self.eoa] = -float('inf')
                top_ids = ras_sampling(logp.squeeze(dim=0), out_tokens, sampling, top_p=0.8, top_k=sampling, win_size=10, tau_r=0.1, temperature=1.0).item()
            elif sample_method == "greedy":
                top_ids = logp.squeeze(dim=0).argmax(dim=-1).item()
            elif sample_method == "top_p":
                top_ids = nucleus_sampling(logp.squeeze(dim=0), top_p=0.8, top_k=sampling, temperature=1.0).item()
            elif sample_method == "topk":
                prob, indices = logp.squeeze(dim=0).softmax(dim=-1).topk(sampling)
                top_ids_index = prob.multinomial(beam_size, replacement=True)
                top_ids = indices[top_ids_index].item()
            else:
                raise ValueError(f"Unknown sample_method: {sample_method}")
            if top_ids == self.eoa:
                break
            out_tokens.append(top_ids)
            inputs_embeds = self.llama_embedding(torch.LongTensor([top_ids]).to(device))[None]
        return torch.tensor([out_tokens], dtype=torch.int64, device=device) - self.ats

def get_special_token_ids(tokenize_fn):
    _special_token_ids = {
        "ats": "<|audio_0|>",
        "ate": "<|audio_32767|>",
        "boa": "<|begin_of_audio|>",
        "eoa": "<|user|>",
        "pad": "<|endoftext|>",
    }
    special_token_ids = {}
    endoftext_id = tokenize_fn("<|endoftext|>")[0]
    for k, v in _special_token_ids.items():
        __ids = tokenize_fn(v)
        if len(__ids) != 1:
            raise AssertionError(f"Token '{k}' ({v}) encoded to multiple tokens: {__ids}")
        if __ids[0] < endoftext_id:
            raise AssertionError(f"Token '{k}' ({v}) ID {__ids[0]} is smaller than endoftext ID {endoftext_id}")
        special_token_ids[k] = __ids[0]
    return special_token_ids

def _assert_shape_and_get_len(token):
    assert token.ndim == 2 and token.shape[0] == 1
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32).to(token.device)
    return token_len

def local_llm_forward(llm, prompt_text_token, tts_text_token, prompt_speech_token, beam_size=1, sampling=25, sample_method="ras"):
    prompt_text_token_len = _assert_shape_and_get_len(prompt_text_token)
    tts_text_token_len = _assert_shape_and_get_len(tts_text_token)
    prompt_speech_token_len = _assert_shape_and_get_len(prompt_speech_token)
    tts_speech_token = llm.inference(
        text=tts_text_token,
        text_len=tts_text_token_len,
        prompt_text=prompt_text_token,
        prompt_text_len=prompt_text_token_len,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=prompt_speech_token_len,
        beam_size=beam_size,
        sampling=sampling,
        sample_method=sample_method,
        spk=None,
    )
    return tts_speech_token[0].tolist()

def local_flow_forward(flow, token_list, prompt_speech_tokens, speech_feat, embedding):
    wav, full_mel = flow.token2wav_with_cache(
        token_list,
        prompt_token=prompt_speech_tokens,
        prompt_feat=speech_feat,
        embedding=embedding,
    )
    return wav.detach().cpu(), full_mel

def get_cached_prompt(cache, synth_text_token, device=DEVICE):
    cache_text = cache["cache_text"]
    cache_text_token = cache["cache_text_token"]
    cache_speech_token = cache["cache_speech_token"]
    def __len_cache_text_token():
        return sum(map(lambda x: x.shape[1], cache_text_token))
    def __len_cache_speech_token():
        return sum(map(len, cache_speech_token))
    text_len = __len_cache_text_token()
    ta_ratio = __len_cache_speech_token() / (text_len if text_len > 0 else 1.0)
    __len_synth_text_token = synth_text_token.shape[1]
    __len_synth_audi_token_estim = int(ta_ratio * __len_synth_text_token)
    while (__len_cache_speech_token() + __len_synth_audi_token_estim > MAX_LLM_SEQ_INP_LEN):
        if len(cache_speech_token) <= 1:
            break
        cache_text.pop(1)
        cache_text_token.pop(1)
        cache_speech_token.pop(1)
    prompt_text_token_from_cache = []
    for a_token in cache_text_token:
        prompt_text_token_from_cache.extend(a_token.squeeze().tolist())
    prompt_text_token = torch.tensor([prompt_text_token_from_cache]).to(device)
    speech_tokens = []
    for a_cache_speech_token in cache_speech_token:
        speech_tokens.extend(a_cache_speech_token)
    llm_speech_token = torch.tensor([speech_tokens], dtype=torch.int32).to(device)
    return prompt_text_token, llm_speech_token

def generate_long(
    frontend,
    text_frontend,
    llm,
    flow,
    text_info,
    cache,
    device,
    embedding,
    seed=0,
    sample_method="ras",
    flow_prompt_token=None,
    speech_feat=None,
    use_phoneme=False,
):
    outputs = []
    full_mels = []
    output_token_list = []
    uttid = text_info[0]
    syn_text = text_info[1]
    text_tn_dict = {
        "uttid": uttid,
        "syn_text": syn_text,
        "syn_text_tn": [],
        "syn_text_phoneme": [],
    }
    short_text_list = text_frontend.split_by_len(syn_text)
    for _, tts_text in enumerate(short_text_list):
        set_seed(seed)
        tts_text_tn = text_frontend.text_normalize(tts_text)
        text_tn_dict["syn_text_tn"].append(tts_text_tn)
        if use_phoneme:
            tts_text_tn = text_frontend.g2p_infer(tts_text_tn)
            text_tn_dict["syn_text_phoneme"].append(tts_text_tn)
        tts_text_token = frontend._extract_text_token(tts_text_tn)
        cache_text = cache["cache_text"]
        cache_text_token = cache["cache_text_token"]
        cache_speech_token = cache["cache_speech_token"]
        if cache["use_cache"] and len(cache_text_token) > 1:
            prompt_text_token, prompt_speech_token = get_cached_prompt(cache, tts_text_token, device)
        else:
            prompt_text_token = cache_text_token[0].to(device)
            prompt_speech_token = torch.tensor([cache_speech_token[0]], dtype=torch.int32).to(device)
        token_list_res = local_llm_forward(
            llm=llm,
            prompt_text_token=prompt_text_token,
            tts_text_token=tts_text_token,
            prompt_speech_token=prompt_speech_token,
            sample_method=sample_method
        )
        output_token_list.extend(token_list_res)
        output, full_mel = local_flow_forward(
            flow=flow,
            token_list=token_list_res,
            prompt_speech_tokens=flow_prompt_token,
            speech_feat=speech_feat,
            embedding=embedding
        )
        if cache is not None:
            cache_text.append(tts_text_tn)
            cache_text_token.append(tts_text_token)
            cache_speech_token.append(token_list_res)
        outputs.append(output)
        if full_mel is not None:
            full_mels.append(full_mel)
    tts_speech = torch.concat(outputs, dim=1)
    tts_mel = torch.concat(full_mels, dim=-1) if full_mels else None
    return tts_speech, tts_mel, output_token_list, text_tn_dict

class GLMTTSWrapper:
    def __init__(self, frontend, text_frontend, speech_tokenizer, llm, flow, flow_config, sample_rate):
        self.frontend = frontend
        self.text_frontend = text_frontend
        self.speech_tokenizer = speech_tokenizer
        self.llm = llm
        self.flow = flow
        self.flow_config = flow_config # keeping config if needed
        self.sample_rate = sample_rate

class GLMTTSLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "GLM-TTS"}),
                "sample_rate": ([24000, 32000], {"default": 24000}),
                "use_phoneme": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("GLM_TTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "GLM-TTS"

    def load_model(self, model_path, sample_rate, use_phoneme):
        # Resolve model path
        # Priority 1: ComfyUI models directory
        # Priority 2: Absolute path
        # If none found, default to ComfyUI models directory for download
        
        base_path = None
        
        # 1. Check ComfyUI models dir
        models_dir_path = os.path.join(folder_paths.models_dir, model_path)
        if os.path.exists(models_dir_path):
            base_path = models_dir_path
        
        # 2. Check Absolute path
        if base_path is None and os.path.isabs(model_path) and os.path.exists(model_path):
            base_path = model_path
                
        # 3. If not found, default to ComfyUI models dir for download
        if base_path is None:
            base_path = models_dir_path

        # Check if models exist, if not, try to download
        required_files = ["llm/config.json", "flow/config.yaml"]
        missing = False
        if not os.path.exists(base_path):
             missing = True
        else:
             for f in required_files:
                 if not os.path.exists(os.path.join(base_path, f)):
                     missing = True
                     break
             if not missing:
                 speech_dir = os.path.join(base_path, "speech_tokenizer")
                 pattern = os.path.join(speech_dir, "model*.safetensors")
                 if not (os.path.isdir(speech_dir) and glob.glob(pattern)):
                     print(f"Speech tokenizer weights missing under {speech_dir}, will download GLM-TTS from HuggingFace.")
                     missing = True
        
        if missing:
            print(f"Models missing in {base_path}. Attempting to download...")
            try:
                from huggingface_hub import snapshot_download
                print("Downloading zai-org/GLM-TTS from HuggingFace...")
                # Download to the resolved base_path
                # If base_path didn't exist, we need to create it or let snapshot_download do it.
                # If base_path was inferred from missing plugin folder, use that.
                
                # If user provided "ckpt", and it doesn't exist anywhere, we default to plugin_dir/ckpt
                if not os.path.exists(base_path):
                     os.makedirs(base_path, exist_ok=True)
                
                snapshot_download(repo_id="zai-org/GLM-TTS", local_dir=base_path, local_dir_use_symlinks=False)
                print("Download complete.")
            except ImportError:
                raise ImportError("huggingface_hub not installed. Please install it or manually download models.")
            except Exception as e:
                raise RuntimeError(f"Failed to download models: {e}")

        if not os.path.exists(base_path):
             raise FileNotFoundError(f"Model path not found: {base_path}")

        print(f"Loading GLM-TTS models from: {base_path}")

        if base_path not in sys.path:
            sys.path.insert(0, base_path)
        try:
            glmtts_root = None
            for _p in glmtts_src_candidates:
                if os.path.isdir(_p):
                    glmtts_root = _p
                    break
            if glmtts_root is None:
                glmtts_root = base_path

            def _load_module(name, path):
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

            utils_init = os.path.join(glmtts_root, "utils", "__init__.py")
            if os.path.exists(utils_init):
                try:
                    import utils.glm_g2p  # type: ignore
                except Exception:
                    spec_utils = importlib.util.spec_from_file_location("utils", utils_init)
                    utils_mod = importlib.util.module_from_spec(spec_utils)
                    utils_mod.__path__ = [os.path.dirname(utils_init)]
                    spec_utils.loader.exec_module(utils_mod)
                    sys.modules["utils"] = utils_mod

            cosyvoice_pkg_dir = os.path.join(glmtts_root, "cosyvoice")
            cosyvoice_init = os.path.join(cosyvoice_pkg_dir, "__init__.py")
            if os.path.exists(cosyvoice_init):
                for k in list(sys.modules.keys()):
                    if k == "cosyvoice" or k.startswith("cosyvoice."):
                        sys.modules.pop(k, None)
                spec_cosy = importlib.util.spec_from_file_location("cosyvoice", cosyvoice_init)
                cosy_mod = importlib.util.module_from_spec(spec_cosy)
                cosy_mod.__path__ = [cosyvoice_pkg_dir]
                spec_cosy.loader.exec_module(cosy_mod)
                sys.modules["cosyvoice"] = cosy_mod

            cosy_frontend_mod = _load_module(
                "glmtts_cosyvoice_frontend",
                os.path.join(glmtts_root, "cosyvoice", "cli", "frontend.py"),
            )
            TTSFrontEnd = cosy_frontend_mod.TTSFrontEnd
            SpeechTokenizer = cosy_frontend_mod.SpeechTokenizer
            TextFrontEnd = cosy_frontend_mod.TextFrontEnd

            yaml_util_mod = _load_module(
                "glmtts_yaml_util",
                os.path.join(glmtts_root, "utils", "yaml_util.py"),
            )
            yaml_util = yaml_util_mod

            audio_mod = _load_module(
                "glmtts_audio_util",
                os.path.join(glmtts_root, "utils", "audio.py"),
            )
            mel_spectrogram = audio_mod.mel_spectrogram

            tts_model_util_mod = _load_module(
                "glmtts_tts_model_util",
                os.path.join(glmtts_root, "utils", "tts_model_util.py"),
            )
            tts_model_util = tts_model_util_mod
        except Exception as e:
            raise ImportError(f"Failed to import GLM-TTS modules from {base_path}: {e}")
        
        # 1. Load Speech Tokenizer
        speech_tokenizer_path = os.path.join(base_path, "speech_tokenizer")
        if not os.path.exists(speech_tokenizer_path):
            # Fallback for structure where ckpt is root
            speech_tokenizer_path = os.path.join(base_path, "..", "speech_tokenizer")
            
        # We assume standard structure inside `model_path` (which might be 'ckpt')
        # ckpt/speech_tokenizer
        # ckpt/vq32k-phoneme-tokenizer
        # ckpt/llm
        # ckpt/flow
        
        _model, _feature_extractor = yaml_util.load_speech_tokenizer(speech_tokenizer_path)
        speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)

        # 2. Load Frontends
        # load_frontends hardcodes 'ckpt/vq32k-phoneme-tokenizer' in glmtts_inference.py
        # We need to manually do what load_frontends does to control paths.
        
        if sample_rate == 32000:
             feat_extractor = partial(mel_spectrogram, sampling_rate=sample_rate, hop_size=640, n_fft=2560, num_mels=80, win_size=2560, fmin=0, fmax=8000, center=False)
        else:
             feat_extractor = partial(mel_spectrogram, sampling_rate=sample_rate, hop_size=480, n_fft=1920, num_mels=80, win_size=1920, fmin=0, fmax=8000, center=False)

        tokenizer_path = os.path.join(base_path, "vq32k-phoneme-tokenizer")
        glm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        tokenize_fn = lambda text: glm_tokenizer.encode(text)

        frontend_dir = os.path.join(base_path, "frontend")
        campplus_path = os.path.join(frontend_dir, "campplus.onnx")
        if not os.path.exists(campplus_path):
            fallback_candidates = [
                os.path.join(glmtts_root, "frontend", "campplus.onnx"),
                os.path.join(current_dir, "frontend", "campplus.onnx"),
            ]
            for _p in fallback_candidates:
                if os.path.exists(_p):
                    print(f"campplus.onnx not found under {frontend_dir}, using fallback: {_p}")
                    campplus_path = _p
                    break
        if not os.path.exists(campplus_path):
            raise FileNotFoundError(f"campplus.onnx not found in frontend dir or fallbacks: {campplus_path}")
        spk2info_path = os.path.join(frontend_dir, "spk2info.pt")
        if not os.path.exists(spk2info_path):
            os.makedirs(frontend_dir, exist_ok=True)
            try:
                import torch as _torch_for_spk2info
                _torch_for_spk2info.save({}, spk2info_path)
            except Exception:
                pass
        
        frontend = TTSFrontEnd(
            tokenize_fn,
            speech_tokenizer,
            feat_extractor,
            campplus_path,
            spk2info_path,
            DEVICE,
        )
        text_frontend = TextFrontEnd(use_phoneme)

        # 3. Load LLM
        llama_path = os.path.join(base_path, "llm")
        llm_model = LLMWrapper(
            llama_cfg_path=os.path.join(llama_path, "config.json"),
            llama_path=llama_path
        )

        special_token_ids = get_special_token_ids(frontend.tokenize_fn)
        llm_model.set_runtime_vars(special_token_ids=special_token_ids)

        # 4. Load Flow
        flow_path = os.path.join(base_path, "flow", "flow.pt")
        flow_config = os.path.join(base_path, "flow", "config.yaml")
        flow_model = yaml_util.load_flow_model(flow_path, flow_config, DEVICE)
        token2wav = tts_model_util.Token2Wav(flow_model, sample_rate=sample_rate, device=DEVICE)

        model_wrapper = GLMTTSWrapper(
            frontend=frontend,
            text_frontend=text_frontend,
            speech_tokenizer=speech_tokenizer,
            llm=llm_model,
            flow=token2wav,
            flow_config=flow_config,
            sample_rate=sample_rate
        )

        return (model_wrapper,)

class GLMTTSSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("GLM_TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "你好，这是一个测试。"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "use_cache": ("BOOLEAN", {"default": True}),
                "sample_method": (["ras", "greedy", "top_p"], {"default": "ras"}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "reference_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "GLM-TTS"

    def generate(self, model, text, seed, use_cache, sample_method, reference_audio=None, reference_text=""):
        set_seed(seed)
        
        frontend = model.frontend
        text_frontend = model.text_frontend
        llm = model.llm
        flow = model.flow
        sample_rate = model.sample_rate

        prompt_text = reference_text.strip() if isinstance(reference_text, str) else ""
        speech_feat = None
        prompt_speech_token = None
        embedding = None
        
        if reference_audio is not None:
            ref_wav = reference_audio["waveform"]
            ref_sr = reference_audio["sample_rate"]
            ref_wav_mono = ref_wav[0]
            if ref_wav_mono.shape[0] > 1:
                ref_wav_mono = torch.mean(ref_wav_mono, dim=0, keepdim=True)
            if ref_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=ref_sr, new_freq=sample_rate).to(ref_wav_mono.device)
                ref_wav_mono = resampler(ref_wav_mono)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            torchaudio.save(temp_path, ref_wav_mono, sample_rate)
            try:
                prompt_speech_token = frontend._extract_speech_token([temp_path])
                speech_feat = frontend._extract_speech_feat(temp_path, sample_rate=sample_rate)
                embedding = frontend._extract_spk_embedding(temp_path)
                if not prompt_text:
                    prompt_text = auto_transcribe_reference_audio(temp_path, language="zh")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            raise ValueError("Reference audio is required for GLM-TTS Zero-shot generation.")

        prompt_text_norm = text_frontend.text_normalize(prompt_text)
        synth_text_norm = text_frontend.text_normalize(text)
        
        prompt_text_token = frontend._extract_text_token(prompt_text_norm + " ")
        
        cache_speech_token = [prompt_speech_token.squeeze().tolist()]
        flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(DEVICE)
        
        cache = {
            "cache_text": [prompt_text_norm],
            "cache_text_token": [prompt_text_token],
            "cache_speech_token": cache_speech_token,
            "use_cache": use_cache,
        }
        
        tts_speech, tts_mel, output_token_list, text_tn_dict = generate_long(
            frontend=frontend,
            text_frontend=text_frontend,
            llm=llm,
            flow=flow,
            text_info=["generated", synth_text_norm], # dummy uttid
            cache=cache,
            device=DEVICE,
            embedding=embedding,
            seed=seed,
            sample_method=sample_method,
            flow_prompt_token=flow_prompt_token,
            speech_feat=speech_feat,
            use_phoneme=model.text_frontend.use_phoneme # verify this attr exists or check initialization
        )
        
        out_wav = tts_speech.unsqueeze(0).cpu() # [1, 1, samples]
        
        return ({"waveform": out_wav, "sample_rate": sample_rate},)

class GLMTTSASR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "language": (["auto", "zh", "en"], {"default": "zh"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = "GLM-TTS"

    def transcribe(self, audio, language):
        if audio is None:
            return ("" ,)
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        wav_mono = waveform[0]
        if wav_mono.shape[0] > 1:
            wav_mono = torch.mean(wav_mono, dim=0, keepdim=True)
        target_sr = 16000
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr).to(wav_mono.device)
            wav_mono = resampler(wav_mono)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        torchaudio.save(temp_path, wav_mono.cpu(), target_sr)
        try:
            text = auto_transcribe_reference_audio(temp_path, language=language)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        if not isinstance(text, str):
            text = ""
        return (text, )

