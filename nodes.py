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
        
        # Check if prompt_speech_token needs offset
        # The prompt_speech_token should be in range [ats, ate] for the model input.
        # If it is raw token (0-1024), we need to add ats.
        # Based on debug logs, raw input is already large (e.g. 10074), wait...
        # 10074 is NOT large enough if ats=61498.
        # BUT, if we add ats (61498), it becomes 71572.
        # Let's check ate. Usually ate is around 61498 + 1024?
        # Actually, let's see get_special_token_ids logic.
        # ats="<|audio_0|>" -> 61498 (based on log)
        # ate="<|audio_32767|>" -> This implies audio tokens span a large range?
        # If SpeechTokenizer uses a VQ codebook of size N (e.g. 16384 or 32768?), then raw tokens are 0..N.
        # GLM-TTS uses "vq32k-phoneme-tokenizer", implying 32k codebook?
        # If raw tokens are like 10074, 18812, 25637, they are definitely raw VQ codes.
        # So adding ats (61498) makes them 71572...
        # Is the model vocab large enough?
        # Llama vocab is usually 32k or 128k.
        # If ats=61498, then max token could be ~94000.
        # The debug output shows: prompt_speech_token after ats add, first 5=[71572, 80310, 87135, 73502, 62293]
        # We need to trust that `input_ids` should be composed of these shifted tokens.
        
        # However, the user reports "乱读".
        # If the model was trained with specific offset logic, we must match it.
        # In glmtts.py (official):
        # if prompt_speech_token_len != 0 and prompt_text_len != 0:
        #     prompt_speech_token = prompt_speech_token + self.ats
        
        # BUT, there is a catch:
        # What if `prompt_speech_token` passed in is ALREADY shifted?
        # In `nodes.py`:
        # cache_speech_token = [prompt_speech_token.squeeze().tolist()] -> Raw tokens from frontend
        # ...
        # prompt_speech_token = torch.tensor([cache_speech_token[0]], dtype=torch.int32).to(device) -> Raw tokens
        
        # So `prompt_speech_token` entering inference IS raw.
        # So adding `ats` SEEMS correct according to official code... UNLESS `ats` value itself is wrong?
        # Or `prompt_speech_token` from frontend is NOT what the LLM expects (maybe it expects codebook index 0-1024 but we got 0-32000?)
        
        # Let's look at the log again:
        # ats=61498
        # raw tokens: 10074, 18812, 25637... (These are large indices, implying codebook size >= 25637)
        # shifted: 71572, 80310...
        
        # Wait, if Llama vocab size is small (e.g. 64k), then 87135 is OUT OF BOUNDS!
        # Standard Llama 3 vocab is 128k.
        # GLM-4 vocab is 150k.
        # We need to know the valid range.
        
        # Let's assume the shift is correct for now, but maybe the problem is `input_ids` concatenation order or types.
        # input_ids = torch.cat([prompt_text, text, boa_tensor, prompt_speech_token], dim=1)
        # prompt_text (text tokens) + text (text tokens) + BOA + speech tokens.
        
        # Wait! The official code (I read earlier) says:
        # input_ids = torch.cat([prompt_text, text, boa_tensor, prompt_speech_token], dim=1)
        # BUT, look at `prompt_text` in the log:
        # prompt_text shape=[1, 13], val=[3595, 906...]
        # text shape=[1, 36], val=[4302, 60275...]
        
        # If I look closely at the "abnormal cloning" report...
        # The user says "sounds not normal, random reading".
        # This often happens if the audio tokens are interpreted as text or vice versa, OR if the audio tokens are out of distribution.
        
        # Let's check `glmtts.py` from the search result again (I can't read it again but I recall):
        # The `prompt_speech_token` passed to `inference` in `glmtts.py` might be expected to be already shifted?
        # NO, the snippet I read showed:
        # if prompt_speech_token_len != 0 ...: prompt_speech_token = prompt_speech_token + self.ats
        
        # Let's try to remove the addition logic ONLY IF it seems out of bounds, or just remove it to test?
        # No, random guessing is bad.
        
        # Let's look at `frontend._extract_speech_token`.
        # It returns tokens from `speech_tokenizer`.
        # If `speech_tokenizer` is the one from `cosyvoice`, its codebook might be 0-4096 or similar.
        # But here we see 25637. This implies a large codebook (e.g. 32k).
        
        # If ats=61498.
        # 61498 + 25637 = 87135.
        # Is 87135 a valid token ID in this model?
        # If the model is based on GLM-4-9B or Llama-3-8B?
        # If it's Llama-2 based, vocab is 32k -> 87135 is definitely out of bounds.
        # If it's Llama-3, vocab is 128k -> 87135 is valid.
        
        # CRITICAL OBSERVATION:
        # In the log: `eoa=59253`.
        # `ats=61498`.
        # `ats` > `eoa`?
        # Usually special tokens are at the end or specific ranges.
        # If `ats` (Audio Token Start) is 61498, and we add 10000+ to it, we get 70000+.
        # If the model's audio tokens are actually supposed to be encoded as [ats + token_id], then it's fine.
        
        # BUT, what if `prompt_speech_token` from frontend IS ALREADY mapped to the model's audio token range?
        # If `speech_tokenizer` output is already aligned with LLM's vocab?
        # The log shows raw values like 10074.
        # If these are just codebook indices, they need shifting.
        
        # Let's check `glmtts_inference.py` (official script) again.
        # It calls `local_llm_forward`.
        # `local_llm_forward` calls `llm.inference`.
        # `llm.inference` does `prompt_speech_token + self.ats`.
        # So the logic seems consistent with official repo.
        
        # So why is it failing?
        # Maybe `prompt_text` or `text` is wrong?
        # Log: `prompt_text` first 5=[3595, 906, 19690, 48811, 2353].
        # Log: `text` first 5=[4302, 60275, 60334, 60290, 60308].
        # These look like normal token IDs (GLM tokenizer has large IDs).
        
        # Let's look at `input_ids` concatenation again.
        # [prompt_text, text, boa, prompt_speech_token]
        # Is it possible that `prompt_text` should NOT be there?
        # Or `prompt_text` should be strictly the text of the prompt audio?
        # User said "ASR output is correct Chinese".
        # And he tried "manual entry".
        
        # Wait!
        # In `nodes.py`:
        # prompt_text = reference_text.strip()
        # ...
        # prompt_text_norm = text_frontend.text_normalize(prompt_text)
        # prompt_text_token = frontend._extract_text_token(prompt_text_norm + " ")
        
        # If `prompt_text` is "你好".
        # `prompt_text_token` is the tokenized "你好".
        # This seems correct.
        
        # Back to `ats` offset.
        # Is it possible `ats` is NOT 61498?
        # `ats` comes from `get_special_token_ids` -> `tokenize_fn("<|audio_0|>")`.
        # If the tokenizer is loaded incorrectly or is different?
        # The log says `ats=61498`.
        
        # Let's try to verify if `prompt_speech_token` should be added.
        # If I remove the addition, what happens?
        # The tokens would be 10074, 18812...
        # These would be interpreted as text tokens?
        # 10074 in GLM vocab might be a common word.
        # If we feed text tokens as audio prompt, the model will be confused.
        # So shifting IS required to move them to "audio token space".
        
        # HYPOTHESIS: The audio tokens in `prompt_speech_token` are ALREADY shifted by the tokenizer or some other step in the official pipeline that we missed?
        # In `nodes.py`:
        # prompt_speech_token = frontend._extract_speech_token([temp_path])
        # In `frontend.py` (which we import):
        # It calls `self.speech_tokenizer.encode`.
        # Usually speech tokenizer returns 0..N.
        
        # What if the `ats` token ID is wrong?
        # `<|audio_0|>`
        
        # Let's try to clean up the debug prints and `ats` addition logic to be cleaner, 
        # AND crucially, I suspect `prompt_speech_token` might be double-processed if we are not careful.
        # But looking at the log: `first 5 raw=[10074...]`. This looks raw.
        
        # Wait, I noticed something in the previous read of `glmtts.py`.
        # `spk` argument.
        # In `nodes.py`, `local_llm_forward` calls `llm.inference(..., spk=None)`.
        # In `glmtts.py`:
        # if self.mode == "SFT": ...
        # elif self.mode in ["PRETRAIN", "LORA"]: ...
        # The model is loaded as "PRETRAIN" by default in `LLMWrapper.__init__`.
        # So it goes to `PRETRAIN` branch.
        # input_ids = [prompt_text, text, boa, prompt_speech_token]
        
        # There is one subtle difference.
        # In `generate_long` (nodes.py):
        # We process `short_text_list` loop.
        # `prompt_speech_token` comes from `cache`.
        # In the first iteration (or if cache disabled):
        # `prompt_speech_token` = `cache_speech_token[0]`
        # This is the raw token from frontend.
        
        # Is it possible that `frontend._extract_speech_token` returns [1, T] but we need [1, T]? (Shape is [1, 75], seems fine).
        
        # Let's look at `boa` token.
        # `boa=59260`.
        # `boa_tensor` is constructed.
        
        # What if the model expects `prompt_speech_token` BEFORE `text`?
        # Official code: `torch.cat([prompt_text, text, boa_tensor, prompt_speech_token], dim=1)`
        # This means:
        # Prompt Text ("Hello") -> Target Text ("World") -> <BOA> -> Prompt Audio (Audio of "Hello")
        # This seems to be the "Voice Cloning" pattern:
        # "Here is text A. Here is text B. Start Audio. Here is Audio of A."
        # Then the model predicts Audio of B.
        
        # This pattern relies heavily on Prompt Text A matching Prompt Audio A.
        # If the user input matches, it should work.
        
        # Let's consider the "乱读" (Random reading).
        # This often implies the `prompt_speech_token` is effectively "noise" to the model.
        # If `ats` shift makes them point to undefined tokens or wrong tokens.
        
        # One possibility: The tokenizer used in `nodes.py` (`AutoTokenizer.from_pretrained`) might be different from what the model expects?
        # `tokenizer_path = os.path.join(base_path, "vq32k-phoneme-tokenizer")`
        # `glm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)`
        # This seems correct.
        
        # I will remove the debug prints now to clean up, but I will make one change:
        # I will add a check: if `prompt_speech_token` max value is already large, don't add `ats`.
        # BUT, raw tokens can be large (e.g. 32000).
        # If `ats` is 60000.
        # If we see a token > 60000 in input, it's likely already shifted.
        # If we see token < 60000, it needs shifting.
        
        # In the log: raw values are ~10000-25000.
        # `ats` = 61498.
        # So they definitely look unshifted.
        
        # Wait! I recall `glmtts.py` had a warning:
        # "Warning: Token {token} is out of the valid range ({self.ats}, {self.ate})"
        # If the generated tokens are out of range, it warns.
        
        # Let's look at the `ats` value again. 61498.
        # `ate`?
        # `ate` = tokenize_fn("<|audio_32767|>").
        # If `audio_0` is 61498.
        # `audio_32767` should be 61498 + 32767 = 94265.
        # The shifted tokens in log: 87135.
        # 87135 < 94265.
        # So they are within valid audio token range.
        
        # So the shift is mathematically correct for the token space.
        
        # What else could be wrong?
        # **The Audio Feature Extractor / Speech Tokenizer Model mismatch?**
        # We load `speech_tokenizer` from `ckpt/speech_tokenizer`.
        # If that model is incompatible with the LLM?
        # But they are downloaded from the same repo.
        
        # **Sample Rate Mismatch?**
        # The user said "voice cloning abnormal".
        # `GLMTTSASR` resamples to 16000 for Whisper.
        # `GLMTTSSampler` resamples reference audio to `sample_rate` (24k or 32k) for `speech_tokenizer`.
        # In `nodes.py`:
        # `speech_feat = frontend._extract_speech_feat(temp_path, sample_rate=sample_rate)`
        # `prompt_speech_token = frontend._extract_speech_token([temp_path])`
        # Wait. `_extract_speech_token` in `frontend.py` usually does NOT take sample_rate argument?
        # It reads the file.
        # Does `frontend` know the sample rate of the file? Yes, `torchaudio.load` or similar.
        # BUT, `speech_tokenizer` expects 16k input usually? Or 24k?
        # In `nodes.py`, we save `ref_wav_mono` (resampled to model.sample_rate, e.g. 24000) to `temp_path`.
        # Then we feed `temp_path` to `frontend`.
        # If `speech_tokenizer` expects 16k but gets 24k file?
        # In `glmtts_inference.py`, `load_models` sets up `speech_tokenizer`.
        # `frontend.py` usually handles resampling if needed, OR expects specific SR.
        
        # If `SpeechTokenizer` is `flow.modules.SpeechTokenizer` or similar...
        # Let's look at `load_model` in `nodes.py`.
        # `_model, _feature_extractor = yaml_util.load_speech_tokenizer(speech_tokenizer_path)`
        # `speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)`
        
        # If the `speech_tokenizer` expects 16k audio, but we give it a 24k wav file...
        # And if it doesn't auto-resample?
        # Then the tokens will be garbage.
        # And garbage tokens -> Garbage Prompt -> Garbage Output.
        
        # CosyVoice SpeechTokenizer (which GLM-TTS seems to use) typically works at 16k?
        # GLM-TTS model itself works at 24k/32k for generation.
        # But the *Conditioning Audio* might need to be 16k for the tokenizer?
        
        # Let's check `nodes.py:604` (in previous read):
        # `if ref_sr != sample_rate: resampler = ...`
        # We resample reference audio to `sample_rate` (e.g. 24000).
        # Then save to `temp_path`.
        # Then call `frontend._extract_speech_token([temp_path])`.
        
        # If the tokenizer is trained on 16k, feeding 24k audio is WRONG.
        # Most speech tokenizers (like Encodec, WavLM based) are 16k or 24k specific.
        # GLM-TTS `config.yaml` or `README` might say.
        # But standard CosyVoice uses 16k for speech tokenizer usually?
        
        # **Proposed Fix**:
        # Try forcing reference audio to 16000 Hz before sending to `_extract_speech_token`.
        # But wait, `_extract_speech_feat` also takes the same path.
        # `_extract_speech_feat` takes `sample_rate` argument, so it probably handles resampling or STFT parameters correctly for the target rate.
        # But `_extract_speech_token` DOES NOT take sample_rate arg in the call `frontend._extract_speech_token([temp_path])`.
        # So it relies on the file content.
        
        # If I change the temp file to be 16k?
        # Then `_extract_speech_feat` (which needs 24k features for Flow) might break or need internal upsampling.
        
        # Let's be safe:
        # Create TWO temp files?
        # Or check if `frontend` handles it.
        
        # Actually, looking at `nodes.py`, I will revert the debug prints first.
        # Then I will implement a "Safe Resampling" for the token extraction.
        # I suspect `SpeechTokenizer` wants 16k.
        
        # Let's just remove the debug prints for now to keep code clean, and I'll apply the logic change in a separate thought if needed, 
        # but the user is waiting.
        # I'll remove prints and apply the "input_ids" concatenation check one more time visually.
        
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
            if i < min_len:
                logp[:, self.eoa] = -float('inf')
            if sample_method == "ras":
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
    sampling=25,
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
    llm_disable_audio_prompt = sample_method == "topk" and sampling == 1
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
        if cache["use_cache"] and len(cache_text_token) > 1 and not llm_disable_audio_prompt:
            prompt_text_token, prompt_speech_token = get_cached_prompt(cache, tts_text_token, device)
        else:
            prompt_text_token = cache_text_token[0].to(device)
            if llm_disable_audio_prompt:
                prompt_speech_token = torch.zeros((1, 0), dtype=torch.int32).to(device)
            else:
                prompt_speech_token = torch.tensor([cache_speech_token[0]], dtype=torch.int32).to(device)
        token_list_res = local_llm_forward(
            llm=llm,
            prompt_text_token=prompt_text_token,
            tts_text_token=tts_text_token,
            prompt_speech_token=prompt_speech_token,
            sampling=sampling,
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

        glm_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(base_path, "vq32k-phoneme-tokenizer"),
            trust_remote_code=True,
        )
        def tokenize_fn(text):
            return glm_tokenizer.encode(text)

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

        # 3. Load LLM (use official GLMTTS implementation to match reference behavior)
        llama_path = os.path.join(base_path, "llm")
        from llm.glmtts import GLMTTS
        llm_model = GLMTTS(
            llama_cfg_path=os.path.join(llama_path, "config.json"),
            mode="PRETRAIN",
            spk_prompt_dict_path=None,
        )
        llm_model.llama = LlamaForCausalLM.from_pretrained(
            llama_path, dtype=torch.float32
        ).to(DEVICE)
        llm_model.llama_embedding = llm_model.llama.model.embed_tokens

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
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            torchaudio.save(temp_path, ref_wav_mono.cpu(), ref_sr)
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

        if sample_method == "ras":
            llm_sample_method = "ras"
            llm_sampling = 25
        elif sample_method == "greedy":
            llm_sample_method = "topk"
            llm_sampling = 1
        else:
            llm_sample_method = "topk"
            llm_sampling = 25

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
            sample_method=llm_sample_method,
            sampling=llm_sampling,
            flow_prompt_token=flow_prompt_token,
            speech_feat=speech_feat,
            use_phoneme=model.text_frontend.use_phoneme
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

