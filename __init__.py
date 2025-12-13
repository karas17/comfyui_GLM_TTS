from .nodes import GLMTTSLoader, GLMTTSSampler, GLMTTSASR

NODE_CLASS_MAPPINGS = {
    "GLMTTSLoader": GLMTTSLoader,
    "GLMTTSSampler": GLMTTSSampler,
    "GLMTTSASR": GLMTTSASR
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GLMTTSLoader": "GLM-TTS Loader",
    "GLMTTSSampler": "GLM-TTS Sampler",
    "GLMTTSASR": "GLM-TTS ASR"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
