from .nodes import GLMTTSLoader, GLMTTSSampler

NODE_CLASS_MAPPINGS = {
    "GLMTTSLoader": GLMTTSLoader,
    "GLMTTSSampler": GLMTTSSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GLMTTSLoader": "GLM-TTS Loader",
    "GLMTTSSampler": "GLM-TTS Sampler"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
