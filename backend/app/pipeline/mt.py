"""
MT Engine — Machine Translation using MarianMT (Helsinki-NLP).

Translates committed text segments.  Models are loaded lazily and cached
per language pair so we only load what's needed.

Supported MVP pairs: es→en, en→es.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Map of "src-tgt" → HuggingFace model id
DEFAULT_MODELS: dict[str, str] = {
    "es-en": "Helsinki-NLP/opus-mt-es-en",
    "en-es": "Helsinki-NLP/opus-mt-en-es",
}


class MTEngine:
    """MarianMT-based translation engine with lazy model loading."""

    def __init__(self, model_map: dict[str, str] | None = None, device: str = "cpu"):
        self.device = device
        self.model_map = model_map or DEFAULT_MODELS
        self._models: dict[str, tuple] = {}  # pair → (tokenizer, model)
        self._loaded_pairs: set[str] = set()

    def load_pair(self, src: str, tgt: str) -> None:
        """Pre-load a language pair."""
        pair = f"{src}-{tgt}"
        if pair in self._loaded_pairs:
            return
        model_id = self.model_map.get(pair)
        if not model_id:
            logger.warning(f"No MT model for pair {pair}")
            return

        try:
            from transformers import MarianTokenizer, MarianMTModel
            import torch

            logger.info(f"Loading MT model: {model_id}")
            tokenizer = MarianTokenizer.from_pretrained(model_id)
            model = MarianMTModel.from_pretrained(model_id)

            # Move to device
            effective_device = self.device
            if effective_device == "mps":
                try:
                    model = model.to("mps")
                except Exception:
                    logger.info("MPS not supported for MarianMT, using CPU")
                    effective_device = "cpu"
                    model = model.to("cpu")
            elif effective_device == "cuda":
                model = model.to("cuda")
            else:
                model = model.to("cpu")

            model.eval()
            self._models[pair] = (tokenizer, model, effective_device)
            self._loaded_pairs.add(pair)
            logger.info(f"MT model loaded: {pair} on {effective_device}")
        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers sentencepiece")
            raise

    async def translate(self, text: str, src: str, tgt: str) -> str:
        """Translate text from src to tgt language. Returns translated string."""
        if not text or not text.strip():
            return ""

        pair = f"{src}-{tgt}"
        if pair not in self._models:
            self.load_pair(src, tgt)
        if pair not in self._models:
            logger.warning(f"MT pair {pair} not available, returning original")
            return text

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._translate_sync, text, pair
        )

    def _translate_sync(self, text: str, pair: str) -> str:
        """Synchronous translation."""
        import torch

        tokenizer, model, device = self._models[pair]
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_length=512, num_beams=4)
            result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return result.strip()
        except Exception as e:
            logger.error(f"MT error: {e}")
            return text
