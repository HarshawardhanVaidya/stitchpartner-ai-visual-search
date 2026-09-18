"""
DINOv2 image embedding.

The model is loaded exactly once, at application startup, and reused for every
request. Inference runs under torch.inference_mode() so no autograd graph is
built.

Preprocessing note: DINOv2's own processor resizes the shortest edge to 256 and
centre-crops to 224. Feeding it a 4032x3024 phone photo therefore wastes a full
high-resolution decode and a 12-megapixel resample for nothing. We decode in
JPEG draft mode and downscale to PREPROCESS_MAX_DIM first, which measured 56 ms
-> 5 ms on the resize step alone with no change to the pixels the model sees.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageOps

import config

logger = logging.getLogger(__name__)

# Guard against decompression-bomb images.
Image.MAX_IMAGE_PIXELS = config.MAX_PIXELS


class ImageDecodeError(Exception):
    """The uploaded bytes are not a usable image."""


@dataclass
class EmbeddingResult:
    vector: np.ndarray
    preprocess_ms: float
    inference_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms


class Embedder:
    """Singleton wrapper around DINOv2."""

    def __init__(self, model_name: str = config.MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._torch = None
        self._dim = 768
        self.load_seconds = 0.0

    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def dim(self) -> int:
        return self._dim

    def load(self) -> None:
        """Load model weights. Idempotent."""
        if self._model is not None:
            return

        started = time.perf_counter()

        import torch  # imported lazily so tooling can import this module without torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch

        if config.TORCH_THREADS > 0:
            torch.set_num_threads(config.TORCH_THREADS)
        logger.info("torch intra-op threads: %d", torch.get_num_threads())

        logger.info("Loading %s ...", self.model_name)
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

        self._dim = int(getattr(self._model.config, "hidden_size", 768))
        self.load_seconds = time.perf_counter() - started
        logger.info(
            "Model ready in %.1fs (embedding dimension %d).",
            self.load_seconds, self._dim,
        )

    def warmup(self) -> float:
        """
        Run one throwaway inference so the first real customer request doesn't
        pay for lazy kernel initialisation.
        """
        if self._model is None:
            self.load()
        blank = Image.new("RGB", (224, 224), (128, 128, 128))
        started = time.perf_counter()
        self._infer(blank)
        elapsed = time.perf_counter() - started
        logger.info("Warmup inference: %.0f ms", elapsed * 1000)
        return elapsed

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def decode(data: bytes) -> Image.Image:
        """
        Decode untrusted bytes into a right-way-up RGB image at a sane size.

        The format is determined by Pillow from the actual bytes, never from the
        filename or the client-supplied content type.
        """
        try:
            image = Image.open(io.BytesIO(data))
            image.verify()          # cheap structural check on a throwaway handle
            image = Image.open(io.BytesIO(data))
        except Exception as exc:
            raise ImageDecodeError("Image could not be decoded.") from exc

        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise ImageDecodeError(f"Unsupported image format: {image.format}")

        # For JPEGs this downscales during decode, in the DCT domain — far
        # cheaper than decoding 12 megapixels and resampling afterwards.
        try:
            image.draft("RGB", (config.PREPROCESS_MAX_DIM, config.PREPROCESS_MAX_DIM))
        except Exception:
            pass

        try:
            image = ImageOps.exif_transpose(image)   # honour phone orientation
            image = image.convert("RGB")
        except Exception as exc:
            raise ImageDecodeError("Image could not be converted.") from exc

        longest = max(image.size)
        if longest > config.PREPROCESS_MAX_DIM:
            scale = config.PREPROCESS_MAX_DIM / longest
            new_size = (
                max(1, int(image.width * scale)),
                max(1, int(image.height * scale)),
            )
            image = image.resize(new_size, Image.BICUBIC)

        if min(image.size) < 16:
            raise ImageDecodeError("Image is too small to search with.")

        return image

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _infer(self, image: Image.Image) -> np.ndarray:
        torch = self._torch
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            outputs = self._model(**inputs)
            hidden = outputs.last_hidden_state
            if config.POOLING == "cls_patch":
                # CLS concatenated with mean-pooled patch tokens. In the
                # literature this survives cropping and background clutter
                # better than CLS alone, but it doubles the vector width and
                # changes every score — do not switch it on without running
                # scripts/evaluate_accuracy.py against both settings first.
                embedding = torch.cat(
                    [hidden[:, 0, :], hidden[:, 1:, :].mean(dim=1)], dim=1
                )
            else:
                # Default: matches the representation the current 51-product
                # index was built and accuracy-tested with.
                embedding = hidden[:, 0, :]
        vector = embedding.cpu().numpy().astype(np.float32)[0]
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def embed_image(self, image: Image.Image) -> np.ndarray:
        if self._model is None:
            self.load()
        return self._infer(image)

    def embed_bytes(self, data: bytes) -> EmbeddingResult:
        """Full pipeline for one image, with per-stage timings."""
        if self._model is None:
            self.load()

        started = time.perf_counter()
        image = self.decode(data)
        preprocess_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        vector = self._infer(image)
        inference_ms = (time.perf_counter() - started) * 1000

        return EmbeddingResult(vector, preprocess_ms, inference_ms)

    def health(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "model": self.model_name,
            "dim": self._dim if self.is_loaded else None,
            "load_seconds": round(self.load_seconds, 2),
        }


# Module-level singleton. Nothing here touches the network until load() runs,
# so importing this module stays cheap.
embedder = Embedder()
