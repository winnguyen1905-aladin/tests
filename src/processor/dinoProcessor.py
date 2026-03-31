#!/usr/bin/env python3
"""
Dino Processor - Global Feature Extraction using DINOv3

Extracts global feature vectors from images using Facebook's DINOv3 model.
Used for coarse similarity search in vector database (Milvus).
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import os
import logging
import gc
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

logger = logging.getLogger(__name__)

# Global CUDA stream for async inference
_cuda_stream: Optional[torch.cuda.Stream] = None

def get_cuda_stream() -> torch.cuda.Stream:
    """Get or create a CUDA stream for async inference."""
    global _cuda_stream
    if _cuda_stream is None and torch.cuda.is_available():
        _cuda_stream = torch.cuda.Stream()
    return _cuda_stream


def _dino_defaults():
    """Pull defaults from appConfig so DinoConfig is never out of sync."""
    from src.config.appConfig import get_config
    cfg = get_config()
    return {
        "model_type": cfg.dino_model_type,
        "device": cfg.dino_device,
        "image_size": cfg.dino_image_size,
        "hf_token": cfg.dino_hf_token,
        "use_multi_gpu": cfg.dino_use_multi_gpu,
        "gpu_ids": cfg.dino_gpu_ids,
    }


@dataclass
class DinoConfig:
    """Configuration for DINO processor. Defaults are sourced from appConfig."""
    model_type: str = field(default_factory=lambda: _dino_defaults()["model_type"])
    device: str = field(default_factory=lambda: _dino_defaults()["device"])
    image_size: int = field(default_factory=lambda: _dino_defaults()["image_size"])
    hf_token: str = field(default_factory=lambda: _dino_defaults()["hf_token"])
    use_multi_gpu: bool = field(default_factory=lambda: _dino_defaults()["use_multi_gpu"])
    gpu_ids: Optional[list] = field(default_factory=lambda: _dino_defaults()["gpu_ids"])
    normalize: bool = True
    verbose: bool = False
    enable_memory_optimization: bool = False
    use_4bit_quantization: bool = False
    use_gradient_checkpointing: bool = False
    per_gpu_batch_size: int = 1


@dataclass
class DinoResult:
    """Result from DINO feature extraction."""
    global_descriptor: np.ndarray
    image_size: tuple  # (H, W) of input image
    model_name: str

    @property
    def vector(self) -> np.ndarray:
        """Alias for global_descriptor for compatibility."""
        return self.global_descriptor


class DinoProcessor:
    """Processor for DINO feature extraction."""

    # raw_dim = what the model actually outputs (pooler_output).
    SUPPORTED_MODELS = {
        "dinov3-vith16plus": ("facebook/dinov3-vith16plus-pretrain-lvd1689m", 2048),
        "dinov3-vitl16": ("facebook/dinov3-vitl16-pretrain-lvd1689m", 1024),
        "dinov3-vitg14": ("facebook/dinov3-vitg14-pretrain-lvd1689m", 1536),
        "dinov2-vits14": ("facebook/dinov2-vits14", 384),
        "dinov2-vitb14": ("facebook/dinov2-vitb14", 768),
        "dinov2-vitl14": ("facebook/dinov2-vitl14", 1024),
        "dinov2-giant": ("facebook/dinov2-giant", 1536),
        "dinov3-vit7b16": ("facebook/dinov3-vit7b16-pretrain-lvd1689m", 4096),
        # dinov3-vitb16: ViT-Base 21M, outputs 384 dims (matches halfvec(384))
        "dinov3-vitb16-pretrain-lvd1689m": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 384),
        # dinov3-convnext-small: outputs 768, truncated to 384 for halfvec(384) storage
        "dinov3-convnext-small-pretrain-lvd1689m": ("facebook/dinov3-convnext-small-pretrain-lvd1689m", 768)
    }

    def __init__(self, config: Optional[DinoConfig] = None) -> None:
        """Initialize DINO processor.
        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or DinoConfig()
        self.model = None
        self.processor = None
        self.transform = None

    def _init_model(self) -> None:
        """Initialize DINO model and transforms with optional multi-GPU support."""
        if self.config.verbose:
            print(f"Initializing DINO model: {self.config.model_type}")
            print(f"Device: {self.config.device}")
            print(f"Multi-GPU: {self.config.use_multi_gpu}")

        # Get model info
        if self.config.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {self.config.model_type}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        model_name, feat_dim = self.SUPPORTED_MODELS[self.config.model_type]
        # Strip so .env values with accidental whitespace still authenticate
        hf_token = (self.config.hf_token or "").strip()
        if "facebook/dinov3" in model_name and "pretrain" in model_name and not hf_token:
            raise ValueError(
                "DINOv3 checkpoints are gated on Hugging Face. Set DINO_HF_TOKEN or HF_TOKEN "
                "(read access) and accept the model license while logged in:\n"
                f"  https://huggingface.co/{model_name}"
            )

        # Enable memory optimization if requested
        if self.config.enable_memory_optimization:
            # Check if using AMD ROCm (expandable_segments not supported)
            is_rocm = False
            try:
                is_rocm = getattr(torch.version, 'hip', None) is not None
            except Exception:
                pass

            if is_rocm:
                alloc_conf = "max_split_size_mb:256"
            else:
                alloc_conf = "max_split_size_mb:256,expandable_segments:True"

            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = alloc_conf
            if self.config.verbose:
                print(f"Memory optimization enabled: {alloc_conf}")

        # Enable all SDPA backends for maximum GPU throughput
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
                torch.backends.cuda.enable_cudnn_sdp(True)
        except Exception:
            pass

        # Load model from HuggingFace
        try:
            from transformers import AutoModel, AutoImageProcessor

            # Set token for gated models (using 'token' instead of deprecated 'use_auth_token')
            kwargs = {}
            if hf_token:
                kwargs["token"] = hf_token
                if self.config.verbose:
                    print("Using HuggingFace token for gated model")

            # Use bfloat16 for faster inference on modern GPUs
            if self.config.device == "cuda":
                kwargs["torch_dtype"] = torch.bfloat16

            # Enable memory optimization features
            if self.config.use_4bit_quantization:
                if self.config.verbose:
                    print("Enabling 4-bit quantization for memory efficiency")
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                kwargs["quantization_config"] = bnb_config

            # Enable gradient checkpointing for memory efficiency
            if self.config.use_gradient_checkpointing:
                if self.config.verbose:
                    print("Enabling gradient checkpointing")

            # Load image processor with auth token (only needed for gated repos)
            processor_kwargs = {}
            if hf_token:
                processor_kwargs["token"] = hf_token
                if self.config.verbose:
                    print(f"Using auth token for processor: {model_name}")
            if self.config.device == "cuda":
                processor_kwargs["torch_dtype"] = torch.bfloat16
            try:
                self.processor = AutoImageProcessor.from_pretrained(
                    model_name, **processor_kwargs
                )
            except OSError as e:
                if "gated" in str(e).lower() or "401" in str(e):
                    raise OSError(
                        f"{e}\n\n"
                        "Common fixes: (1) Use an account that clicked 'Agree and access repository' on the "
                        f"model page https://huggingface.co/{model_name} — a token alone is not enough. "
                        "(2) Ensure DINO_HF_TOKEN or HF_TOKEN is set (not overridden by an empty .env line). "
                        "(3) Token needs Read permission."
                    ) from e
                raise

            # Load model - use_auth_token is not valid for DINOv3 model
            # DINO models are not shardable across GPUs, so just load on single device
            if self.config.verbose and self.config.use_multi_gpu and torch.cuda.device_count() > 1:
                print(f"Note: DINO model does not support device sharding. Loading on single GPU.")
                print(f"Available GPUs: {torch.cuda.device_count()}")

            # Standard loading for DINO - it handles device placement internally
            model_kwargs = {}
            if hf_token:
                model_kwargs["token"] = hf_token
                if self.config.verbose:
                    print(f"Using auth token for model: {model_name}")
            if self.config.device == "cuda" and torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.bfloat16
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
            # Guard: only move to CUDA if configured and available
            _target = self.config.device
            if _target == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU for DINO")
                _target = "cpu"
            self.model.to(_target)

            if self.config.verbose:
                print(f"DINO model loaded: {model_name}")
                print(f"Feature dimension: {feat_dim}")

                # Print memory stats
                if torch.cuda.is_available():
                    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    print(f"GPU 0 memory: {total_mem:.2f} GB")

        except ImportError as e:
            raise ImportError(
                "transformers is required for DINO. "
                "Install with: pip install transformers"
            ) from e

        if self.config.verbose:
            print("DINO model initialized successfully!")

    def _descriptor_to_storage_dim(self, descriptor: np.ndarray) -> np.ndarray:
        """Trim descriptor to ``postgres_vector_dim`` when the backbone emits more dims.

        Keeps ``halfvec(384)`` storage aligned with config when a checkpoint returns
        a wider vector (e.g. 768). Re-L2-normalizes after truncation if ``normalize``.
        """
        vec = np.asarray(descriptor, dtype=np.float32).reshape(-1)
        try:
            from src.config.appConfig import get_config
            target = int(get_config().postgres_vector_dim)
        except Exception:
            return vec
        if target < 1 or vec.shape[0] <= target:
            return vec
        out = vec[:target].copy()
        if self.config.normalize:
            nrm = float(np.linalg.norm(out))
            if nrm > 1e-12:
                out = out / nrm
        logger.debug(
            "DINO descriptor truncated %d -> %d for postgres_vector_dim",
            vec.shape[0],
            target,
        )
        return out

    def extract(self, image: np.ndarray) -> Optional[DinoResult]:
        """Extract global feature vector from image.

        Args:
            image: Input image in BGR format (H, W, 3)

        Returns:
            DinoResult containing global descriptor vector, or None if extraction fails
        """
        if self.model is None:
            self._init_model()

        original_size = image.shape[:2]

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            # Resolve effective device: fall back to CPU if CUDA not available
            _target = self.config.device
            if _target == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU for DINO extract")
                _target = "cpu"

            # Process image using the processor
            # Handle both image processor with and without resize
            try:
                inputs = self.processor(images=image_rgb, return_tensors="pt")
            except Exception as e:
                if self.config.verbose:
                    print(f"Processor error, using manual preprocessing: {e}")
                from PIL import Image
                pil_img = Image.fromarray(image_rgb)
                inputs = self.processor(pil_img, return_tensors="pt")

            # Use CUDA streams for async data transfer if GPU is available (disabled on ROCm)
            is_rocm = False
            try:
                is_rocm = getattr(torch.version, 'hip', None) is not None
            except Exception:
                pass

            use_cuda_stream = _target == "cuda" and torch.cuda.is_available() and not is_rocm

            if use_cuda_stream:
                cuda_stream = get_cuda_stream()
                # Use non_blocking for async CPU->GPU transfer
                inputs = {k: v.to(_target, non_blocking=True) for k, v in inputs.items()}

                # Run inference in default stream, but capture output
                with torch.cuda.stream(cuda_stream):
                    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        outputs = self.model(**inputs)

                        # DINOv3 has pooler_output, use it directly
                        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                            features = outputs.pooler_output
                        else:
                            # Fallback: use mean of last hidden state (exclude CLS token)
                            last_hidden = outputs.last_hidden_state
                            if last_hidden.dim() == 3:  # (batch, seq, hidden)
                                # Mean over sequence dimension
                                features = last_hidden.mean(dim=1)
                            else:
                                features = last_hidden

                    # Sync and copy to CPU
                    torch.cuda.synchronize()
                    descriptor = features.squeeze().cpu().numpy()
            else:
                # Standard CPU/GPU path — always use resolved _target device
                inputs = {k: v.to(_target) for k, v in inputs.items()}

                # Use autocast only when on CUDA; on CPU run in fp32
                _use_autocast = _target == "cuda"
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=_use_autocast):
                    outputs = self.model(**inputs)

                    # DINOv3 has pooler_output, use it directly
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        features = outputs.pooler_output
                    else:
                        # Fallback: use mean of last hidden state (exclude CLS token)
                        last_hidden = outputs.last_hidden_state
                        if last_hidden.dim() == 3:  # (batch, seq, hidden)
                            # Mean over sequence dimension
                            features = last_hidden.mean(dim=1)
                        else:
                            features = last_hidden

                # Convert to numpy
                descriptor = features.squeeze().cpu().numpy()

            descriptor = self._descriptor_to_storage_dim(descriptor)
            return DinoResult(
                global_descriptor=descriptor,
                image_size=original_size,
                model_name=self.config.model_type
            )

        except torch.cuda.OutOfMemoryError as e:
            print(f"⚠️ CUDA out of memory: {str(e)}")
            print(f"Attempting recovery...")

            # Clear cache and garbage collect
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

            # Try with smaller image size
            if self.config.verbose:
                print("Trying with smaller image size...")

            try:
                # Create new inputs with smaller image size
                smaller_image = cv2.resize(image_rgb, (self.config.image_size // 2, self.config.image_size // 2))

                try:
                    inputs_small = self.processor(images=smaller_image, return_tensors="pt")
                except Exception:
                    from PIL import Image
                    pil_img = Image.fromarray(smaller_image)
                    inputs_small = self.processor(pil_img, return_tensors="pt")

                inputs_small = {k: v.to(self.config.device) for k, v in inputs_small.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs_small)
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        features = outputs.pooler_output
                    else:
                        last_hidden = outputs.last_hidden_state
                        if last_hidden.dim() == 3:
                            features = last_hidden.mean(dim=1)
                        else:
                            features = last_hidden

                descriptor = features.squeeze().cpu().numpy()
                descriptor = self._descriptor_to_storage_dim(descriptor)
                return DinoResult(
                    global_descriptor=descriptor,
                    image_size=original_size,
                    model_name=self.config.model_type
                )
            except torch.cuda.OutOfMemoryError:
                print(f"❌ DINO extraction failed - GPU memory exhausted")
                print(f"Available GPUs: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    mem = torch.cuda.get_device_capability(i)
                    print(f"  GPU {i}: {props.name} - Compute Capability: {mem}")
                    mem_free = torch.cuda.memory_allocated(i) / 1024**3
                    print(f"  Memory allocated: {mem_free:.2f} GB")
                return None

        except Exception as e:
            print(f"❌ DINO extraction error: {str(e)}")
            return None

    def extract_batch(self, images: list, batch_size: int = 8) -> list:
        """Extract features from multiple images using TRUE batch processing.

        Args:
            images: List of BGR images
            batch_size: Batch size for GPU processing (default: 8 for RTX 5060 Ti)

        Returns:
            List of DinoResult
        """
        if not images:
            return []

        if self.model is None:
            self._init_model()

        # Process in batches for maximum GPU utilization
        all_results = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            # Convert all images to RGB
            batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in batch_images]

            try:
                # Process entire batch at once using the processor
                inputs = self.processor(images=batch_rgb, return_tensors="pt")

                # Determine device: use CUDA if configured and available (both NVIDIA and AMD GPUs)
                use_cuda = self.config.device == "cuda" and torch.cuda.is_available()

                if use_cuda:
                    # GPU batch processing
                    inputs = {k: v.to(self.config.device, non_blocking=True) for k, v in inputs.items()}

                    # Synchronize for batch inference
                    torch.cuda.synchronize()

                    # Batch inference - all images processed simultaneously
                    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        outputs = self.model(**inputs)

                        # Extract features from batch
                        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                            features = outputs.pooler_output  # (batch_size, dim)
                        else:
                            last_hidden = outputs.last_hidden_state
                            if last_hidden.dim() == 3:
                                features = last_hidden.mean(dim=1)
                            else:
                                features = last_hidden

                        # Sync and move to CPU
                        torch.cuda.synchronize()
                        descriptors = features.cpu().numpy()

                        # Clear GPU references to prevent memory leaks
                        del features, outputs
                        torch.cuda.empty_cache()

                else:
                    # CPU batch processing
                    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

                    with torch.no_grad(), torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
                        outputs = self.model(**inputs)

                        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                            features = outputs.pooler_output
                        else:
                            last_hidden = outputs.last_hidden_state
                            if last_hidden.dim() == 3:
                                features = last_hidden.mean(dim=1)
                            else:
                                features = last_hidden

                    descriptors = features.cpu().numpy()

                # Create DinoResult objects for the batch
                for j, desc in enumerate(descriptors):
                    try:
                        d = self._descriptor_to_storage_dim(desc)
                        all_results.append(DinoResult(
                            global_descriptor=d,
                            image_size=batch_images[j].shape[:2],
                            model_name=self.config.model_type
                        ))
                    except Exception as e:
                        if self.config.verbose:
                            print(f"⚠️ Error creating result for image {i+j}: {e}")
                        # Append None to maintain index alignment
                        all_results.append(None)

            except torch.cuda.OutOfMemoryError:
                # Fallback to smaller batch or single image processing
                if self.config.verbose:
                    print(f"⚠️ CUDA OOM in batch processing, falling back to single image processing")

                for img in batch_images:
                    try:
                        result = self.extract(img)
                        all_results.append(result)
                    except Exception as e:
                        if self.config.verbose:
                            print(f"⚠️ Error processing image in fallback: {e}")
                        all_results.append(None)

            except Exception as e:
                if self.config.verbose:
                    print(f"❌ Batch extraction error: {e}")
                # Fallback to single image processing with individual error handling
                for img in batch_images:
                    try:
                        result = self.extract(img)
                        all_results.append(result)
                    except Exception as img_error:
                        if self.config.verbose:
                            print(f"⚠️ Error processing individual image: {img_error}")
                        all_results.append(None)

        return all_results

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """Convenience method to get just the embedding vector.

        Args:
            image: Input image in BGR format

        Returns:
            Global descriptor vector (1D numpy array)
        """
        return self.extract(image).global_descriptor

    def close(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None


def create_dino_processor(
    model_type: Optional[str] = None,
    device: Optional[str] = None,
    hf_token: Optional[str] = None,
    verbose: bool = False,
    use_multi_gpu: Optional[bool] = None,
    gpu_ids: Optional[list] = None,
    enable_memory_optimization: bool = True,
    use_4bit_quantization: bool = False,
    use_gradient_checkpointing: bool = False
) -> DinoProcessor:
    """Factory function to create DINO processor.

    Args:
        model_type: Model variant (dinov3-vitl16, dinov3-vitg14, dinov2-vitl14, dinov3-vit7b16)
        device: Device to run on ('cuda' or 'cpu')
        hf_token: HuggingFace token for gated models
        verbose: Enable verbose output
        use_multi_gpu: Enable DataParallel for multi-GPU inference
        gpu_ids: List of GPU IDs to use (e.g., [0, 1]), or None for all GPUs
        enable_memory_optimization: Enable PYTORCH memory optimization
        use_4bit_quantization: Enable 4-bit quantization for ultra-low memory (experimental)
        use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency

    Returns:
        Configured DinoProcessor instance

    Example:
        # For DINOv3 7B with 2 GPUs
        processor = create_dino_processor(
            model_type="dinov3-vit7b16",
            use_multi_gpu=True,
            gpu_ids=[0, 1],
            enable_memory_optimization=True,
            use_gradient_checkpointing=True,
            verbose=True
        )
    """
    # Build config from appConfig defaults, then apply any explicit overrides
    cfg = DinoConfig(verbose=verbose,
                     enable_memory_optimization=enable_memory_optimization,
                     use_4bit_quantization=use_4bit_quantization,
                     use_gradient_checkpointing=use_gradient_checkpointing)
    if model_type is not None:
        cfg.model_type = model_type
    if device is not None:
        cfg.device = device
    if hf_token is not None:
        cfg.hf_token = hf_token
    if use_multi_gpu is not None:
        cfg.use_multi_gpu = use_multi_gpu
    if gpu_ids is not None:
        cfg.gpu_ids = gpu_ids
    return DinoProcessor(cfg)


# Backward compatibility alias
DinoV3Processor = DinoProcessor


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="DINO Feature Extraction")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default=None, help="Model type (default: from appConfig)")
    parser.add_argument("--token", help="HuggingFace token")
    parser.add_argument("--output", help="Save descriptor to .npy file")
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Failed to load image: {args.image}")

    # Create processor
    dino = create_dino_processor(
        model_type=args.model,
        hf_token=args.token or "",
        verbose=True
    )

    # Extract features
    result = dino.extract(image) 

    if args.output:
        np.save(args.output, result.global_descriptor)
        print(f"Descriptor saved to: {args.output}")

