from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
import os


def get_model(**kwargs):
    """Load HuggingFace pretrained model."""
    # Remove custom-only kwargs that are not valid for HF from_pretrained
    custom_only = {
        "vocab_size",
        "max_seq_len",
        "dim",
        "n_heads",
        "n_layers",
        "mlp_ratio",
        "n_latent_recursions",
        "n_improvement_cycles",
        "checkpoint_path",
    }
    kwargs = {k: v for k, v in kwargs.items() if k not in custom_only}
    if kwargs.pop("model_parallel", False):
        model = AutoModelForCausalLM.from_pretrained(device_map="auto", **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
    return model


def get_custom_model(model_type, checkpoint_path=None, **kwargs):
    """
    Load custom model classes.

    Args:
        model_type (str): Type of custom model to load.
            Currently supports: "tiny_recursive"
        checkpoint_path (str | None): Path to a trained .pt checkpoint.
            If provided, loads model_state_dict from the checkpoint so the
            model starts from trained weights instead of random initialisation.
            Pass None or the string "null" to skip checkpoint loading.
        **kwargs: Model-specific architecture parameters.

    Returns:
        Instantiated model (optionally with trained weights loaded).
    """
    if model_type == "tiny_recursive":
        from .tiny_recursive_model import TinyRecursiveModel

        supported_kwargs = {
            "vocab_size",
            "dim",
            "n_heads",
            "n_kv_heads",
            "n_layers",
            "mlp_ratio",
            "ffn_multiplier",
            "max_seq_len",
            "n_latent_recursions",
            "n_improvement_cycles",
            "dropout",
            "adapter_dropout",
            "adapter_every_k",
            "use_task_adapter",
            "use_checkpoint",
            "use_less_is_more",
            "ema_decay",
            "numerical_input_dim",
        }
        clean_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}
        # Default vocab_size=50000 matches the inference command (vocab_size=50000)
        clean_kwargs.setdefault("vocab_size", kwargs.get("vocab_size", 50000))
        model = TinyRecursiveModel(**clean_kwargs)

        # ── Load trained checkpoint ──────────────────────────────────────────
        if checkpoint_path is not None and str(checkpoint_path) not in ("null", "None", ""):
            if os.path.isfile(str(checkpoint_path)):
                print(f"[model.py] Loading checkpoint: {checkpoint_path}")
                ckpt = torch.load(str(checkpoint_path), map_location="cpu")
                # Support both raw state_dict and wrapped checkpoint dicts
                state_dict = ckpt.get("model_state_dict", ckpt)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    n = len(missing)
                    print(f"[model.py] Missing keys ({n}): {missing[:5]}{'...' if n > 5 else ''}")
                if unexpected:
                    n = len(unexpected)
                    print(f"[model.py] Unexpected keys ({n}): {unexpected[:5]}{'...' if n > 5 else ''}")
                print("[model.py] Checkpoint loaded successfully.")
            else:
                print(
                    f"[model.py] WARNING: checkpoint_path='{checkpoint_path}' does not exist. "
                    "Model will use random weights — predictions will be meaningless."
                )

        return model
    else:
        raise ValueError(f"Unknown custom model type: {model_type}")


def get_model_by_name(model_name, model_type="huggingface", checkpoint_path=None, **kwargs):
    """
    Unified interface to load models by name.

    Args:
        model_name (str): HuggingFace model identifier OR custom model type
            (e.g. "tiny_recursive").
        model_type (str): "huggingface" or "custom".
        checkpoint_path (str | None): Path to a trained checkpoint (.pt).
            Only used when model_type="custom".
        **kwargs: Additional model/tokenizer parameters.

    Returns:
        Instantiated model.
    """
    if model_type == "huggingface":
        return get_model(pretrained_model_name_or_path=model_name, **kwargs)
    elif model_type == "custom":
        return get_custom_model(model_name, checkpoint_path=checkpoint_path, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
