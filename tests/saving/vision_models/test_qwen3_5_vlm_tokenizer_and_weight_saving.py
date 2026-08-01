import sys
import importlib.metadata
from unittest.mock import MagicMock

_orig_version = importlib.metadata.version
def _mock_version(name):
    if name == "unsloth_zoo":
        return "2025.1.1"
    return _orig_version(name)
importlib.metadata.version = _mock_version

if "unsloth_zoo" not in sys.modules:
    mock_zoo = MagicMock()
    mock_zoo.__version__ = "2025.1.1"
    sys.modules["unsloth_zoo"] = mock_zoo
    sys.modules["unsloth_zoo.saving_utils"] = MagicMock()
    sys.modules["unsloth_zoo.llama_cpp"] = MagicMock()
    sys.modules["unsloth_zoo.device_type"] = MagicMock()

import os
import json
import tempfile
import pytest
import torch
from unsloth.save import (
    _preserve_tokenizer_eos_token,
    _is_qwen3_5_vlm,
    _qwen3_5_vlm_state_dict_for_save,
    _remap_qwen3_5_vlm_saved_weights,
)


def test_clean_up_invalid_tokenizer_class():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = os.path.join(tmp_dir, "tokenizer_config.json")
        invalid_config = {
            "tokenizer_class": "TokenizersBackend",
            "bos_token": "<|im_start|>",
            "eos_token": "<|im_end|>",
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(invalid_config, f)

        # Mock tokenizer object
        class DummyTokenizer:
            tokenizer_class = "PreTrainedTokenizerFast"
            eos_token = "<|im_end|>"

        dummy_tok = DummyTokenizer()
        _preserve_tokenizer_eos_token(dummy_tok, tmp_dir)

        with open(config_path, "r", encoding="utf-8") as f:
            updated_config = json.load(f)

        assert updated_config["tokenizer_class"] == "PreTrainedTokenizerFast"
        assert updated_config["eos_token"] == "<|im_end|>"


def test_qwen3_5_vlm_detection():
    class DummyConfig:
        vision_config = {}
        architectures = ["Qwen3_5ForConditionalGeneration"]
        model_type = "qwen3_5"

    class DummyModel:
        config = DummyConfig()

    assert _is_qwen3_5_vlm(DummyModel()) is True


def test_qwen3_5_vlm_detection_through_peft_wrapper():
    class DummyConfig:
        vision_config = {}
        architectures = ["Qwen3_5ForConditionalGeneration"]

    class DummyBaseModel:
        config = DummyConfig()

    class DummyPeftModel:
        base_model = DummyBaseModel()

    assert _is_qwen3_5_vlm(DummyPeftModel()) is True


def test_qwen3_5_vlm_state_dict_remapping():
    raw_state_dict = {
        "language_model.model.embed_tokens.weight": torch.ones(10, 10),
        "visual.patch_embed.proj.weight": torch.ones(5, 5),
        "language_model.lm_head.weight": torch.ones(10, 10),
        "other_key": torch.zeros(2, 2),
    }

    remapped = _qwen3_5_vlm_state_dict_for_save(raw_state_dict)

    assert "model.language_model.embed_tokens.weight" in remapped
    assert "model.visual.patch_embed.proj.weight" in remapped
    assert "lm_head.weight" in remapped
    assert "other_key" in remapped
    assert "language_model.model.embed_tokens.weight" not in remapped


def test_remap_qwen3_5_vlm_saved_weights_files():
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_state_dict = {
            "language_model.model.embed_tokens.weight": torch.ones(4, 4),
            "visual.conv.weight": torch.ones(2, 2),
            "language_model.lm_head.weight": torch.ones(4, 4),
        }
        pt_path = os.path.join(tmp_dir, "model.bin")
        torch.save(raw_state_dict, pt_path)

        _remap_qwen3_5_vlm_saved_weights(tmp_dir)

        loaded_pt = torch.load(pt_path, map_location="cpu")
        assert "model.language_model.embed_tokens.weight" in loaded_pt
        assert "model.visual.conv.weight" in loaded_pt
        assert "lm_head.weight" in loaded_pt

        try:
            from safetensors.torch import save_file, load_file
            st_path = os.path.join(tmp_dir, "model.safetensors")
            save_file(raw_state_dict, st_path)

            _remap_qwen3_5_vlm_saved_weights(tmp_dir)

            loaded_st = load_file(st_path)
            assert "model.language_model.embed_tokens.weight" in loaded_st
            assert "model.visual.conv.weight" in loaded_st
            assert "lm_head.weight" in loaded_st
        except ImportError:
            pass
