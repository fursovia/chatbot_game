from pathlib import Path

from transformers import GPT2LMHeadModel
from .configuration_rugpt2 import RuGPT2Config


class RuGPT2LMHeadModel(GPT2LMHeadModel):
    pretrained_model_archive_map = {
        "ru-gpt2": str(Path(__file__).parent / "models" / "ru-gpt2" / "pytorch_model.bin")
    }
    config_class = RuGPT2Config

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
