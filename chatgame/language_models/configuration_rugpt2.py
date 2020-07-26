from pathlib import Path

from transformers import GPT2Config


class RuGPT2Config(GPT2Config):
    pretrained_config_archive_map = {
        "ru-gpt2": str(Path(__file__).parent / "models" / "ru-gpt2" / "config.json")
    }
    model_type = "gpt2"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
