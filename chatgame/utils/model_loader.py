import torch
from transformers import GPT2Tokenizer
from transformers.modeling_gpt2 import GPT2LMHeadModel

MODELS = {
    "gpt2-medium":
        {
            "model": GPT2LMHeadModel,
            "tokenizer": GPT2Tokenizer,
            "default_model_settings": {"output_hidden_states": True}
        }
}


def initialize_model_and_tokenizer(model_name: str, device: str):
    model_info = MODELS[model_name]
    model_class = model_info["model"]
    model_tokenizer = model_info["tokenizer"]
    model = model_class.from_pretrained(**model_info["default_model_settings"])
    model.to(device)
    model.eval()
    tokenizer = model_tokenizer.from_pretrained(model_name)
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def prepare_text_primer(tokenizer, cond_text, device, debug_print=False):

    # Токенизируем строку
    if cond_text:
        tokenized_cond_text = tokenizer.encode(
            tokenizer.bos_token + cond_text,
            add_special_tokens=False
        )
    else:
        tokenized_cond_text = tokenizer.encode(
            [tokenizer.bos_token],
            add_special_tokens=False
        )

    # Превращаем список токенов в вектор токенов
    context_t = torch.tensor(tokenized_cond_text, device=device, dtype=torch.long)
    while len(context_t.shape) < 2:
        context_t = context_t.unsqueeze(0)

    if debug_print:
        print("= Prefix of sentence =")
        print(tokenizer.decode(tokenized_cond_text))
        print()

    return context_t
