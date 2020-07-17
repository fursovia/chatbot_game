from torch import topk, where, ones_like

BIG_CONST = 1e10


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return where(logits < batch_mins,
                         ones_like(logits) * 0.0, logits)
        return where(logits < batch_mins,
                     ones_like(logits) * -BIG_CONST,
                     logits)


def decode_text_from_tokens(tokenizer, tokens) -> str:
    """
    Декодирует список (вектор) индексов токенов в строку текста
    :param tokenizer:
    :param tokens: torch.tensor
    :param colorama: bool
    :param bow_word_ids: set
    :return:
    """

    pert_gen_text = tokenizer.decode(tokens.tolist()[0])

    return pert_gen_text.lstrip("<|endoftext|>")
