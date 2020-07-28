import typing as tp
from torch import topk, where, ones_like, ones, cat, zeros, tensor, arange

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
    :return:
    """

    pert_gen_text = tokenizer.decode(tokens.tolist()[0])

    return pert_gen_text.lstrip("<|endoftext|>")


def mask_for_gradients(curr_length: int,
                       window_length: int,
                       past: tp.Tuple[tensor],
                       decay_mask: tp.Union[float, arange]) -> tensor:
    # Собираем тензор-маску, которая пригодится при вычислении нормы градиента
    # Форма как у тензоров в past, только на месте измерения, которое отвечало за длину послед-ти,
    # здесь стоит window_length
    ones_key_val_shape = (
            tuple(past[0].shape[:-2])
            + tuple([window_length])
            + tuple(past[0].shape[-1:])
    )

    # Форма как у тензоров в past, только на месте измерения, которое отвечало за длину послед-ти,
    # здесь стоит разница между текущей длиной послед-ти и window_length (curr_length > window_length) здесь
    zeros_key_val_shape = (
            tuple(past[0].shape[:-2])
            + tuple([curr_length - window_length])
            + tuple(past[0].shape[-1:])
    )

    ones_mask = ones(ones_key_val_shape)
    ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
    # permute переставляет измерения тензора местами, это аналог транспонирования для тензоров
    ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

    # Полученная маска будет умножаться на градиенты past, чтобы занулить значения, относящиеся к
    # дальним словам
    return cat((ones_mask, zeros(zeros_key_val_shape)), dim=-2)
