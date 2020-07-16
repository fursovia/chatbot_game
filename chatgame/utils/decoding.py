def decode_text_from_tokens(tokenizer, tokens):
    """
    Декодирует список (вектор) индексов токенов в строку текста
    :param tokenizer:
    :param tokens: torch.tensor
    :param colorama: bool
    :param bow_word_ids: set
    :return:
    """

    pert_gen_text = tokenizer.decode(tokens.tolist()[0])

    return pert_gen_text
