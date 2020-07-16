from torch import tensor, zeros


def build_bows_one_hot_vectors(bow_indices, tokenizer, device):
    """
    Превращает вектор индексов токенов в матрицу, где в строках
    находятся one-hot-вектора соответствующих индексов токенов.

    Это нужно, чтобы впоследствии удобно вычислять лосс для направления градиентов модели
    с помощью BOW
    """
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        one_hot_bow = zeros(num_words, tokenizer.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors
