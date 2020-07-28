from typing import List, Any, Dict
from torch import tensor, zeros


def build_bows_one_hot_vectors(bow_indices: tensor,
                               tokenizer: Any,
                               device: str) -> tensor:
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


def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str],
                             tokenizer: Any,
                             bow_filepaths: Dict[str, str]) -> List[List[List[int]]]:
    """
    Превращает список слов из файла в список списков индексов токенов.
    Для каждой строки - свой список с токенами
    :param bag_of_words_ids_or_paths: Список названий тем. Зачастую длины один.
    :param tokenizer: Токенизатор соответствующей языковой модели.
    :param bow_filepaths: Словарь, связывающий название темы с путем до файла,
    содержащего её слова.
    :return: Каждая запись в мешке слов представляется списком индексов токенов, возвращается
    список этих списков по типу [[ [1], [2] ]]
    """

    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        filepath = bow_filepaths[id_or_path]  # TODO потенциальный KeyError
        with open(filepath, "r") as f:
            words = f.read().strip().split("\n")
        bow_indices.append(
            [tokenizer.encode(word.strip(),
                              add_prefix_space=True,
                              add_special_tokens=False)
             for word in words])
    return bow_indices
