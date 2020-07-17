#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command with bag of words:
python examples/run_pplm.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95

Example command with discriminator:
python examples/run_pplm.py -D sentiment --class_label 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import argparse
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel

# from pplm_classification_head import ClassificationHead
from chatgame.utils.misc import decode_text_from_tokens, top_k_filter
from chatgame.utils.model_loader import initialize_model_and_tokenizer, prepare_text_primer
from chatgame.bag_of_words.bow_utils import build_bows_one_hot_vectors, get_bag_of_words_indices
from chatgame.language_models.gpt2.text_generation import generate_unperturbed_text

SMALL_CONST = 1e-15
BIG_CONST = 1e10

# Переменные для if-else
PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

# Адреса с текстовыми файлами для BOW
# TODO Сделать универсальные относительные пути
BAG_OF_WORDS_ADDRESSES = {
    "fantasy": "/home/andronov/chatbot_game/chatgame/bag_of_words/wordlists/fantasy.txt"
}


def full_text_generation(
        model,
        tokenizer,
        context=None,
        num_samples=1,
        device="cuda",
        bag_of_words=None,
        discrim=None,
        class_label=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        **kwargs
):
    """
    Загружает нужный мешок слов и/или нужный дискриминатор, исходя из переданного в аргументы
    командной строки. Генерирует невозмущенный текст и указанное количество возмущенных текстов.
    """

    bow_indices = []

    # Если указан --bag_of_words (напр. military), то читаем нужный файл со списком слов
    # и вовращаем список списков индексов токенов, полученных из этого мешка слов
    if bag_of_words:
        bow_indices = get_bag_of_words_indices(bag_of_words_ids_or_paths=bag_of_words.split(";"),
                                               tokenizer=tokenizer,
                                               bow_filepaths=BAG_OF_WORDS_ADDRESSES)

    unpert_gen_tok_text = generate_unperturbed_text(
        model=model,
        context=context,
        length=length,
        sample=sample
    )
    # if device == 'cuda':
    #     torch.cuda.empty_cache()
    #
    # pert_gen_tok_texts = []
    #
    # # Генерируем возмущенный текст столько раз, сколько требуется num_samples
    # for i in range(num_samples):
    #     pert_gen_tok_text = generate_text_pplm(
    #         model=model,
    #         tokenizer=tokenizer,
    #         context=context,
    #         device=device,
    #         perturb=True,
    #         bow_indices=bow_indices,
    #         length=length,
    #         stepsize=stepsize,
    #         temperature=temperature,
    #         top_k=top_k,
    #         sample=sample,
    #         num_iterations=num_iterations,
    #         grad_length=grad_length,
    #         horizon_length=horizon_length,
    #         window_length=window_length,
    #         decay=decay,
    #         gamma=gamma,
    #         gm_scale=gm_scale,
    #         kl_scale=kl_scale,
    #         verbosity_level=verbosity_level
    #     )
    #     pert_gen_tok_texts.append(pert_gen_tok_text)
    #
    # if device == 'cuda':
    #     torch.cuda.empty_cache()
    pert_gen_tok_texts = None
    return unpert_gen_tok_text, pert_gen_tok_texts


# def generate_text_pplm(
#         model,
#         tokenizer,
#         context=None,
#         past=None,
#         device="cuda",
#         perturb=True,
#         bow_indices=None,
#         classifier=None,
#         class_label=None,
#         loss_type=0,
#         length=100,
#         stepsize=0.02,
#         temperature=1.0,
#         top_k=10,
#         sample=True,
#         num_iterations=3,
#         grad_length=10000,
#         horizon_length=1,
#         window_length=0,
#         decay=False,
#         gamma=1.5,
#         gm_scale=0.9,
#         kl_scale=0.01,
#         verbosity_level=REGULAR
# ):
#     """
#     Генерирует возмущенный или невозмущенный текст. Что именно из этого - зависит от аргументов функции.
#     Возвращает список индексов токенов в последовательности и историю лоссов.
#     """
#     # Превращаем список индексов токенов затравки в тензор лонгов с двумя измерениями
#     output_so_far = None
#     if context:
#         context_t = torch.tensor(context, device=device, dtype=torch.long)
#         while len(context_t.shape) < 2:
#             context_t = context_t.unsqueeze(0)
#         output_so_far = context_t
#
#     # collect one hot vectors for bags of words
#     # Это нужно, если мы генерируем возмущенный текст
#     one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer,
#                                                       device)
#
#     grad_norms = None
#     last = None
#     unpert_discrim_loss = 0
#     loss_in_time = []
#
#     if verbosity_level >= VERBOSE:
#         range_func = trange(length, ascii=True)
#     else:
#         range_func = range(length)
#
#     for i in range_func:
#
#         # Get past/probs for current output, except for last word
#         # Note that GPT takes 2 inputs: past + current_token
#
#         # run model forward to obtain unperturbed
#         if past is None and output_so_far is not None:
#             last = output_so_far[:, -1:]
#
#             # Для начала, если у нас еще нет начального скрытого состояния,
#             # проходим по сети и получаем её состояние для последовательности без последнего токена
#             if output_so_far.shape[1] > 1:
#                 # past - это скрытое состояние сети, все key и value в self-attention слоях
#                 _, past, _ = model(output_so_far[:, :-1])
#
#         # Получаем предсказание языковой модели для имеющейся последовательности токенов
#         unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
#
#         # unpert_all_hidden - кортеж с выводами всех слоев для всех слов в послед-ти
#         # Во всех слоях каждому токену соответствует вектор размера 1024
#         # Слоёв всего 25
#
#         # Берем вывод последнего слоя в сети
#         unpert_last_hidden = unpert_all_hidden[-1]
#
#         # Не допускаем слишком больших градиентов
#         if i >= grad_length:
#             current_stepsize = stepsize * 0
#         else:
#             current_stepsize = stepsize
#
#         # modify the past if necessary
#
#         # Если мы не хотим возмущать скрытое состояние
#         if not perturb or num_iterations == 0:
#             pert_past = past
#
#         # Здесь мы хотим немного возмутить скрытое состояние сети past,
#         # и запустить сеть снова с возмущенным состоянием
#         else:
#             # Берем вывод последнего слоя сети для всех слов в текущей последовательности, кроме последнего
#             accumulated_hidden = unpert_last_hidden[:, :-1, :]
#
#             # Для всех слов в послед-ти суммируем все порожденные сетью вектора (их длина 1024)
#             accumulated_hidden = torch.sum(accumulated_hidden, dim=1)  # это будет вектор-строка с дробными числами
#
#             # Если у нас есть невозмущенное состояние сети, возмущаем его
#             if past is not None:
#                 pert_past, _, grad_norms, loss_this_iter = perturb_past(
#                     past,
#                     model,
#                     last,
#                     unpert_past=unpert_past,
#                     unpert_logits=unpert_logits,
#                     accumulated_hidden=accumulated_hidden,
#                     grad_norms=grad_norms,
#                     stepsize=current_stepsize,
#                     one_hot_bows_vectors=one_hot_bows_vectors,
#                     classifier=classifier,
#                     class_label=class_label,
#                     loss_type=loss_type,
#                     num_iterations=num_iterations,
#                     horizon_length=horizon_length,
#                     window_length=window_length,
#                     decay=decay,
#                     gamma=gamma,
#                     kl_scale=kl_scale,
#                     device=device,
#                     verbosity_level=verbosity_level
#                 )
#                 loss_in_time.append(loss_this_iter)
#             else:
#                 pert_past = past  # Здесь будет pert_past = None
#
#         # Снова вызываем языковую модель, но теперь с возмущенным скрытым состоянием
#         pert_logits, past, pert_all_hidden = model(last, past=pert_past)
#
#         # Берем логиты только для последнего токена в последовательности, делим на температуру.
#         # Температура T (от 0 до 1) влияет на семплирование. Чем она меньше, тем реже семплируются
#         # маловероятные слова. При T = 0 всегда берется токен, у которого максимальная вероятность.
#         pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
#
#         # Пересчитываем логиты последнего токена в их вероятностное распределение
#         pert_probs = F.softmax(pert_logits, dim=-1)
#
#         # Если указан классификатор
#         if classifier is not None:
#             ce_loss = torch.nn.CrossEntropyLoss()
#
#             # Делаем предсказание классификатора для вектора, который является усредненным по всем словам в
#             # послед-ти вектором вывода последнего слоя сети (вектор имеет размерность 1024)
#             prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
#             label = torch.tensor([class_label], device=device,
#                                  dtype=torch.long)
#
#             # Получаем значение cross entropy loss для полученного предсказания и заранее заявленной метки
#             unpert_discrim_loss = ce_loss(prediction, label)
#             if verbosity_level >= VERBOSE:
#                 print(
#                     "unperturbed discrim loss",
#                     unpert_discrim_loss.data.cpu().numpy()
#                 )
#         else:
#             unpert_discrim_loss = 0
#
#         # Если хотим, учитываем для моделирования последнего слова логиты и
#         # возмущенной, и невозмущенной модели
#         if perturb:
#
#             # Берем вероятностное распределение для последнего токена в последовательности для
#             # невозмущенной модели
#             unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
#
#             # Перемножаем почленно исправленные на степень вероятности токенов для возмущенной
#             # и невозмущенной модели. В статье это называется geometric mean fusion и служит
#             # стабильности модели, чтобы она, семплируя из распределения p(a|x), не забывала про само p(x)
#             pert_probs = ((pert_probs ** gm_scale) * (
#                     unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
#
#             # Из полученного вероятностного распределения отфильтровываем топ k максимальных чисел
#             pert_probs = top_k_filter(pert_probs, k=top_k,
#                                       probs=True)  # + SMALL_CONST
#
#             # Если слетела нормировка вероятностного распределения на единицу, восстанавливаем её
#             if torch.sum(pert_probs) <= 1:
#                 pert_probs = pert_probs / torch.sum(pert_probs)
#
#         # Или же просто получаем вероятностное распределение токенов для возмущенной модели
#         else:
#             pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
#             pert_probs = F.softmax(pert_logits, dim=-1)
#
#         # Из полученного вектора вероятностного распределения получаем
#         # один индекс какого-то логита в этом векторе
#         if sample:
#             # Семплируем индекс в соответствии с вероятностями
#             last = torch.multinomial(pert_probs, num_samples=1)
#
#         else:
#             # Или просто берем индекс максимальной вероятности
#             _, last = torch.topk(pert_probs, k=1, dim=-1)
#
#         # Добавляем полученный индекс в последовательность
#         output_so_far = (
#             last if output_so_far is None
#             else torch.cat((output_so_far, last), dim=1)
#         )
#         if verbosity_level >= REGULAR:
#             print(tokenizer.decode(output_so_far.tolist()[0]))
#
#     # unpert_disrim_loss - не 0 только в случае использования дискриминаторов
#     # loss_in_time - список лоссов из ф-ции perturb_past для каждой итерации, то есть
#     # для каждого генерируемого слова
#     return output_so_far, unpert_discrim_loss, loss_in_time


def run_pplm_example(
        pretrained_model="gpt2-medium",
        cond_text="",
        num_samples=40,
        bag_of_words=None,
        discrim=None,
        class_label=-1,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        seed=0,
        no_cuda=False,
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set the device
    DEVICE = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    model, tokenizer = initialize_model_and_tokenizer(model_name=pretrained_model,
                                                      device=DEVICE)

    context = prepare_text_primer(tokenizer=tokenizer, cond_text=cond_text, device=DEVICE)

    # Генерируем один невозмущенный текст и num_samples возмущенных текстов

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts

    unpert_gen_tok_text, pert_gen_tok_texts = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=DEVICE,
        num_samples=num_samples,
        bag_of_words=bag_of_words,
        discrim=discrim,
        class_label=class_label,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
    )

    # Декодируем последовательность невозмущенной модели
    # Получаем строку со всеми сгенерированными словами
    unpert_gen_text = decode_text_from_tokens(tokenizer=tokenizer,
                                              tokens=unpert_gen_tok_text)

    print("= Unperturbed generated text =")
    print(unpert_gen_text)
    print()

    # Для каждого из num_samples возмущенных текстов:

    # for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
    #     decoded_gen_text = decode_text_from_tokens(tokenizer=tokenizer,
    #                                                tokens=pert_gen_tok_text)
    #
    #     print(decoded_gen_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--cond_text", type=str, default="The lake",
        help="Prefix texts to condition on"
    )
    parser.add_argument(
        "--uncond", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default=None,
        help="Bags of words used for PPLM-BoW. "
             "Either a BOW id (see list in code) or a filepath. "
             "Multiple BoWs separated by ;",
    )
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "sentiment", "toxicity", "generic"),
        help="Discriminator to use",
    )
    parser.add_argument('--discrim_weights', type=str, default=None,
                        help='Weights for the generic discriminator')
    parser.add_argument('--discrim_meta', type=str, default=None,
                        help='Meta information for the generic discriminator')
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--colorama", action="store_true",
                        help="colors keywords")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")

    args = parser.parse_args()
    run_pplm_example(**vars(args))
