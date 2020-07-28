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

from os import path

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from chatgame.utils.misc import decode_text_from_tokens
from chatgame.utils.model_loader import initialize_model_and_tokenizer, prepare_text_primer
from chatgame.bag_of_words.bow_utils import get_bag_of_words_indices
from chatgame.language_models.gpt2.text_generation import generate_unperturbed_text, generate_perturbed_text
from chatgame.classifiers.classifier import ClassificationHead, get_classifier

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
BAG_OF_WORDS_ADDRESSES = {
    "fantasy": "fantasy.txt",
    "politics": "politics.txt",
    "military": "military.txt",
    "science": "science.txt",
    "space": "space.txt",
    "technology": "technology.txt",
    "наука": "наука.txt",
    "технологии": "технологии.txt",
    "космос": "космос.txt"
}


def full_text_generation(
        model,
        tokenizer,
        context=None,
        device="cuda",
        bag_of_words=None,
        discrim=None,
        class_label=None,
        **kwargs

):
    """
    Загружает нужный мешок слов и/или нужный дискриминатор, исходя из переданного в аргументы
    командной строки. Генерирует невозмущенный текст и указанное количество возмущенных текстов.
    """
    print("INSIDE FULL TEXT GENERATION")
    directory = path.dirname(__file__)

    classifiers_dir = str(path.join(directory, '../../classifiers/'))
    classifier, class_id = get_classifier(discrim, class_label, device, classifiers_dir)

    bow_indices = []

    # Добавление универсального пути в BAG_OF_WORDS_ADDRESSES
    for key in BAG_OF_WORDS_ADDRESSES.keys():
        BAG_OF_WORDS_ADDRESSES[key] = path.join(directory,
                                                '../../bag_of_words/wordlists',
                                                BAG_OF_WORDS_ADDRESSES[key])

    # Если указан bag_of_words (напр. military), то читаем нужный файл со списком слов
    # и вовращаем список списков индексов токенов, полученных из этого мешка слов
    if bag_of_words:
        bow_indices = get_bag_of_words_indices(bag_of_words_ids_or_paths=bag_of_words.split(";"),
                                               tokenizer=tokenizer,
                                               bow_filepaths=BAG_OF_WORDS_ADDRESSES)
    # unpert_gen_tok_text = generate_unperturbed_text(
    #     model=model,
    #     context=context,
    #     length=length,
    #     sample=sample
    # )
    # if device == 'cuda':
    #     torch.cuda.empty_cache()
    #
    # print("Unperturbed generated")
    pert_gen_tok_texts = []

    # Генерируем возмущенный текст столько раз, сколько требуется num_samples
    for i in range(kwargs.pop("num_samples")):
        pert_gen_tok_text = generate_perturbed_text(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            bow_indices=bow_indices,
            classifier=classifier,
            class_label=class_id,
            **kwargs
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)

    if device == 'cuda':
        torch.cuda.empty_cache()

    return pert_gen_tok_texts


if __name__ == "__main__":
    from chatgame.language_models import RuGPT2LMHeadModel
    from chatgame.tokenizers import RuGPT2Tokenizer

    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    tokenizer = RuGPT2Tokenizer.from_pretrained("ru-gpt2")
    model = RuGPT2LMHeadModel.from_pretrained("ru-gpt2", output_hidden_states=True)
    bow = "наука"
    context = tokenizer.encode(tokenizer.bos_token + "Сегодня произошло событие в мире науки")

    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    # model = GPT2LMHeadModel.from_pretrained("gpt2-medium", output_hidden_states=True)
    # bow = "science"
    # context = tokenizer.encode(tokenizer.bos_token + "In this year")

    length = {"length": 100,
              "num_samples": 1}

    sampling = {"sample": True,
                "temperature": 1.1,
                "top_k": 10,
                "gm_scale": 0.9}

    gradient_descent = {"gamma": 1.5,
                        "kl_scale": 0.01,
                        "decay": False,
                        "window_length": 3,
                        "grad_length": 10000,
                        "num_iterations": 3,
                        "loss_type": 1,
                        "stepsize": 0.02}

    res = {}
    res.update(length)
    res.update(sampling)
    res.update(gradient_descent)

    for param in model.parameters():
        param.requires_grad = False

    full_text_generation(model, tokenizer,
                         context=torch.tensor([context]),
                         bag_of_words=bow,
                         device="cpu",
                         **res)

