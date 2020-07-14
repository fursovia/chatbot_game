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

from pplm_classification_head import ClassificationHead

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
BAG_OF_WORDS_ARCHIVE_MAP = {
    'legal': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    'military': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    'monsters': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    'politics': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    'positive_words': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    'religion': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    'science': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    'space': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    'technology': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}


def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    """
    Переводит обычные тензоры в тензоры, способные к автоматическому дифференцированию.
    Такой код по сути устарел, сейчас в торче все тензоры и так способны к автоматическому
    дифференцированию.
    """
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)


def perturb_past(
        past,  # tuple[tensor]
        model,
        last,  # [[int]] - tensor
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        one_hot_bows_vectors=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        kl_scale=0.01,
        device='cuda',
        verbosity_level=REGULAR
):
    """
    Принимает на вход кортеж past = H_t из статьи и возвращает возмущенное
    состояние H_t + Delta{H_t}. Подход явно подогнан под архитектуру моделей
    из huggingface. Возможно, даже конкретно под GPT2LMHeadModel (см. док-цию 
    https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel)
    """
    # Generate inital perturbed past
    # Переменная со структурой, как у past, но из одних нулей
    # Это тот тамый \Delta{H_t} из статьи, получаемый обновлениями в несколько итераций
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        # Вектор возрастающих чисел меньше единицы, начиная не с нуля, длины window_length.
        # Используется для построения маски, призванной скрывать вклады далеких от текущего слова
        # слов в последовательности в предсказание следующего слова
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    # Собираем тензор-маску, которая пригодится при вычислении нормы градиента
    if curr_length > window_length > 0:

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

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        # permute переставляет измерения тензора местами, это аналог транспонирования для тензоров
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        # Полученная маска будет умножаться на градиенты past, чтобы занулить значения, относящиеся к
        # дальним словам
        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(device)
    else:
        # Если текущая длина послед-ти слишком мала, то маска состоит из одинх единиц и не
        # играет роли.
        window_mask = torch.ones_like(past[0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    # Получаем финальный Delta{H_t} в несколько итераций
    for i in range(num_iterations):
        if verbosity_level >= VERBOSE:
            print("Iteration ", i + 1)
        # Превращаем нампаевский тензор градиентов в Variable, требующий градиент
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        # Делаем H_t = H_t + /Delta{H_t}
        perturbed_past = list(map(add, past, curr_perturbation))
        # _, _, _, curr_length, _ = curr_perturbation[0].shape

        # Запускаем модель с возмущенным H_t, чтобы получить вывод последнего слоя сети для одного токена
        # last - последний токен в послед-ти
        all_logits, _, all_hidden = model(last, past=perturbed_past)
        hidden = all_hidden[-1]  # shape: 1, 1, 1024

        # accumulated_hidden был суммой векторов на выходе из сети для всех слов в послед-ти,
        # не считая последнего. Теперь мы добавляем вектор последнего слова, но из возмущенной модели
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()  # shape: 1, 1024

        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)

        # Получаем вероятностное распределение для последнего токена
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)  # shape [1, 50257]

        # Считаем лосс как сумму трех двух или трех слагаемых
        loss = torch.tensor(0.0)
        loss_list = []
        
        # Для метода мешка слов
        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            # Это цикл длины один
            for one_hot_bow in one_hot_bows_vectors:
                # Получаем вектор с вероятностями для всех токенов в мешке слов
                bow_probs = torch.mm(probs, torch.t(one_hot_bow))  # (1, 50257) * (18, 50257).T = (1, 18)
                # Используем минус логарифм суммы вероятностей как лосс
                bow_loss = -torch.log(torch.sum(bow_probs))
                loss += bow_loss
                loss_list.append(bow_loss)
            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_bow_loss:", loss.data.cpu().numpy())
                
        # Для метода дискриминаторов
        if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
            ce_loss = torch.nn.CrossEntropyLoss()
            # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings()
            for _ in range(horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                _, curr_unpert_past, curr_all_hidden = model(
                    past=curr_unpert_past,
                    inputs_embeds=inputs_embeds
                )
                curr_hidden = curr_all_hidden[-1]
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                    curr_hidden, dim=1)

            prediction = classifier(new_accumulated_hidden /
                                    (curr_length + 1 + horizon_length))

            label = torch.tensor(prediction.shape[0] * [class_label],
                                 device=device,
                                 dtype=torch.long)
            discrim_loss = ce_loss(prediction, label)
            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            loss_list.append(discrim_loss)

        kl_loss = torch.tensor(0.0)
        # Рассчитаем расхождение Кульбака-Лейблера между вероятностными распределениями
        # для токенов в возмущенной и невозмущенной модели
        if kl_scale > 0.0:
            # Берем распределение для невозмущенной модели
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            # Нулевые вероятности делаем равными малой константе
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                device).detach()
            corrected_probs = probs + correction.detach()

            # Рассчитываем расхождение Кульбака-Лейблера по соответствующей формуле,
            # берем с коэффициентом kl_scale
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            if verbosity_level >= VERY_VERBOSE:
                print(' kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        # Обновляет тензор в поле .grad для curr_perturbation по цепи loss->bow_loss->bow_probs->
        # ->probs->logits->all_logits->perturbed_past->curr_perturbation (аналогично для других вкладов в лосс) 
        loss.backward()

        # calculate gradient norms
        # grad_norms приходит в аргументах и возвращается в значениях,
        # то есть обновляется в цикле по вызовам perturb_past, начиная с None

        # Перед вычислением нормы градиент умножается на вычисленную ранее маску
        # Берется норма матрицы по умолчанию, то есть норма Фробениуса
        # (корень из суммы квадратов всех элементов матрицы)
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                for index, p_ in enumerate(curr_perturbation)
            ]

        # Нормализуем градиенты в curr_perturbation
        grad = [
            -stepsize *
            (p_.grad * window_mask / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # Обновляем определенную в начале функции переменную grad_accumulator
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    # Теперь pert_past - это H_t + \Delta{H_t} из статьи
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def get_classifier(
        name: Optional[str],
        class_label: Union[str, int],
        device: str,
        verbosity_level: int = REGULAR
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    """
    Загружаем предварительно сохранный малый торчевский классификатор-дискриминатор по имени.
    """
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified "
                         "in the discriminator model parameters")
    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str], tokenizer) -> \
        List[List[List[int]]]:
    """
    Превращает список слов из файла в список списков индексов токенов.
    Для каждой строки - свой список с токенами
    """
    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
            filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[id_or_path])
        else:
            filepath = id_or_path
        with open(filepath, "r") as f:
            words = f.read().strip().split("\n")
        bow_indices.append(
            [tokenizer.encode(word.strip(),
                              # похоже, разбивает длинные слова на несколько коротких частей-токенов
                              add_prefix_space=True,
                              add_special_tokens=False)
             for word in words])
    return bow_indices


def build_bows_one_hot_vectors(bow_indices, tokenizer, device='cuda'):
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
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors


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
    classifier, class_id = get_classifier(
        discrim,
        class_label,
        device
    )

    bow_indices = []

    # Если указан --bag_of_words (напр. military), то читаем нужный файл со списком слов
    # и вовращаем список списков индексов токенов, полученных из этого мешка слов
    if bag_of_words:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                               tokenizer)

    if bag_of_words and classifier:
        loss_type = PPLM_BOW_DISCRIM
        if verbosity_level >= REGULAR:
            print("Both PPLM-BoW and PPLM-Discrim are on. "
                  "This is not optimized.")

    elif bag_of_words:
        loss_type = PPLM_BOW
        if verbosity_level >= REGULAR:
            print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        if verbosity_level >= REGULAR:
            print("Using PPLM-Discrim")

    else:
        raise Exception("Specify either a bag of words or a discriminator")

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=length,
        sample=sample,
        perturb=False,
        verbosity_level=verbosity_level
    )
    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    # Генерируем возмущенный текст столько раз, сколько требуется num_samples
    for i in range(num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            perturb=True,
            bow_indices=bow_indices,
            classifier=classifier,
            class_label=class_id,
            loss_type=loss_type,
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
            verbosity_level=verbosity_level
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == 'cuda':
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time


def generate_text_pplm(
        model,
        tokenizer,
        context=None,
        past=None,
        device="cuda",
        perturb=True,
        bow_indices=None,
        classifier=None,
        class_label=None,
        loss_type=0,
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
        verbosity_level=REGULAR
):
    """
    Генерирует возмущенный или невозмущенный текст. Что именно из этого - зависит от аргументов функции.
    Возвращает список индексов токенов в последовательности и историю лоссов.
    """
    # Превращаем список индексов токенов затравки в тензор лонгов с двумя измерениями
    output_so_far = None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

    # collect one hot vectors for bags of words
    # Это нужно, если мы генерируем возмущенный текст
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer,
                                                      device)

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    if verbosity_level >= VERBOSE:
        range_func = trange(length, ascii=True)
    else:
        range_func = range(length)

    for i in range_func:

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]

            # Для начала, если у нас еще нет начального скрытого состояния,
            # проходим по сети и получаем её состояние для последовательности без последнего токена
            if output_so_far.shape[1] > 1:
                # past - это скрытое состояние сети, все key и value в self-attention слоях
                _, past, _ = model(output_so_far[:, :-1])

        # Получаем предсказание языковой модели для имеющейся последовательности токенов
        unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)

        # unpert_all_hidden - кортеж с выводами всех слоев для всех слов в послед-ти
        # Во всех слоях каждому токену соответствует вектор размера 1024
        # Слоёв всего 25

        # Берем вывод последнего слоя в сети
        unpert_last_hidden = unpert_all_hidden[-1]

        # Не допускаем слишком больших градиентов
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary

        # Если мы не хотим возмущать скрытое состояние
        if not perturb or num_iterations == 0:
            pert_past = past

        # Здесь мы хотим немного возмутить скрытое состояние сети past,
        # и запустить сеть снова с возмущенным состоянием
        else:
            # Берем вывод последнего слоя сети для всех слов в текущей последовательности, кроме последнего
            accumulated_hidden = unpert_last_hidden[:, :-1, :]

            # Для всех слов в послед-ти суммируем все порожденные сетью вектора (их длина 1024)
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)  # это будет вектор-строка с дробными числами

            # Если у нас есть невозмущенное состояние сети, возмущаем его
            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    one_hot_bows_vectors=one_hot_bows_vectors,
                    classifier=classifier,
                    class_label=class_label,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                    verbosity_level=verbosity_level
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past  # Здесь будет pert_past = None

        # Снова вызываем языковую модель, но теперь с возмущенным скрытым состоянием
        pert_logits, past, pert_all_hidden = model(last, past=pert_past)

        # Берем логиты только для последнего токена в последовательности, делим на температуру.
        # Температура T (от 0 до 1) влияет на семплирование. Чем она меньше, тем реже семплируются
        # маловероятные слова. При T = 0 всегда берется токен, у которого максимальная вероятность.
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST

        # Пересчитываем логиты последнего токена в их вероятностное распределение
        pert_probs = F.softmax(pert_logits, dim=-1)

        # Если указан классификатор
        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()

            # Делаем предсказание классификатора для вектора, который является усредненным по всем словам в
            # послед-ти вектором вывода последнего слоя сети (вектор имеет размерность 1024)
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([class_label], device=device,
                                 dtype=torch.long)

            # Получаем значение cross entropy loss для полученного предсказания и заранее заявленной метки
            unpert_discrim_loss = ce_loss(prediction, label)
            if verbosity_level >= VERBOSE:
                print(
                    "unperturbed discrim loss",
                    unpert_discrim_loss.data.cpu().numpy()
                )
        else:
            unpert_discrim_loss = 0

        # Если хотим, учитываем для моделирования последнего слова логиты и
        # возмущенной, и невозмущенной модели
        if perturb:

            # Берем вероятностное распределение для последнего токена в последовательности для
            # невозмущенной модели
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            # Перемножаем почленно исправленные на степень вероятности токенов для возмущенной
            # и невозмущенной модели. В статье это называется geometric mean fusion и служит
            # стабильности модели, чтобы она, семплируя из распределения p(a|x), не забывала про само p(x)
            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST

            # Из полученного вероятностного распределения отфильтровываем топ k максимальных чисел
            pert_probs = top_k_filter(pert_probs, k=top_k,
                                      probs=True)  # + SMALL_CONST

            # Если слетела нормировка вероятностного распределения на единицу, восстанавливаем её
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        # Или же просто получаем вероятностное распределение токенов для возмущенной модели
        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # Из полученного вектора вероятностного распределения получаем
        # один индекс какого-то логита в этом векторе
        if sample:
            # Семплируем индекс в соответствии с вероятностями
            last = torch.multinomial(pert_probs, num_samples=1)

        else:
            # Или просто берем индекс максимальной вероятности
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # Добавляем полученный индекс в последовательность
        output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )
        if verbosity_level >= REGULAR:
            print(tokenizer.decode(output_so_far.tolist()[0]))

    # unpert_disrim_loss - не 0 только в случае использования дискриминаторов
    # loss_in_time - список лоссов из ф-ции perturb_past для каждой итерации, то есть
    # для каждого генерируемого слова
    return output_so_far, unpert_discrim_loss, loss_in_time


def set_generic_model_params(discrim_weights, discrim_meta):
    if discrim_weights is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_weights need to be specified')
    if discrim_meta is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_meta need to be specified')

    with open(discrim_meta, 'r') as discrim_meta_file:
        meta = json.load(discrim_meta_file)
    meta['path'] = discrim_weights
    DISCRIMINATOR_MODELS_PARAMS['generic'] = meta


def run_pplm_example(
        pretrained_model="gpt2-medium",
        cond_text="",
        uncond=False,
        num_samples=1,
        bag_of_words=None,
        discrim=None,
        discrim_weights=None,
        discrim_meta=None,
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
        colorama=False,
        verbosity='regular'
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    if discrim == 'generic':
        set_generic_model_params(discrim_weights, discrim_meta)

    if discrim is not None:
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim][
            "pretrained_model"
        ]
        if pretrained_model != discriminator_pretrained_model:
            pretrained_model = discriminator_pretrained_model
            if verbosity_level >= REGULAR:
                print("discrim = {}, pretrained_model set "
                      "to discriminator's = {}".format(discrim, pretrained_model))

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        pretrained_model,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    if uncond:
        tokenized_cond_text = tokenizer.encode(
            [tokenizer.bos_token],
            add_special_tokens=False
        )
    else:
        raw_text = cond_text
        while not raw_text:
            print("Did you forget to add `--cond_text`? ")
            raw_text = input("Model prompt >>> ")
        tokenized_cond_text = tokenizer.encode(
            tokenizer.bos_token + raw_text,
            add_special_tokens=False
        )

    print("= Prefix of sentence =")
    print(tokenizer.decode(tokenized_cond_text))
    print()

    # Генерируем один невозмущенный текст и num_samples возмущенных текстов

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time

    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_cond_text,
        device=device,
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
        verbosity_level=verbosity_level
    )

    # Декодируем последовательность невозмущенной модели
    # Получаем строку со всеми сгенерированными словами
    unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])

    if verbosity_level >= REGULAR:
        print("=" * 80)
    print("= Unperturbed generated text =")
    print(unpert_gen_text)
    print()

    generated_texts = []

    bow_word_ids = set()

    # colorama отвечает за подсветку слов в консольном выводе
    if bag_of_words and colorama:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                               tokenizer)
        for single_bow_list in bow_indices:
            # Для каждого слова в мешке слов берем только те, которые состоят из одного токена
            filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
            # Получаем множество индексов токенов для мешка слов
            bow_word_ids.update(w[0] for w in filtered)

    # Для каждого из num_samples возмущенных текстов:
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            # untokenize unperturbed text
            if colorama:
                import colorama

                pert_gen_text = ''
                # Для каждого индекса токена в последовательности возмущенной модели
                # добавляем в строку с декодированными словами либо просто декодированное слово,
                # либо декодированное слово красным шрифтом, если токен этого слова есть в множестве
                # индексов для мешка слов
                for word_id in pert_gen_tok_text.tolist()[0]:
                    if word_id in bow_word_ids:
                        pert_gen_text += '{}{}{}'.format(
                            colorama.Fore.RED,
                            tokenizer.decode([word_id]),
                            colorama.Style.RESET_ALL
                        )
                    else:
                        pert_gen_text += tokenizer.decode([word_id])
            else:
                pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])

            # Выводим в консоль расшифрованный возмущенный текст
            print("= Perturbed generated text {} =".format(i + 1))
            print(pert_gen_text)
            print()
        except:
            pass

        # keep the prefix, perturbed seq, original seq for each index
        generated_texts.append(
            (tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text)
        )

    return


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

