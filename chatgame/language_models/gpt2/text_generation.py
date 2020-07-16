import torch
from chatgame.bag_of_words.bow_utils import build_bows_one_hot_vectors


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
):
    """
    Генерирует возмущенный или невозмущенный текст. Что именно из этого - зависит от аргументов функции.
    Возвращает список индексов токенов в последовательности и историю лоссов.
    """
    # Превращаем список индексов токенов затравки в тензор лонгов с двумя измерениями

    output_so_far = context

    # collect one hot vectors for bags of words
    # Это нужно, если мы генерируем возмущенный текст
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer,
                                                      device)

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    for i in range(length):

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
