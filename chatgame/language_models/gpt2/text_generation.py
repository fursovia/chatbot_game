import torch
import torch.nn.functional as F
from chatgame.bag_of_words.bow_utils import build_bows_one_hot_vectors
from chatgame.utils.misc import top_k_filter


def generate_unperturbed_text(
        model,
        context=None,
        past=None,
        length=40,
        temperature=1.0,
        top_k=10,
        sample=True
):
    """
    Генерирует возмущенный или невозмущенный текст. Что именно из этого - зависит от аргументов функции.
    Возвращает список индексов токенов в последовательности и историю лоссов.
    """
    # Превращаем список индексов токенов затравки в тензор лонгов с двумя измерениями

    output_so_far = context

    last = None

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

        logits, past, _ = model(last, past=past)
        logits = logits[:, -1, :] / temperature  # + SMALL_CONST

        logits = top_k_filter(logits, k=top_k)  # + SMALL_CONST
        probs = F.softmax(logits, dim=-1)

        if sample:
            # Семплируем индекс в соответствии с вероятностями
            last = torch.multinomial(probs, num_samples=1)

        else:
            # Или просто берем индекс максимальной вероятности
            _, last = torch.topk(probs, k=1, dim=-1)

        # Добавляем полученный индекс в последовательность
        output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )

    return output_so_far


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
# ):
#     """
#     Генерирует возмущенный или невозмущенный текст. Что именно из этого - зависит от аргументов функции.
#     Возвращает список индексов токенов в последовательности и историю лоссов.
#     """
#     # Превращаем список индексов токенов затравки в тензор лонгов с двумя измерениями
#
#     output_so_far = context
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
#     for i in range(length):
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
#
#
# def perturb_past(
#         past,  # tuple[tensor]
#         model,
#         last,  # [[int]] - tensor
#         unpert_past=None,
#         unpert_logits=None,
#         accumulated_hidden=None,
#         grad_norms=None,
#         stepsize=0.01,
#         one_hot_bows_vectors=None,
#         classifier=None,
#         class_label=None,
#         loss_type=0,
#         num_iterations=3,
#         horizon_length=1,
#         window_length=0,
#         decay=False,
#         gamma=1.5,
#         kl_scale=0.01,
#         device='cuda',
#         verbosity_level=REGULAR
# ):
#     """
#     Принимает на вход кортеж past = H_t из статьи и возвращает возмущенное
#     состояние H_t + Delta{H_t}. Подход явно подогнан под архитектуру моделей
#     из huggingface. Возможно, даже конкретно под GPT2LMHeadModel (см. док-цию
#     https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel)
#     """
#     # Generate inital perturbed past
#     # Переменная со структурой, как у past, но из одних нулей
#     # Это тот тамый \Delta{H_t} из статьи, получаемый обновлениями в несколько итераций
#     grad_accumulator = [
#         (np.zeros(p.shape).astype("float32"))
#         for p in past
#     ]
#
#     if accumulated_hidden is None:
#         accumulated_hidden = 0
#
#     if decay:
#         # Вектор возрастающих чисел меньше единицы, начиная не с нуля, длины window_length.
#         # Используется для построения маски, призванной скрывать вклады далеких от текущего слова
#         # слов в последовательности в предсказание следующего слова
#         decay_mask = torch.arange(
#             0.,
#             1.0 + SMALL_CONST,
#             1.0 / (window_length)
#         )[1:]
#     else:
#         decay_mask = 1.0
#
#     # TODO fix this comment (SUMANTH)
#     # Generate a mask is gradient perturbated is based on a past window
#     _, _, _, curr_length, _ = past[0].shape
#
#     # Собираем тензор-маску, которая пригодится при вычислении нормы градиента
#     if curr_length > window_length > 0:
#
#         # Форма как у тензоров в past, только на месте измерения, которое отвечало за длину послед-ти,
#         # здесь стоит window_length
#         ones_key_val_shape = (
#                 tuple(past[0].shape[:-2])
#                 + tuple([window_length])
#                 + tuple(past[0].shape[-1:])
#         )
#
#         # Форма как у тензоров в past, только на месте измерения, которое отвечало за длину послед-ти,
#         # здесь стоит разница между текущей длиной послед-ти и window_length (curr_length > window_length) здесь
#         zeros_key_val_shape = (
#                 tuple(past[0].shape[:-2])
#                 + tuple([curr_length - window_length])
#                 + tuple(past[0].shape[-1:])
#         )
#
#         ones_mask = torch.ones(ones_key_val_shape)
#         ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
#         # permute переставляет измерения тензора местами, это аналог транспонирования для тензоров
#         ones_mask = ones_mask.permute(0, 1, 2, 4, 3)
#
#         # Полученная маска будет умножаться на градиенты past, чтобы занулить значения, относящиеся к
#         # дальним словам
#         window_mask = torch.cat(
#             (ones_mask, torch.zeros(zeros_key_val_shape)),
#             dim=-2
#         ).to(device)
#     else:
#         # Если текущая длина послед-ти слишком мала, то маска состоит из одинх единиц и не
#         # играет роли.
#         window_mask = torch.ones_like(past[0]).to(device)
#
#     # accumulate perturbations for num_iterations
#     loss_per_iter = []
#     new_accumulated_hidden = None
#     # Получаем финальный Delta{H_t} в несколько итераций
#     for i in range(num_iterations):
#         if verbosity_level >= VERBOSE:
#             print("Iteration ", i + 1)
#         # Превращаем нампаевский тензор градиентов в Variable, требующий градиент
#         curr_perturbation = [
#             to_var(torch.from_numpy(p_), requires_grad=True, device=device)
#             for p_ in grad_accumulator
#         ]
#
#         # Compute hidden using perturbed past
#         # Делаем H_t = H_t + /Delta{H_t}
#         perturbed_past = list(map(add, past, curr_perturbation))
#         # _, _, _, curr_length, _ = curr_perturbation[0].shape
#
#         # Запускаем модель с возмущенным H_t, чтобы получить вывод последнего слоя сети для одного токена
#         # last - последний токен в послед-ти
#         all_logits, _, all_hidden = model(last, past=perturbed_past)
#         hidden = all_hidden[-1]  # shape: 1, 1, 1024
#
#         # accumulated_hidden был суммой векторов на выходе из сети для всех слов в послед-ти,
#         # не считая последнего. Теперь мы добавляем вектор последнего слова, но из возмущенной модели
#         new_accumulated_hidden = accumulated_hidden + torch.sum(
#             hidden,
#             dim=1
#         ).detach()  # shape: 1, 1024
#
#         # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
#
#         # Получаем вероятностное распределение для последнего токена
#         logits = all_logits[:, -1, :]
#         probs = F.softmax(logits, dim=-1)  # shape [1, 50257]
#
#         # Считаем лосс как сумму трех двух или трех слагаемых
#         loss = torch.tensor(0.0)
#         loss_list = []
#
#         # Для метода мешка слов
#         if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
#             # Это цикл длины один
#             for one_hot_bow in one_hot_bows_vectors:
#                 # Получаем вектор с вероятностями для всех токенов в мешке слов
#                 bow_probs = torch.mm(probs, torch.t(one_hot_bow))  # (1, 50257) * (18, 50257).T = (1, 18)
#                 # Используем минус логарифм суммы вероятностей как лосс
#                 bow_loss = -torch.log(torch.sum(bow_probs))
#                 loss += bow_loss
#                 loss_list.append(bow_loss)
#             if verbosity_level >= VERY_VERBOSE:
#                 print(" pplm_bow_loss:", loss.data.cpu().numpy())
#
#         # Для метода дискриминаторов
#         if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
#             ce_loss = torch.nn.CrossEntropyLoss()
#             # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
#             curr_unpert_past = unpert_past
#             curr_probs = torch.unsqueeze(probs, dim=1)
#             wte = model.resize_token_embeddings()
#             for _ in range(horizon_length):
#                 inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
#                 _, curr_unpert_past, curr_all_hidden = model(
#                     past=curr_unpert_past,
#                     inputs_embeds=inputs_embeds
#                 )
#                 curr_hidden = curr_all_hidden[-1]
#                 new_accumulated_hidden = new_accumulated_hidden + torch.sum(
#                     curr_hidden, dim=1)
#
#             prediction = classifier(new_accumulated_hidden /
#                                     (curr_length + 1 + horizon_length))
#
#             label = torch.tensor(prediction.shape[0] * [class_label],
#                                  device=device,
#                                  dtype=torch.long)
#             discrim_loss = ce_loss(prediction, label)
#             if verbosity_level >= VERY_VERBOSE:
#                 print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
#             loss += discrim_loss
#             loss_list.append(discrim_loss)
#
#         kl_loss = torch.tensor(0.0)
#         # Рассчитаем расхождение Кульбака-Лейблера между вероятностными распределениями
#         # для токенов в возмущенной и невозмущенной модели
#         if kl_scale > 0.0:
#             # Берем распределение для невозмущенной модели
#             unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
#
#             # Нулевые вероятности делаем равными малой константе
#             unpert_probs = (
#                     unpert_probs + SMALL_CONST *
#                     (unpert_probs <= SMALL_CONST).float().to(device).detach()
#             )
#             correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
#                 device).detach()
#             corrected_probs = probs + correction.detach()
#
#             # Рассчитываем расхождение Кульбака-Лейблера по соответствующей формуле,
#             # берем с коэффициентом kl_scale
#             kl_loss = kl_scale * (
#                 (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
#             )
#             if verbosity_level >= VERY_VERBOSE:
#                 print(' kl_loss', kl_loss.data.cpu().numpy())
#             loss += kl_loss
#
#         loss_per_iter.append(loss.data.cpu().numpy())
#         if verbosity_level >= VERBOSE:
#             print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())
#
#         # compute gradients
#         # Обновляет тензор в поле .grad для curr_perturbation по цепи loss->bow_loss->bow_probs->
#         # ->probs->logits->all_logits->perturbed_past->curr_perturbation (аналогично для других вкладов в лосс)
#         loss.backward()
#
#         # calculate gradient norms
#         # grad_norms приходит в аргументах и возвращается в значениях,
#         # то есть обновляется в цикле по вызовам perturb_past, начиная с None
#
#         # Перед вычислением нормы градиент умножается на вычисленную ранее маску
#         # Берется норма матрицы по умолчанию, то есть норма Фробениуса
#         # (корень из суммы квадратов всех элементов матрицы)
#         if grad_norms is not None and loss_type == PPLM_BOW:
#             grad_norms = [
#                 torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
#                 for index, p_ in enumerate(curr_perturbation)
#             ]
#         else:
#             grad_norms = [
#                 (torch.norm(p_.grad * window_mask) + SMALL_CONST)
#                 for index, p_ in enumerate(curr_perturbation)
#             ]
#
#         # Нормализуем градиенты в curr_perturbation
#         grad = [
#             -stepsize *
#             (p_.grad * window_mask / grad_norms[
#                 index] ** gamma).data.cpu().numpy()
#             for index, p_ in enumerate(curr_perturbation)
#         ]
#
#         # Обновляем определенную в начале функции переменную grad_accumulator
#         grad_accumulator = list(map(add, grad, grad_accumulator))
#
#         # reset gradients, just to make sure
#         for p_ in curr_perturbation:
#             p_.grad.data.zero_()
#
#         # removing past from the graph
#         new_past = []
#         for p_ in past:
#             new_past.append(p_.detach())
#         past = new_past
#
#     # apply the accumulated perturbations to the past
#     grad_accumulator = [
#         to_var(torch.from_numpy(p_), requires_grad=True, device=device)
#         for p_ in grad_accumulator
#     ]
#     # Теперь pert_past - это H_t + \Delta{H_t} из статьи
#     pert_past = list(map(add, past, grad_accumulator))
#
#     return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter
