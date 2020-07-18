from torch import mm as torch_mm, t as torch_t, log as torch_log, sum as torch_sum, long as torch_long
from torch import unsqueeze, matmul, tensor
from torch.nn import CrossEntropyLoss


# Для метода мешка слов
def bag_of_words_loss(probs, one_hot_bow):
    # Получаем вектор с вероятностями для всех токенов в мешке слов
    bow_probs = torch_mm(probs, torch_t(one_hot_bow))  # Напр. (1, 50257) * (18, 50257).T = (1, 18)
    # Используем минус логарифм суммы вероятностей как лосс
    bow_loss = -torch_log(torch_sum(bow_probs))
    return bow_loss


def discriminator_loss(model,
                       classifier,
                       probs,
                       horizon_length,
                       unpert_past,
                       accum_hidden,
                       curr_length,
                       device,
                       class_label):
    ce_loss = CrossEntropyLoss()
    curr_probs = unsqueeze(probs, dim=1)
    # Возвращаем ссылку на принадлежащий модели модуль Embeddings для входных токенов
    wte = model.resize_token_embeddings()
    new_accumulated_hidden = accum_hidden
    for _ in range(horizon_length):
        # curr_probs - [1, 1, 50257]
        # wte.weight.data - [50257, 1024]
        # inputs_embeds - [1, 1, 1024] - взвешенная с вероятностями почленная сумма эмбеддингов всех токенов словаря
        inputs_embeds = matmul(curr_probs, wte.weight.data)
        _, curr_unpert_past, curr_all_hidden = model(
            past=unpert_past,
            inputs_embeds=inputs_embeds
        )
        curr_hidden = curr_all_hidden[-1]
        new_accumulated_hidden = new_accumulated_hidden + torch_sum(
            curr_hidden, dim=1)

    prediction = classifier(new_accumulated_hidden /
                            (curr_length + 1 + horizon_length))

    label = tensor(prediction.shape[0] * [class_label],
                   device=device,
                   dtype=torch_long)
    discrim_loss = ce_loss(prediction, label)
    return discrim_loss


def kullback_leibler_loss(perturbed_distrib, unpert_distrib, kl_scale):
    # Рассчитываем расхождение Кульбака-Лейблера по соответствующей формуле,
    # берем с коэффициентом kl_scale
    kl_loss = kl_scale * (
        (perturbed_distrib * (perturbed_distrib / unpert_distrib).log()).sum()
    )
    return kl_loss
