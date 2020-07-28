import typing as tp

import torch
import torch.nn as nn


def compute_bow_loss(probs: torch.Tensor,
                     one_hot_bow: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss where target is BOW indexes in the vocab.

    :param probs: probability distribution over vocabulary from model
    :param one_hot_bow: one-hot representation of words from BOW
    :return: loss - computed CE loss
    """
    bow_probs = torch.mm(probs, torch.t(one_hot_bow))
    loss = -torch.log(torch.sum(bow_probs))
    return loss


def compute_discriminator_loss(model: nn.Module,
                               classifier: nn.Module,
                               probs: torch.Tensor,
                               horizon_length: int,
                               unpert_past: tp.Tuple[torch.Tensor, ...],
                               accum_hidden: torch.Tensor,
                               cur_length: int,
                               device: str,
                               class_label: int) -> torch.Tensor:
    """
    Compute cross-entropy loss for Discriminator model (p(a|x))

    :param model: GPT2 or another model
    :param classifier: Discriminator for computing p(a|x)
    :param probs: probability distribution over vocabulary from model
    :param horizon_length:
    :param unpert_past: Key, Values from previous timesteps ('history')
    :param accum_hidden: tensor that aggregates previous hidden states
    :param cur_length: current length of generated sequence
    :param device: cuda or cpu
    :param class_label: class index
    :return: loss - computed CE loss
    """

    ce_loss = nn.CrossEntropyLoss()
    cur_probs = torch.unsqueeze(probs, dim=1)
    cur_unpert_past = unpert_past
    # Get weights of embeddings from model
    model_embs = model.resize_token_embeddings().weight.data
    new_accum_hidden = accum_hidden
    for _ in range(horizon_length):
        # Get weighted sum of all model embeddings w.r.t word probs
        inputs_embeds = torch.matmul(cur_probs, model_embs)
        _, cur_unpert_past, cur_all_hidden = model(
            past=cur_unpert_past,
            inputs_embeds=inputs_embeds
        )
        cur_hidden = cur_all_hidden[-1]
        new_accum_hidden = new_accum_hidden + torch.sum(cur_hidden, dim=1)

    new_accum_hidden /= cur_length + horizon_length + 1
    prediction = classifier(new_accum_hidden)
    label = torch.tensor(prediction.shape[0] * [class_label],
                         device=device,
                         dtype=torch.long)
    loss = ce_loss(prediction, label)
    return loss


def compute_kl_loss(p_dist: torch.Tensor,
                    q_dist: torch.Tensor,
                    scale: float = 1.0) -> torch.Tensor:
    """
    Compute Kullback–Leibler divergence with P and Q distributions.

    :param p_dist: first distribution
    :param q_dist: second distribution
    :param scale: KL coefficient
    :return: loss - computed KL loss
    """
    logits = (p_dist / q_dist).log()
    loss = (p_dist * logits).sum()
    loss *= scale
    return loss
