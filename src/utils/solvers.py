import torch
from torch import Tensor
from typing import Callable


def top_k_percent(attrs: Tensor, k: int, min_val: float = -1e10) -> Tensor:
    attn_mask = attrs.bool().int()
    num_tokens = torch.sum(attn_mask, dim=1) - 1  # don't include CLS token when computing num_tokens
    num_highlight_tokens = torch.round(num_tokens * k / 100)
    ones = torch.ones_like(num_highlight_tokens)
    num_highlight_tokens = torch.maximum(num_highlight_tokens, ones).long()

    attrs = attrs + (1 - attn_mask) * min_val  # ignore pad tokens when computing sorted_attrs_indices
    attrs[:, 0] = min_val  # don't include CLS token when computing sorted_attrs_indices

    expls = []
    for i in range(len(attrs)):
        scores, indices = torch.topk(attrs[i, :], num_highlight_tokens[i].item(), sorted=True)
        threshold = scores[-1]

        expl = (attrs[i:i + 1, :] >= threshold).float()
        expl[:, 0] = 1  # always treat CLS token as positive token

        expls.append(expl)

    return torch.cat(expls)


# #####################################################################################################
# OLD VERSION - DOESN'T WORK (NOT DIFFERENTIABLE WITH IMLE)
# #####################################################################################################
#
# def top_k_percent(attrs: Tensor, k: int, min_val: float = -1e10) -> Tensor:
#     attn_mask = attrs.bool().int()
#     num_tokens = torch.sum(attn_mask, dim=1) - 1  # don't include CLS token when computing num_tokens
#     num_highlight_tokens = torch.round(num_tokens * k / 100)
#     ones = torch.ones_like(num_highlight_tokens)
#     num_highlight_tokens = torch.maximum(num_highlight_tokens, ones).long()
#
#     attrs = attrs + (1 - attn_mask) * min_val  # ignore pad tokens when computing sorted_attrs_indices
#     attrs[:, 0] = min_val  # don't include CLS token when computing sorted_attrs_indices
#     sorted_attrs_indices = torch.argsort(attrs, dim=1, descending=True)
#
#     expl = torch.zeros_like(attn_mask).long()
#     for i in range(len(attrs)):
#         salient_indices = sorted_attrs_indices[i][:num_highlight_tokens[i]]
#         expl[i, salient_indices] = 1
#     expl[:, 0] = 1  # always treat CLS token as positive token
#
#     return expl
