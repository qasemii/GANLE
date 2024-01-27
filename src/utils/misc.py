import torch


def clear_cache():
    torch.cuda.empty_cache()

def is_subsequence(a, b):
    # Check if list a is a subsequence if list b
    return any(a == b[i:i + len(a)] for i in range(len(b) - len(a) + 1))


###############################################################
###############################################################
###############################################################


ret_dict['attrs'] = attrs.detach()
ret_dict['has_rationale'] = has_rationale.detach()
ret_dict['rationale'] = rationale.detach()
ret_dict['attn_mask'] = attn_mask.detach()

ret_dict = {'attrs': None, 'rationale': None, 'attn_mask': None, 'has_rationale': None}
for i, output in enumerate(outputs):
    if i == 0:
        ret_dict['attrs'] = output['attrs']
        ret_dict['rationale'] = output['rationale']
        ret_dict['attn_mask'] = output['attn_mask']
        ret_dict['has_rationale'] = output['has_rationale']
    else:
        ret_dict['attrs'] = torch.cat((ret_dict['attrs'], output['attrs']))
        ret_dict['rationale'] = torch.cat((ret_dict['rationale'], output['rationale']))
        ret_dict['attn_mask'] = torch.cat((ret_dict['attn_mask'], output['attn_mask']))
        ret_dict['has_rationale'] = torch.cat((ret_dict['has_rationale'], output['has_rationale']))

from sklearn.metrics import f1_score

k = 10
highlights = torch.zeros_like(ret_dict['attrs'])
for i, attrs in enumerate(ret_dict['attrs']):
    max_len = torch.nonzero(attrs)[-1].item() + 1
    topk = int(k * max_len / 100)
    _, idx = torch.topk(attrs, topk)
    for j in idx.tolist():
        highlights[i][j] = 1

# suff_metric = calc_suff(logits_dict['task'], suff_logits, suff_targets, False)
# comp_metric = calc_comp(logits_dict['task'], comp_logits, comp_targets, False)
plaus_metrics = calc_plaus(ret_dict['rationale'], highlights, ret_dict['attn_mask'], ret_dict['has_rationale'])

# print('sufficiency -------------> ', suff_metric)
# print('comprehensiveness -------------> ', comp_metric)
print('plausibility -------------> ', plaus_metrics)


###############################################################
###############################################################
###############################################################


ret_dict['attrs'] = attrs.detach()
ret_dict['has_rationale'] = has_rationale.detach()
ret_dict['rationale'] = rationale.detach()
ret_dict['attn_mask'] = attn_mask.detach()
ret_dict['suff_logits'] = suff_logits.detach()
ret_dict['comp_logits'] = comp_logits.detach()

new_outputs = []
for i, output in enumerate(outputs):
    temp = {}
    true_preds = torch.argmax(output['logits'], dim=1) == output['targets']
    for key in ['logits', 'targets', 'suff_logits', 'comp_logits', 'attrs', 'rationale', 'attn_mask']:
        temp[key] = output[key][true_preds]
    temp['has_rationale'] = output['has_rationale']
    new_outputs.append(temp)

ret_dict = {}
for i, output in enumerate(new_outputs):
    if i == 0:
        for key in output.keys():
            ret_dict[key] = output[key]
    else:
        for key in output.keys():
            ret_dict[key] = torch.cat((ret_dict[key], output[key]))

suff = calc_aopc(torch.stack([calc_suff(ret_dict['logits'], ret_dict['suff_logits'], ret_dict['targets'], False)]))
comp = calc_aopc(torch.stack([calc_comp(ret_dict['logits'], ret_dict['comp_logits'], ret_dict['targets'], False)]))
plaus = calc_plaus(ret_dict['rationale'], ret_dict['attrs'], ret_dict['attn_mask'], ret_dict['has_rationale'])

print('suff ---->', suff)
print('comp ---->', comp)
print('plaus ---->', plaus)





