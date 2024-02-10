import argparse, json, math, os, sys, random, logging
from collections import defaultdict as ddict, Counter
from itertools import chain

import numpy as np
import pandas as pd
from pickle5 import pickle
from tqdm import tqdm

import torch
import datasets
from transformers import AutoTokenizer
from datasets import load_dataset


sys.path.append(os.path.join(sys.path[0], '..'))
from src.utils.data import dataset_info, eraser_datasets, data_keys
from src.utils.eraser.utils import annotations_from_jsonl, load_documents
from src.utils.eraser.data_utils import (
    bert_tokenize_doc,
    bert_intern_doc,
    bert_intern_annotation,
    annotations_to_evidence_identification,
    annotations_to_evidence_token_identification,
)

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def update_dataset_dict(
    idx, dataset_dict, input_ids, rationale, max_length, actual_max_length, tokenizer, interned_annotations, classes, expl_ids
):
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    rationale = [0] + rationale + [0]
    assert len(input_ids) == len(rationale)
    num_tokens = len(input_ids)
    if num_tokens > actual_max_length:
        actual_max_length = num_tokens

    num_pad_tokens = max_length - num_tokens
    assert num_pad_tokens >= 0

    input_ids += [tokenizer.pad_token_id] * num_pad_tokens
    attention_mask = [1] * num_tokens + [0] * num_pad_tokens
    rationale += [0] * num_pad_tokens

    inv_rationale = [1.0-x for x in rationale]
    rand_rationale = list(np.random.randn(max_length))

    has_rationale = int(sum(rationale) > 0)
    if has_rationale == 0:
        raise ValueError('empty rationale')

    label = classes.index(interned_annotations[idx].classification)

    # Explanations ################################################
    expl_ids = [tokenizer.cls_token_id] + expl_ids + [tokenizer.sep_token_id]

    num_tokens = len(expl_ids)
    if num_tokens > actual_max_length:
        actual_max_length = num_tokens

    num_pad_tokens = max_length - num_tokens
    assert num_pad_tokens >= 0

    input_ids += [tokenizer.pad_token_id] * num_pad_tokens
    expl_mask = [1] * num_tokens + [0] * num_pad_tokens

    dataset_dict['item_idx'].append(idx)
    dataset_dict['input_ids'].append(input_ids)
    dataset_dict['attention_mask'].append(attention_mask)
    dataset_dict['rationale'].append(rationale)
    dataset_dict['inv_rationale'].append(inv_rationale)
    dataset_dict['rand_rationale'].append(rand_rationale)
    dataset_dict['has_rationale'].append(has_rationale)
    dataset_dict['label'].append(label)

    dataset_dict['expl_ids'].append(expl_ids)
    dataset_dict['expl_mask'].append(expl_mask)

    return dataset_dict, actual_max_length

def align_rationale_with_tokens(input_ids, raw_tokens, raw_rationale, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    rationale = []
    j = 0
    cur_token = tokens[j]

    for i in range(len(raw_tokens)):
        cur_raw_token = raw_tokens[i]
        cur_raw_rationale = raw_rationale[i]
        cur_reconstructed_raw_token = ''

        while len(cur_raw_token) > 0:
            for char in cur_token:
                if char == cur_raw_token[0]:
                    cur_raw_token = cur_raw_token[1:]
                    cur_reconstructed_raw_token += char

            rationale.append(cur_raw_rationale)
            j += 1
            cur_token = tokens[j] if j < len(tokens) else None

        assert cur_reconstructed_raw_token == raw_tokens[i]

    return rationale

def load_dataset(data_path):
    dataset_dict = ddict(list)
    for key in tqdm(data_keys, desc=f'Loading {args.split} dataset'):
        with open(os.path.join(data_path, f'{key}.pkl'), 'rb') as f:
            dataset_dict[key] = pickle.load(f)
    return dataset_dict

def save_dataset(data_path, dataset_dict, split, num_samples, seed):
    for key in tqdm(data_keys, desc=f'Saving {split} dataset'):
        filename = f'{key}.pkl' if num_samples is None else f'{key}_{num_samples}_{seed}.pkl'
        with open(os.path.join(data_path, filename), 'wb') as f:
            pickle.dump(dataset_dict[key], f)

def main(args):
    set_random_seed(args.seed)

    assert args.split is not None and args.arch is not None
    assert args.num_samples is None or args.num_samples >= 1

    split, num_examples = dataset_info[args.dataset][args.split]
    if args.num_samples is not None:
        assert args.num_samples < num_examples

    num_classes = dataset_info[args.dataset]['num_classes']
    max_length = dataset_info[args.dataset]['max_length'][args.arch]
    num_special_tokens = dataset_info[args.dataset]['num_special_tokens']
    tokenizer = AutoTokenizer.from_pretrained(args.arch)

    data_path = os.path.join(args.data_dir, args.dataset, args.arch, args.split)
    classes = dataset_info[args.dataset]['classes']
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if args.dataset in eraser_datasets:
        eraser_path = os.path.join(args.data_dir, 'eraser', args.dataset)
        documents_path = os.path.join(args.data_dir, args.dataset, args.arch, 'documents.pkl')
        documents = load_documents(eraser_path) # -> Dict[str, List[List[str]]]
        logger.info(f'Loaded {len(documents)} documents')

        if args.dataset == 'cose':
            logger.info(f'Loading CoS-E v1.11 {args.split}')
            dataset = load_dataset('cose', 'v1.11')

        #     if args.split == 'train':
        #         dataset = dataset['train'].train_test_split(test_size=0.11)['train']
        #     elif args.split == 'dev':
        #         dataset = dataset['train'].train_test_split(test_size=0.11)['test']
        #     elif args.split == 'test':
        #         dataset = dataset['test']
        #     else:
        #         raise NotImplementedError
        # else:
            # raise NotImplementedError

        if os.path.exists(documents_path):
            # pass
            logger.info(f'Loading processed documents from {documents_path}')
            (interned_documents, interned_document_token_slices) = torch.load(documents_path)
            logger.info(f'Loaded {len(interned_documents)} processed documents')
        else:
            logger.info(f'Processing documents')
            special_token_map = {
                'SEP': [tokenizer.sep_token_id],
                '[SEP]': [tokenizer.sep_token_id],
                '[sep]': [tokenizer.sep_token_id],
                'UNK': [tokenizer.unk_token_id],
                '[UNK]': [tokenizer.unk_token_id],
                '[unk]': [tokenizer.unk_token_id],
                'PAD': [tokenizer.unk_token_id],
                '[PAD]': [tokenizer.unk_token_id],
                '[pad]': [tokenizer.unk_token_id],
            }
            interned_documents = {}
            interned_document_token_slices = {}
            
            for d, doc in tqdm(documents.items(), desc='Processing documents'):
            # for d, doc in tqdm(zip(dataset['id'], dataset['question']), desc='Processing documents'):
                tokenized, w_slices = bert_tokenize_doc(doc, tokenizer, special_token_map=special_token_map)
                interned_documents[d] = bert_intern_doc(tokenized, tokenizer, special_token_map=special_token_map)
                interned_document_token_slices[d] = w_slices
            logger.info(f'Saving processed documents to {documents_path}')
            torch.save((interned_documents, interned_document_token_slices), documents_path)
            sys.exit()

        annotations_path = os.path.join(eraser_path, f'{split}.jsonl')
        annotations = annotations_from_jsonl(annotations_path, dataset)
        interned_annotations = bert_intern_annotation(annotations, tokenizer)
        evidence_data = annotations_to_evidence_token_identification(annotations, documents, interned_documents, interned_document_token_slices)
        assert len(evidence_data) == num_examples

    missing_data_keys = [x for x in data_keys if not os.path.exists(os.path.join(data_path, f'{x}.pkl'))]
    if missing_data_keys:
        dataset_dict = ddict(list)
        actual_max_length = 0
            
        q_marker = tokenizer('Q:', add_special_tokens=False)['input_ids']
        a_marker = tokenizer('A:', add_special_tokens=False)['input_ids']
        for idx, (instance_id, instance_evidence) in tqdm(enumerate(evidence_data.items()), desc=f'Building {args.split} dataset', total=num_examples):
            # instance_docs = ddict(dict)
            assert len(instance_evidence) == 1
            doc = interned_documents[instance_id]
            evidence_sentences = instance_evidence[instance_id]

            question = list(chain.from_iterable(doc))
            question_rationale = list(chain.from_iterable([x.kls for x in evidence_sentences]))
            answers = evidence_sentences[0].query.split(' [sep] ')
            answer_ids = [tokenizer(x, add_special_tokens=False)['input_ids'] for x in answers]

            # Explanations (TODO: put breakpointy to see what to do)
            explanations = None

            input_ids, attention_mask, rationale, inv_rationale, rand_rationale, has_rationale, expl_ids, expl_mask = [], [], [], [], [], [], [], []
            for answer in answer_ids:
                cur_input_ids = [tokenizer.cls_token_id] + q_marker + question + [tokenizer.sep_token_id] + a_marker + answer + [tokenizer.sep_token_id]

                num_tokens = len(cur_input_ids)
                if num_tokens > actual_max_length:
                    actual_max_length = num_tokens
                num_pad_tokens = max_length - num_tokens
                assert num_pad_tokens >= 0

                cur_input_ids += [tokenizer.pad_token_id] * num_pad_tokens
                input_ids.append(cur_input_ids)

                cur_attention_mask = [1] * num_tokens + [0] * num_pad_tokens
                attention_mask.append(cur_attention_mask)

                cur_rationale = [0] + [0]*len(q_marker) + question_rationale + [0] + [0]*len(a_marker) + [0]*len(answer) + [0]
                cur_rationale += [0] * num_pad_tokens
                assert len(cur_input_ids) == len(cur_rationale)
                rationale.append(cur_rationale)

                inv_rationale.append([1.0-x for x in cur_rationale])
                rand_rationale.append(list(np.random.randn(max_length)))

                cur_has_rationale = int(sum(cur_rationale) > 0)
                if cur_has_rationale == 0:
                    raise ValueError('empty rationale')
                has_rationale.append(cur_has_rationale)


                # Explanation ##############
                cur_expl_ids = [tokenizer.cls_token_id] + explanations + [tokenizer.sep_token_id] + [tokenizer.sep_token_id]

                num_tokens = len(cur_expl_ids)
                if num_tokens > actual_max_length:
                    actual_max_length = num_tokens
                num_pad_tokens = max_length - num_tokens
                assert num_pad_tokens >= 0

                cur_expl_ids += [tokenizer.pad_token_id] * num_pad_tokens
                expl_ids.append(cur_expl_ids)

                cur_expl_mask = [1] * num_tokens + [0] * num_pad_tokens
                expl_mask.append(cur_expl_mask)

            label = classes.index(interned_annotations[idx].classification)

            dataset_dict['item_idx'].append(idx)
            dataset_dict['input_ids'].append(input_ids)
            dataset_dict['attention_mask'].append(attention_mask)
            dataset_dict['rationale'].append(rationale)
            dataset_dict['inv_rationale'].append(inv_rationale)
            dataset_dict['rand_rationale'].append(rand_rationale)
            dataset_dict['has_rationale'].append(has_rationale)
            dataset_dict['label'].append(label)

            dataset_dict['expl_ids'].append(expl_ids)
            dataset_dict['expl_mask'].append(expl_mask)

        print(f'Actual max length: {actual_max_length}')

    else:
        dataset_dict = load_dataset(data_path)

    save_dataset(data_path, dataset_dict, args.split, args.num_samples, args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing')
    parser.add_argument('--data_dir', type=str, default='../data/', help='Root directory for datasets')
    parser.add_argument('--dataset', type=str,
                        choices=['cose', 'esnli', 'movies', 'multirc', 'sst', 'amazon', 'yelp', 'stf', 'olid', 'irony'])
    parser.add_argument('--arch', type=str, default='google/bigbird-roberta-base', choices=['google/bigbird-roberta-base', 'bert-base-uncased'])
    parser.add_argument('--split', type=str, help='Dataset split', choices=['train', 'dev', 'test'])
    parser.add_argument('--num_samples', type=int, default=None, help='Number of examples to sample. None means all available examples are used.')
    parser.add_argument('--pct_train_rationales', type=float, default=None, help='Percentage of train examples to provide gold rationales for. None means all available train examples are used.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    main(args)
