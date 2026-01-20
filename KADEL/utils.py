from torch.utils.data import TensorDataset
import numpy as np
import logging
import os
import random
import torch
import time
from tqdm import tqdm
from _utils import *
from preprocess_MCMD import get_sub_task_list

logger = logging.getLogger(__name__)


def load_and_cache_gen_data(
    args,
    filename,
    pool,
    tokenizer,
    split_tag,
    only_src=False,
    is_sample=False,
    setting=None,
    filter_none_scope=False,
    target_is_knwon_part=False,
    no_pos_no_segment=False,
    select_idx_list=None,
    get_idx_at_last=False,
):
    """
    Versione semplificata e robusta per il tuo scenario:
    - task = 'cmt_msg_gen'
    - multi_task = False
    - co_teaching = False

    Risolve il problema delle lunghezze diverse di target_ids / position_ids / segment_ids
    facendo padding/troncamento esplicito PRIMA di creare i tensori.
    """
    # cache the data into args.cache_path except it is sampled
    data_tag = "_all" if args.data_num == -1 else "_%d" % args.data_num
    cache_fn = "{}/{}.pt".format(
        args.cache_path, split_tag + ("_src" if only_src else "") + data_tag
    )
    os.makedirs(args.cache_path, exist_ok=True)

    # 1) Leggi gli examples
    examples = read_examples(filename, args.data_num, args.task, select_idx_list)

    # Filtra se richiesto
    if filter_none_scope:
        for idx, example in enumerate(examples):
            if len(get_content(example.target, content_type=["scope"])) == 0:
                examples[idx] = None
        examples = list(filter(None, examples))

    # Sampling opzionale (solo per debug/bleu)
    if is_sample:
        examples = random.sample(examples, min(5000, len(examples)))

    # Statistiche (solo logging)
    if split_tag == "train":
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)

    # Se esiste già la cache e non è sample, la ricarichiamo
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
        return examples, data

    # Altrimenti creiamo le features
    if is_sample:
        logger.info("Sample 5k data for computing bleu from %s", filename)
    else:
        logger.info("Create cache data into %s", cache_fn)

    # Per il tuo caso: multi_task = False, quindi sempre questo ramo
    tuple_examples = [
        (
            example,
            idx,
            tokenizer,
            args,
            split_tag,
            setting,
            filter_none_scope,
            target_is_knwon_part,
            no_pos_no_segment,
        )
        for idx, example in enumerate(examples)
    ]

    # Costruiamo sempre le features (sia per train che eval/test)
    features = pool.map(
        convert_examples_to_features,
        tqdm(tuple_examples, total=len(tuple_examples)),
    )

    # helper per il padding
    def pad_or_trunc(seq, max_len, pad_id=0):
        if len(seq) > max_len:
            return seq[:max_len]
        elif len(seq) < max_len:
            return seq + [pad_id] * (max_len - len(seq))
        else:
            return seq

    # SOURCE IDS (di solito sono già tutti della stessa lunghezza, ma non fa male)
    max_src_len = max(len(f.source_ids) for f in features)
    all_source_ids = torch.tensor(
        [pad_or_trunc(f.source_ids, max_src_len, pad_id=0) for f in features],
        dtype=torch.long,
    )

    # Se ci serve solo la sorgente (per BLEU su dev/test), fermiamoci qui
    if split_tag in ["dev", "valid", "test"] and (only_src or args.task != "cmt_msg_gen"):
        data = TensorDataset(all_source_ids)
        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(data, cache_fn)
        return examples, data

    # Da qui in poi: task = cmt_msg_gen, vogliamo anche target/position/segment
    if args.task == "cmt_msg_gen":
        # Lunghezze massime effettive nel dataset
        max_tgt_len = max(len(f.target_ids) for f in features)
        max_pos_len = max(len(f.position_ids) for f in features)
        max_seg_len = max(len(f.segment_ids) for f in features)

        # Facciamo comunque un cap sulle lunghezze massime per sicurezza
        # (lasciamole abbastanza alte per non massacrare i dati)
        max_tgt_len = min(max_tgt_len, args.max_target_length + 20)
        max_pos_len = min(max_pos_len, args.max_target_length + 20)
        max_seg_len = min(max_seg_len, args.max_target_length + 20)

        all_target_ids = torch.tensor(
            [
                pad_or_trunc(f.target_ids, max_tgt_len, pad_id=0)
                for f in features
            ],
            dtype=torch.long,
        )
        all_position_ids = torch.tensor(
            [
                pad_or_trunc(f.position_ids, max_pos_len, pad_id=0)
                for f in features
            ],
            dtype=torch.long,
        )
        all_segment_ids = torch.tensor(
            [
                pad_or_trunc(f.segment_ids, max_seg_len, pad_id=0)
                for f in features
            ],
            dtype=torch.long,
        )

        # Indici, se richiesti
        if get_idx_at_last:
            all_index = torch.tensor(
                [f.index for f in features],
                dtype=torch.long,
            )
        else:
            all_index = None

        # Co-teaching (nel tuo run è False, ma lo gestiamo lo stesso)
        if args.co_teaching:
            all_noise_or_not = torch.tensor(
                [f.noise_or_not for f in features],
                dtype=torch.bool,
            )
            if get_idx_at_last:
                data = TensorDataset(
                    all_source_ids,
                    all_target_ids,
                    all_position_ids,
                    all_segment_ids,
                    all_noise_or_not,
                    all_index,
                )
            else:
                data = TensorDataset(
                    all_source_ids,
                    all_target_ids,
                    all_position_ids,
                    all_segment_ids,
                    all_noise_or_not,
                )
        else:
            if get_idx_at_last:
                data = TensorDataset(
                    all_source_ids,
                    all_target_ids,
                    all_position_ids,
                    all_segment_ids,
                    all_index,
                )
            else:
                data = TensorDataset(
                    all_source_ids,
                    all_target_ids,
                    all_position_ids,
                    all_segment_ids,
                )
    else:
        # Per altri task (che tu non usi) manteniamo il comportamento minimale
        if split_tag == "test" or only_src:
            data = TensorDataset(all_source_ids)
        else:
            all_target_ids = torch.tensor(
                [f.target_ids for f in features],
                dtype=torch.long,
            )
            data = TensorDataset(all_source_ids, all_target_ids)

    # Salviamo in cache
    if args.local_rank in [-1, 0] and not is_sample:
        torch.save(data, cache_fn)

    return examples, data



def load_and_cache_clone_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + '_all' if args.data_num == -1 else '_%d' % args.data_num)
    examples = read_examples(filename, args.data_num, args.task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_clone_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_defect_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    examples = read_examples(filename, args.data_num, args.task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_multi_gen_data(args, pool, tokenizer, split_tag, only_src=False, is_sample=False, setting=None, filter_none_scope=False, target_is_knwon_part=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        examples_data_dict = torch.load(cache_fn)
    else:
        examples_data_dict = {}

        task_list = ['summarize', 'cmt_msg_gen', 'translate', 'refine', 'concode', 'defect']
        for task in task_list:
            if task == 'summarize':
                sub_tasks = ['ruby', 'javascript', 'go', 'python', 'java', 'php']
            elif task == 'cmt_msg_gen':
                sub_tasks = get_sub_task_list()
            elif task == 'translate':
                sub_tasks = ['java-cs', 'cs-java']
            elif task == 'refine':
                sub_tasks = ['small', 'medium']
            else:
                sub_tasks = ['none']
            args.task = task
            for sub_task in sub_tasks:
                args.sub_task = sub_task
                if task == 'summarize':
                    args.max_source_length = 256
                    args.max_target_length = 128
                elif task == 'cmt_msg_gen':
                    args.max_source_length = 200
                    args.max_target_length = 50
                elif task == 'translate':
                    args.max_source_length = 320
                    args.max_target_length = 256
                elif task == 'refine':
                    if sub_task == 'small':
                        args.max_source_length = 130
                        args.max_target_length = 120
                    else:
                        args.max_source_length = 240
                        args.max_target_length = 240
                elif task == 'concode':
                    args.max_source_length = 320
                    args.max_target_length = 150
                elif task == 'defect':
                    args.max_source_length = 512
                    args.max_target_length = 3  # as do not need to add lang ids

                filename = get_filenames(args.data_dir, args.task, args.sub_task, split_tag)
                examples = read_examples(filename, args.data_num, args.task)
                if is_sample:
                    examples = random.sample(examples, min(5000, len(examples)))
                if split_tag == 'train':
                    calc_stats(examples, tokenizer, is_tokenize=True)
                else:
                    calc_stats(examples)

                tuple_examples = [(example, idx, tokenizer, args, split_tag, setting, filter_none_scope, target_is_knwon_part) for idx, example in enumerate(examples)]
                if args.data_num == -1:
                    features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
                else:
                    features = [convert_examples_to_features(x) for x in tuple_examples]
                all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
                if only_src:
                    data = TensorDataset(all_source_ids)
                else:
                    pad_id = tokenizer.pad_token_id
                    max_tgt_len = max(len(f.target_ids) for f in features)
                    padded_targets = []
                    for f in features:
                        ids = list(f.target_ids)
                        if len(ids) < max_tgt_len:
                            ids = ids + [pad_id] * (max_tgt_len - len(ids))
                        else:
                            ids = ids[:max_tgt_len]
                        padded_targets.append(ids)

                    all_target_ids = torch.tensor(padded_targets, dtype=torch.long)
                    data = TensorDataset(all_source_ids, all_target_ids)

        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(examples_data_dict, cache_fn)
            logger.info("Save data into %s", cache_fn)
    return examples_data_dict


def get_filenames(data_root, task, sub_task, split=''):
    if task == 'concode':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.json'.format(data_dir)
        dev_fn = '{}/dev.json'.format(data_dir)
        test_fn = '{}/test.json'.format(data_dir)
    elif task == 'summarize':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    elif task == 'cmt_msg_gen':
        data_dir = os.path.join(data_root, task, sub_task)
        train_fn = os.path.join(data_dir, 'train.diff.txt')
        dev_fn = os.path.join(data_dir, 'valid.diff.txt')
        test_fn = os.path.join(data_dir, 'test.diff.txt')
    elif task == 'refine':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.buggy-fixed.buggy,{}/train.buggy-fixed.fixed'.format(data_dir, data_dir)
        dev_fn = '{}/valid.buggy-fixed.buggy,{}/valid.buggy-fixed.fixed'.format(data_dir, data_dir)
        test_fn = '{}/test.buggy-fixed.buggy,{}/test.buggy-fixed.fixed'.format(data_dir, data_dir)
    elif task == 'translate':
        data_dir = '{}/{}'.format(data_root, task)
        if sub_task == 'cs-java':
            train_fn = '{}/train.java-cs.txt.cs,{}/train.java-cs.txt.java'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.cs,{}/valid.java-cs.txt.java'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.cs,{}/test.java-cs.txt.java'.format(data_dir, data_dir)
        else:
            train_fn = '{}/train.java-cs.txt.java,{}/train.java-cs.txt.cs'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.java,{}/valid.java-cs.txt.cs'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.java,{}/test.java-cs.txt.cs'.format(data_dir, data_dir)
    elif task == 'clone':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.txt'.format(data_dir)
        dev_fn = '{}/valid.txt'.format(data_dir)
        test_fn = '{}/test.txt'.format(data_dir)
    elif task == 'defect':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    if split == 'train':
        return train_fn
    elif split == 'dev':
        return dev_fn
    elif split == 'test':
        return test_fn
    else:
        return train_fn, dev_fn, test_fn


def read_examples(filename, data_num, task, select_idx_list=None):
    read_example_dict = {
        'summarize': read_summarize_examples,
        'cmt_msg_gen': read_cmt_msg_examples,
        'refine': read_refine_examples,
        'translate': read_translate_examples,
        'concode': read_concode_examples,
        'clone': read_clone_examples,
        'defect': read_defect_examples,
    }
    return read_example_dict[task](filename, data_num, select_idx_list)


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)
