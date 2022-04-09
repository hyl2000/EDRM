import random
import sys

sys.path += ['./']
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, BertTokenizer, AutoConfig
# from accelerate import Accelerator
# from utils.evaluation import eval_f1, eval_all
# from utils.evaluation import f1_score
# from utils.io import write_file
import torch.nn.functional as F
import nltk
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import torch.nn as nn
import jieba
import copy
import torch
import os
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# wiki = json.load(open('dataset/wiki_small.json'))


def takeFirst(elem):
    return elem[0]


def f1_score(golden_response, response):
    # response是预测的，golden是真实的
    """
    calc_f1
    """
    response = jieba.lcut(response)
    golden_response = jieba.lcut(golden_response)
    golden_char_total = 0.0
    pred_char_total = 0.0
    hit_char_total = 0.0
    golden_response = "".join(golden_response)
    response = "".join(response)
    common = Counter(response) & Counter(golden_response)
    hit_char_total += sum(common.values())
    golden_char_total += len(golden_response)
    pred_char_total += len(response)
    p = hit_char_total / pred_char_total
    r = hit_char_total / golden_char_total
    if p != 0 or r != 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0
    return f1


'''
def get_best_match(query, candidates):
    return max([[f1_score(query.lower(), [line.lower()]), line] for line in candidates], key=lambda x: x[0])
'''


def get_best_match(query, candidates, pos_num):
    temp = [[f1_score(query.lower(), line.lower()), line] for line in candidates]
    temp.sort(key=takeFirst, reverse=True)
    return temp[0: pos_num]


def norm_text(text):
    return ' '.join(text.split()).strip().lower().replace(' ', '')


def load_data(file, pos_num):
    data = json.load(open(file, encoding='utf-8'))
    context = []
    knowledge = []
    response = []
    for session in tqdm(data):
        dialog_his = "<context>" + session['context']
        knowledge_pool = []
        for knowledge_ in session['knowledge']:
            knowledge_pool.append(norm_text(knowledge_))
        knowledge_pool.append('no_passages_used')
        pos = get_best_match(session['response'], knowledge_pool, pos_num)
        positive = []
        for (score, match_k) in pos:
            # if score > 0.5:
            positive.append(match_k)
        if len(positive) == 0:
            positive.append('no_knowledge_used')
        t_negative = set()
        for sentence in positive:
            for s in knowledge_pool:
                if f1_score(s.lower(), sentence) < 0.5:
                    t_negative.add(s)
        negative = list(t_negative)
        # negative = [s for s in knowledge_pool if f1_score(s.lower(), [positive.lower()]) < 0.5]
        knowledge.append(positive + negative)
        context.append(copy.deepcopy(dialog_his))
        response.append(session['response'].lower())
    return context, knowledge, response, data


def load_app_data(file, pos_num):
    data = json.load(open(file, encoding='utf-8'))
    context = []
    knowledge = []
    response = []
    for session in tqdm(data):
        dialog_his = "<context>" + session['context']
        knowledge_pool = []
        for knowledge_ in session['knowledge']:
            knowledge_t = norm_text(knowledge_)
            if len(knowledge_t) == 0:
                continue
            knowledge_pool.append(knowledge_t)
        knowledge_pool.append('no_knowledge_used')
        pos = get_best_match(session['response'], knowledge_pool, pos_num)
        positive = []
        for (score, match_k) in pos:
            # if score > 0.5:
            positive.append(match_k)
        if len(positive) == 0:
            positive .append('no_knowledge_used')
        t_negative = set()
        for sentence in positive:
            for s in knowledge_pool:
                if f1_score(s.lower(), sentence) < 0.5:
                    t_negative.add(s)
        negative = list(t_negative)
        # negative = [s for s in knowledge_pool if f1_score(s.lower(), [positive.lower()]) < 0.5]
        knowledge.append(positive + negative)
        context.append(dialog_his)
        response.append(session['response'].lower())
    return context, knowledge, response


class RankData(Dataset):
    def __init__(self, context, knowledge, response, tokenizer, context_len=256, response_len=128, neg_num=4, pos_num=1,
                 pad_none=True):
        super(Dataset, self).__init__()
        self.context = context
        self.knowledge = knowledge
        self.response = response
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.response_len = response_len
        self.neg_num = neg_num
        self.pos_num = pos_num
        self.pad_none = pad_none

    def __getitem__(self, index):
        context = self.context[index]
        response = self.response[index]
        knowledge = self.knowledge[index]

        # topic = self.tokenizer.encode(context[0])
        # his = topic + self.tokenizer.encode(' '.join(context[1:]))[-(self.context_len - len(topic)):]
        his = self.tokenizer.encode(' '.join(context))

        neg = knowledge[self.pos_num:]
        if self.pad_none:
            random.shuffle(neg)
            neg = neg + ['<none>'] * (self.neg_num - len(neg))
        neg = neg[:self.neg_num]
        knowledge = knowledge[0:self.pos_num] + neg
        response = self.tokenizer.encode(response, truncation=True, max_length=self.response_len)[:self.response_len]
        batch_context = []
        for k in knowledge:
            context = torch.tensor(his + self.tokenizer.encode(
                ' <knowledge> ' + norm_text(k), truncation=True, max_length=self.context_len))[:self.context_len].cuda()
            batch_context.append(context)
        return batch_context, response

    def __len__(self):
        return len(self.context)

    @staticmethod
    def collate_fn(data):
        batch_context, response = zip(*data)
        batch_context = sum(batch_context, [])
        context = pad_sequence(batch_context, batch_first=True, padding_value=0)
        return {
            'input_ids': context,
            'attention_mask': context.ne(0),
        }


def main():
    # accelerator = Accelerator()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    epochs = 20
    batch_size = 8
    neg_num = 9
    pos_num = 3
    model_name = './mengzi-bert-base'
    ckpt_name = 'app-bert-rank'
    print(model_name, ckpt_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    context, knowledge, response = load_app_data('data/train_new.json', pos_num)
    dataset = RankData(context, knowledge, response, tokenizer, context_len=256, response_len=128, neg_num=neg_num, pos_num=pos_num)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, shuffle=True)

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer.add_tokens(['<context>', '<knowledge>', '<none>'])
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=len(data_loader), num_training_steps=epochs * len(data_loader))
    # scheduler = accelerator.prepare(scheduler)

    for epoch in range(epochs):
        # accelerator.wait_for_everyone()
        # accelerator.print(f'train epoch={epoch}')
        tk0 = tqdm(data_loader, total=len(data_loader))
        losses = []
        for batch in tk0:
            output = model(**batch)
            logits = output.logits.view(-1, pos_num+neg_num)
            loss = F.cross_entropy(logits, torch.zeros((logits.size(0),)).long().cuda())
            # print(output.logits.size())
            # loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            # accelerator.backward(loss)
            # accelerator.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            tk0.set_postfix(loss=sum(losses) / len(losses))

        os.makedirs(f'ckpt/{ckpt_name}', exist_ok=True)
        # if accelerator.is_local_main_process:
        #     accelerator.save(accelerator.unwrap_model(model).state_dict(), f'ckpt/{ckpt_name}/{epoch}.pt')
        torch.save({'state_dict': model.state_dict(), 'optimizer_state': optimizer.state_dict()},
                   f'ckpt/{ckpt_name}/{epoch}.pt')


def lower(text):
    if isinstance(text, str):
        text = text.strip().lower()
        text = ' '.join(nltk.word_tokenize(text))
        return text.strip()
    return [lower(item) for item in text]


def test():
    batch_size = 4
    model_name = './mengzi-bert-base'
    ckpt_name = 'app-bert-rank'

    tokenizer = BertTokenizer.from_pretrained(model_name)
    context, knowledge, response, data = load_data('data/test_new.json', 1)
    dataset = RankData(context, knowledge, response, tokenizer, context_len=256, response_len=128, neg_num=63, pos_num=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, shuffle=False)

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer.add_tokens(['<context>', '<knowledge>', '<none>'])
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()

    for epoch in range(0, 20):
        if not os.path.exists(f'ckpt/{ckpt_name}/{epoch}.pt'):
            continue
        print(f'Test ckpt/{ckpt_name}/{epoch}.pt')
        # model.load_state_dict(torch.load(f'ckpt/{ckpt_name}/{epoch}.pt'))
        checkpoint = torch.load(f'ckpt/{ckpt_name}/{epoch}.pt')
        os.makedirs(f'ckpt/result', exist_ok=True)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.cuda()
        tk0 = tqdm(data_loader, total=len(data_loader))
        acc = []
        model.eval()
        now_sample = 0
        with torch.no_grad():
            for batch in tk0:
                batch = {k: v.cuda() for k, v in batch.items()}
                output = model(**batch)
                logits = output.logits.view(-1, 64)
                num, _ = logits.size()
                acc.append((logits.argmax(-1) == 0).float().mean().item())  # 可以改一下
                sorted, indices = torch.sort(logits, dim=1, descending=True)
                indices = indices.tolist()
                for i in range(0, num):
                    out_knowledge = []
                    for j in indices[i]:
                        if j < len(knowledge[now_sample]):
                            out_knowledge.append(knowledge[now_sample][j])
                    data[now_sample]['knowledge'] = ' | '.join(out_knowledge)
                    now_sample += 1
                # print((logits.argmax(-1) == 0).float().mean().item())
                tk0.set_postfix(acc=sum(acc) / len(acc))
        print(sum(acc) / len(acc))
        json.dump(data, open(f'ckpt/result/{epoch}.json', 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    # main()
    test()
    # inference()
