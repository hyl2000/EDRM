from torch import optim
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import argparse
import os
from Utils import *
import torch
from tqdm import tqdm, trange
import json
from transformers import AdamW, Adafactor, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import sys
# cudaid = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaid)

base_output_path = 'output/'

embedding_size = 300
hidden_size = 512
knowledge_len = 100
min_vocab_freq = 50


class Data(Dataset):
    def __init__(self, data, tokenizer, context_len=256, response_len=128, inputs_len=256, goal_len=128):
        super(Dataset, self).__init__()
        self.data = data
        self.response = [item['response'] for item in data]
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.response_len = response_len
        self.inputs_len = inputs_len
        self.goal_len = goal_len

    def __getitem__(self, index):
        item = self.data[index]
        knowledge = ' <knowledge> ' + item['knowledge']
        goal = ' <goal> ' + item['goal']
        if len(item['emotion']) == 0:
            emotion_true = 'None'
        else:
            emotion_true = item['emotion'][-1]
        if len(item['emotion']) > 1:
            emotion = ' <emotion> ' + '-'.join(item['emotion'][:-1])
        else:
            emotion = ' <emotion> ' + 'None'
        context = (' <context> ' + item['context']).strip().replace("\n", "")
        response = (emotion_true + ' <emotion> ' + item['response']).strip().replace("\n", "")
        goal = self.tokenizer.encode(goal, truncation=True, max_length=self.goal_len)
        emo = self.tokenizer.encode(emotion, truncation=True, max_length=self.goal_len)
        context = self.tokenizer.encode(context, truncation=True, max_length=self.context_len)
        knowledge = self.tokenizer.encode(knowledge)[:(self.inputs_len - len(context) - len(goal))]
        inputs = knowledge + goal + context + emo
        target = self.tokenizer.encode(response, truncation=True, max_length=self.response_len)
        return torch.tensor(inputs), torch.tensor(target)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(data):
        context, response = zip(*data)
        context = pad_sequence(context, batch_first=True, padding_value=0)
        return {
            'input_ids': context,
            'attention_mask': context.ne(0),
            'labels': pad_sequence(response, batch_first=True, padding_value=-100),
            'response': response
        }


def train(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    batch_size = 8

    output_path = base_output_path
    data_path = 'data/'
    train_path = data_path + "train_emotion.json"
    print('go...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(['<knowledge>', '<goal>', '<context>', '<emotion>', '认同', '不认同', '开心', '伤心', '惊讶', '好奇', '中立'])

    train_dataset = Data(json.load(open(train_path, encoding='utf-8')), tokenizer, context_len=256, response_len=128, inputs_len=512 + 256, goal_len=128)
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size=batch_size, shuffle=True, num_workers=4)
    print('build data done')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()
    model_optimizer = AdamW(model.parameters(), lr=5e-4)
    if args.load_epoch != 0:
        file = output_path + 'model/' + str(args.load_epoch) + '.pkl'
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint["state_dict"])
        model_optimizer.load_state_dict(checkpoint["optimizer_state"])
        args.load_epoch += 1
    print('build model done')

    print('start training main model...')
    for i in range(args.load_epoch, args.max_epoch):
        print('#', i)
        start_time = time.time()
        losses = []
        for j, data in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda
            output = model(input_ids=data['input_ids'], attention_mask=data['attention_mask'], labels=data['labels'])
            loss = output.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            model_optimizer.step()
            model_optimizer.zero_grad()
            torch.cuda.empty_cache()
            losses.append(loss.item())
            if j >= 0 and j % 100 == 0:
                elapsed_time = time.time() - start_time
                print('Method', 'mle_train', 'Epoch', i, 'Batch ', j, 'Loss ', loss, 'Time ', elapsed_time)
                sys.stdout.flush()
        sys.stdout.flush()

        serialize(model_optimizer, model, i, output_path=output_path)


def test(args, beam_width):
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    batch_size = 4

    output_path = base_output_path
    data_path = 'data/'
    # model_name = 't5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(['<knowledge>', '<goal>', '<context>', '<emotion>', '认同', '不认同', '开心', '伤心', '惊讶', '好奇', '中立'])

    test_dataset = Data(json.load(open(data_path + "test_emotion.json", encoding='utf-8')), tokenizer, context_len=256, response_len=128, inputs_len=512 + 256, goal_len=128)

    for i in range(args.max_epoch):
        print('epoch', i)
        file = output_path + 'model/' + str(i) + '.pkl'
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn,
                                 num_workers=0)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.cuda()
        output_text_collect = []
        response = []
        emo_total = 0
        emo_count = 0
        for k, batch in enumerate(test_loader, 0):
            output = model.generate(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                max_length=128,
                no_repeat_ngram_size=3,
                num_beams=beam_width
            )
            output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
            res = tokenizer.batch_decode(batch['response'], skip_special_tokens=True)
            for j in range(len(output_text)):
                emo = output_text[j].split('<emotion>')[0].strip()
                # output_text_ = output_text[j].split('<emotion>')[1].strip()
                if len(output_text[j].split('<emotion>')) < 2:
                    output_text_ = output_text[j].split('<emotion>')[0].strip()
                else:
                    output_text_ = output_text[j].split('<emotion>')[1].strip()
                output_text_collect.append(output_text_)
                res_emo = res[j].split('<emotion>')[0].strip()
                res_ = res[j].split('<emotion>')[1].strip()
                response.append(res_)
                if emo == res_emo:
                    emo_count += 1
                emo_total += 1
        print("Emo_acc: ", emo_count/emo_total)
        '''
        output_text_full = tokenizer.batch_decode(output, skip_special_tokens=True)
        for j in range(len(output_text_full)):
            res = tokenizer.decode(batch['response'][j, :].tolist())
            if "<eos>" in output_text_full[j]:
                output_text = output_text_full[j].split("<eos>")[0]
            else:
                output_text = output_text_full[j]
            if output_text.startswith('<bos> '):
                output_text = output_text[6:]
            elif output_text.startswith('<bos>'):
                output_text = output_text[5:]
            output_text_collect.append(output_text)
            if "<eos>" in res:
                res = res.split("<eos>")[0]
            if res.startswith('<bos> '):
                res = res[6:]
            elif res.startswith('<bos>'):
                res = res[5:]
            response.append(res)
        '''

        output_path_ = os.path.join(output_path, 'result_raw/')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_path_ = os.path.join(output_path_, str(i) + '.txt')
        file = codecs.open(output_path_, "w", "utf-8")
        for j in range(len(output_text_collect)):
            file.write(output_text_collect[j] + '\t' + response[j][:-1] + os.linesep)  # TODO -1:<\s>
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--mode", default='train', type=str)
    parser.add_argument("--beam_width", default=5, type=int)
    parser.add_argument("--load_epoch", default=0, type=int)
    parser.add_argument("--max_epoch", default=20, type=int)
    args = parser.parse_args()

    # test(args)

    if args.mode == 'test':
        test(args, args.beam_width)
    elif args.mode == 'train':
        train(args)

'''
    Hugging face Examples::

    >> > from transformers import T5Tokenizer, T5ForConditionalGeneration

    >> > tokenizer = T5Tokenizer.from_pretrained('t5-small')
    >> > model = T5ForConditionalGeneration.from_pretrained('t5-small')

    >> >  # training
    >> > input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
    >> > labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
    >> > outputs = model(input_ids=input_ids, labels=labels)
    >> > loss = outputs.loss
    >> > logits = outputs.logits

    >> >  # inference
    >> > input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you",
                               return_tensors="pt").input_ids  # Batch size 1
    >> > outputs = model.generate(input_ids)
    >> > print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    >> >  # studies have shown that owning a dog is good for you.
'''
