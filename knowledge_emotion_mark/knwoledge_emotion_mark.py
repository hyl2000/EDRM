import sys

sys.path += ['./']
import json
from collections import defaultdict
from tqdm import tqdm
import re
from collections import Counter
import jieba
import numpy as np


VAD = json.load(open("VAD.json", "r", encoding="utf-8"))
concept = json.load(open("ConceptNet.json", "r", encoding="utf-8"))


def takeFirst(elem):
    return elem[0]


def pad_knowledge(knowledge):
    for i in range(len(knowledge)):
        if len(knowledge[i]) > 50:
            knowledge[i] = knowledge[i][0:50]
        else:
            knowledge[i] = knowledge[i] + (50 - len(knowledge[i])) * [0]
    return knowledge


def emotion_intensity(NRC, word):
    '''
    Function to calculate emotion intensity (Eq. 1 in our paper)
    :param NRC: NRC_VAD vectors
    :param word: query word
    :return:
    '''
    v, a, d = NRC[word]
    a = a/2
    return (np.linalg.norm(np.array([v, a]) - np.array([0.5, 0])) - 0.06467)/0.607468


def emotion_net(sentence):
    concept_num = 1
    # words_pos = nltk.pos_tag(sentence, lang='cmn')  # TODO
    vads = []  # each item is sentence, each sentence contains a list word' vad vectors
    vad = []
    choice = set()

    # vads.append([VAD[word] if word in VAD else [0.5, 0.0, 0.5] for word in sentence])
    # vad.append([emotion_intensity(VAD, word) if word in VAD else 0.0 for word in sentence])

    # sentence_concepts = [concept[word] if word in concept and wordCate(words_pos[wi]) else []
    #                      for wi, word in enumerate(sentence)]
    sentence_concepts = [concept[word] if word in concept else [] for wi, word in enumerate(sentence)]

    sentence_concept_words = []  # for each sentence

    for cti, uc in enumerate(sentence_concepts):  # filter concepts of each token, complete their VAD value, select top total_concept_num.
        concept_words = []  # for each token
        if uc != []:  # this token has concepts
            for c in uc:  # iterate the concept lists [c,r,w] of each token
                if c in VAD and emotion_intensity(VAD, c) >= 0.8 and c not in choice:
                    concept_words.append((emotion_intensity(VAD, c), c))
                    choice.add(c)

            concept_words = concept_words[:concept_num]

        sentence_concept_words.extend(concept_words)
    sentence_concept_words.sort(key=takeFirst)
    sentence_concept_words = sentence_concept_words[:concept_num]
    return sentence_concept_words


def f1_score(golden_response, response_):
    # response是预测的，golden是真实的
    """
    calc_f1
    """
    response_ = jieba.lcut(response_)
    golden_response = jieba.lcut(golden_response)
    golden_char_total = 0.0
    pred_char_total = 0.0
    hit_char_total = 0.0
    golden_response = "".join(golden_response)
    response_ = "".join(response_)
    common = Counter(response_) & Counter(golden_response)
    hit_char_total += sum(common.values())
    golden_char_total += len(golden_response)
    pred_char_total += len(response_)
    p = hit_char_total / pred_char_total
    r = hit_char_total / golden_char_total
    if p != 0 or r != 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0
    return f1


def get_best_match(query, candidates):
    temp = [[f1_score(query.lower(), line.lower()), line] for line in candidates]
    temp.sort(key=takeFirst, reverse=True)
    return temp


def del_space(collect):
    return [''.join(s.split()).strip() for s in collect]


def del_id_atom(s):
    s = re.sub(u"\\[.*?]", "", s)
    if len(s) == 0:
        return s
    if s[0] == '[':
        if ']' in s:
            s = s[s.index(']') + 1:]
        else:
            s = s[2:]
    return s


def del_id(collect):
    return [del_id_atom(s) for s in collect]


def merge_graph(graph):
    out = defaultdict(list)
    for item in graph:
        out[item[0] + ' <l> ' + item[1]].append(item[2])
    new_out = []
    for k, v in out.items():
        new_out.append(k.split('<l>') + [' ; '.join(set(v))])
    return new_out


def remove_dup(collect):
    out = ['']
    for s in collect:
        if s != out[-1]:
            out.append(s)
    return out[1:]


data = json.load(open('train_rank.json', encoding='utf-8'))
new = []
none = 0
count = 0
for line in tqdm(data):
    goal = line['goal']
    history = line['context']
    response = line['response']
    knowledge = line['knowledge'].split(' | ')
    knowledge_new = []
    for k in knowledge:
        concept_word = emotion_net(jieba.lcut(k.replace('</s>', '')))
        if len(concept_word) > 0:
            _, word = concept_word[0]
            # print(concept_word[0])
        else:
            word = 'No_emotion'
            none += 1
        count += 1
        knowledge_new.append(k + ':' + word)
    knowledge = ' | '.join(knowledge_new)
    emotion = line['emotion']
    new.append({'context': history, 'goal': goal, 'response': response, 'knowledge': knowledge, 'emotion': emotion})
print(none / count)

json.dump(new, open('train_emotion.json', 'w', encoding='utf-8'), ensure_ascii=False)
