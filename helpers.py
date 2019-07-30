import torch as t
import json


def word2idx(word, vocabulary_path='./data/vocabulary_old.json'):
    with open(vocabulary_path, 'r', encoding='utf-8') as file:
        vocabulary_dict = json.load(file)
    try:
        for (j, i) in enumerate(word):
            temp = t.LongTensor([vocabulary_dict[i]])
            if j == 0:
                target_word_idx = temp
            else:
                target_word_idx = t.cat((target_word_idx, temp), 0)
        print(target_word_idx)
        return target_word_idx
    except:
        print("there is no this word!")


def idx2word(index, vocabulary_path='./data/vocabulary_old.json'):
    with open(vocabulary_path, 'r', encoding='utf-8') as file:
        vocabulary_dict = json.load(file)
    idx_dict = {value: key for key, value in vocabulary_dict.items()}
    word = []
    for (j, i) in enumerate(index):
        try:
            word.append(idx_dict[i])
        except:
            word.append("UKN")
    print(word)
    return word


def find_length(length=3, word_length_dict_path="./data/word_length_dict.json", verbose=True):
    with open(word_length_dict_path, 'r', encoding='utf-8') as file:
        word_length_dict = json.load(file)
    res = []
    for i in word_length_dict.keys():
        if word_length_dict[i] == length:
            res.append(i)
    if verbose:
        print(res)
    return res


def multi_search(file_list=['./data/vocabulary_old.json', './data/strain_dict.json', './data/word_length_dict.json', './data/pinyin_dict.json'], max_length=3, use_strains=True, strains="平仄平", use_last_pinyin=True, last_pinyin='an', verbose=False):
    with open(file_list[1], 'r', encoding='utf-8') as file:
        strain_dict = json.load(file)
    with open(file_list[3], 'r', encoding='utf-8') as file:
        pinyin_dict = json.load(file)
    length_candidate = []
    res = []
    res_1 = []
    for i in range(1, max_length+1):
        length_candidate += find_length(length=i, verbose=False)
    if use_last_pinyin:
        for i in length_candidate:
            if len(i) == max_length and pinyin_dict[i] == last_pinyin:
                res.append(i)
            elif len(i) < max_length:
                res.append(i)
    if use_strains:
        for i in res:
            temp_len = len(i)
            if strain_dict[i] == strains[0: temp_len]:
                res_1.append(i)
        if verbose:
            print(res_1)
        return res_1
    else:
        res = length_candidate
        if verbose:
            print(res)
        return res


def find_poem_length(inp, vocabulary_path="./data/vocabulary_old.json"):
    """inp:batch x seq_len"""
    with open(vocabulary_path, 'r', encoding='utf-8') as file:
        vocabulary = json.load(file)
    vocabulary_list = list(vocabulary.keys())
    batch_size = inp.shape[0]
    res = t.zeros(batch_size).view(-1, 1)
    for i in range(batch_size):
        index_length = len(inp[i])
        poem_length = 0
        for j in range(index_length):
            temp = int(inp[i][j].numpy())
            poem_length += len(vocabulary_list[temp])
            res[i][0] = poem_length
    # print(res)
    return res

def find_right_word_batch(batch_size=5):
    def inner(func):
        def wrapper(*args, **kargs):
            for i in range(batch_size):
                print(func(*args, **kargs))
            return func(*args, **kargs)
        return wrapper
    return inner


def find_right_word(use_strains=True, strains='平', use_pinyin=True, pinyin='an', file_list=['./data/vocabulary.json', './data/strains.json', './data/pinyin.json'], verbose=False):
    with open(file_list[0], 'r', encoding='utf-8') as file:
        vocabulary = json.load(file)
    with open(file_list[1], 'r', encoding='utf-8') as file:
        strain_dict = json.load(file)
    with open(file_list[2], 'r', encoding='utf-8') as file:
        pinyin_dict = json.load(file)
    temp_0 = []
    temp_1 = []
    if use_strains:
        for i in list(vocabulary.keys()):
            if strain_dict[str(vocabulary[i])] == strains:
                temp_0.append(vocabulary[i])
    else:
        temp_0 = list(strain_dict.keys())
        temp_0 = [int(x) for x in temp_0]
    if use_pinyin:
        for i in list(vocabulary.keys()):
            if pinyin_dict[str(vocabulary[i])] == pinyin:
                temp_1.append(vocabulary[i])
    else:
        temp_1 = list(strain_dict.keys())
        temp_1 = [int(x) for x in temp_1]
    res = [x for x in temp_0 if x in temp_1]
    if verbose:
        print("the answer is:", res)
    return res


def find_strains(index, file="./data/strains.json"):
    with open(file, 'r', encoding='utf-8') as file:
        strain_dict = json.load(file)
    return strain_dict[str(index)]

def find_pinyin(index, file="./data/pinyin.json"):
    with open(file, 'r', encoding='utf-8') as file:
        pinyin_dict = json.load(file)
    return pinyin_dict[str(index)]

def index2word(index, file='./data/vocabulary.json', verbose=False):
    with open(file, 'r', encoding='utf-8') as file:
        vocabulary = json.load(file)
    vocabulary = {value: key for key, value in vocabulary.items()}
    res = []
    for i in index:
        res.append(vocabulary[i])
    if verbose:
        print("res is ", res)
    return res








if __name__ == '__main__':
    # print(multi_search(max_length=1, use_strains=False, use_last_pinyin=False))
    # find_poem_length(t.Tensor([[1,3], [0,2]]))

    find_right_word(use_strains=False, verbose=True)