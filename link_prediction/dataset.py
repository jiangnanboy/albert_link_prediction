from torch.utils.data import Dataset, DataLoader
import torch
from torchtext.data import BucketIterator
from sklearn.metrics import classification_report
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GetDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, SPECIAL_TOKENS, label2i, set_type):
        '''
        build dataset
        :param data_path:
        :param tokenizer:
        :param max_length:
        :param SPECIAL_TOKENS:
        :param label2i:
        :param set_type:
        '''
        self.data_list = self.read_data(data_path, set_type)
        self.data_size = len(self.data_list)
        self.tokenizer = tokenizer
        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.max_length = max_length
        self.label2i = label2i

    def _create_examples(self, data_list, set_type):
        '''
        对训练数据构建负样本
        '''
        entities = set() # 实体名
        for ent_rel_ent in data_list:
            entities.add(ent_rel_ent[0]) # head entity
            entities.add(ent_rel_ent[2]) # tail entity
        entities = list(entities)

        examples = []
        for (i, line) in enumerate(data_list):
            head_entity = line[0] # 头实体
            tail_enity = line[2] # 尾实体
            relation = line[1] # 关系
            label = line[3] # label
            if set_type == "dev" or set_type == "test":
                examples.append([head_entity, relation, tail_enity, label])
            elif set_type == "train":
                examples.append([head_entity, relation, tail_enity, label])
                rnd = random.random()
                # 构造负样本
                # 头实体进行随机替换
                if rnd <= 0.5:
                    # corrupting head
                    for j in range(5):
                        tmp_head = ''
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[0]) # 剔除头实体
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_head = random.choice(tmp_ent_list) # 随机获取一个实体作为头实体
                            tmp_triple_str = [tmp_head, line[1], line[2], 0]
                            if tmp_triple_str not in data_list:
                                break
                        examples.append([tmp_head, relation, tail_enity, 0])
                # 尾实体进行随机替换
                else:
                    # corrupting tail
                    tmp_tail = ''
                    for j in range(5):
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[2]) # 剔除尾实体
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_tail = random.choice(tmp_ent_list) # 随机获取一个尾实体
                            tmp_triple_str = [line[0], line[1], tmp_tail, 0]
                            if tmp_triple_str not in data_list:
                                break
                        examples.append([head_entity, relation, tmp_tail, 0])
        return examples

    def read_data(self, data_path, set_type):
        data_list = []
        with open(data_path, 'r', encoding='utf-8') as data_read:
            count = 0
            for line in data_read:
                line = line.strip().split(',')
                entity_1, relation, entity_2 = tuple(line)
                data_list.append([entity_1, relation, entity_2, 1])
                count += 1
        data_list = self._create_examples(data_list, set_type)
        return data_list

    @classmethod
    def _truncate_seq_triple(cls, tokens_a, tokens_b, tokens_c, max_length):
        """Truncates a sequence triple in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
                tokens_a.pop()
            elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
                tokens_b.pop()
            elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
                tokens_c.pop()
            else:
                tokens_c.pop()

    @classmethod
    def convert_examples_to_features(cls, head, rel, tail, label, tokenizer, max_length):
        head_tokens = tokenizer.tokenize(head)
        relation_tokens = tokenizer.tokenize(rel)
        tail_tokens = tokenizer.tokenize(tail)
        GetDataset._truncate_seq_triple(head_tokens, relation_tokens, tail_tokens, max_length - 4)

        tokens = ["[CLS]"] + head_tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        tokens += relation_tokens + ["[SEP]"]
        segment_ids += [1] * (len(relation_tokens) + 1)

        tokens += tail_tokens + ["[SEP]"]
        segment_ids += [0] * (len(tail_tokens) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(segment_ids) == max_length

        return {'label': torch.tensor(label),
                'input_ids': torch.tensor(input_ids),
                'input_mask': torch.tensor(input_mask),
                'segment_ids': torch.tensor(segment_ids)}

    def __getitem__(self, idx):
        head_entity, relation, tail_entity, label = self.data_list[idx]
        return GetDataset.convert_examples_to_features(head_entity, relation, tail_entity, label, self.tokenizer, self.max_length)

    def __len__(self):
        return self.data_size

def get_train_val_dataloader(batch_size, trainset, train_ratio):
    '''
    split trainset to train and val
    :param batch_size:
    :param trainset:
    :param train_ratio
    :return:
    '''

    train_size = int(train_ratio * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

    trainloader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    valloader = DataLoader(val_dataset,
                           batch_size=batch_size,
                           shuffle=False)

    return trainloader, valloader, train_dataset, val_dataset

def get_dataloader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_iterator(dataset: Dataset, batch_size, sort_key=lambda x: len(x.input_ids), sort_within_batch=True, shuffle=True):
    return BucketIterator(dataset, batch_size=batch_size, sort_key=sort_key,
                          sort_within_batch=sort_within_batch, shuffle=shuffle)

def get_score(labels, predicts):
    return classification_report(labels, predicts, target_names=None)