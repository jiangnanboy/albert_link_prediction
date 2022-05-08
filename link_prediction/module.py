import torch
from tqdm import tqdm
import random
import os
import numpy as np

from utils.log import logger

from .model import AlbertFC, load_tokenizer, load_config, load_pretrained_model, build_model

from .dataset import GetDataset, get_dataloader, get_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    '''
    set seed
    :param seed:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(2022)

class LP():
    '''
    lp
    '''
    def __init__(self, args):
        self.args = args
        self.SPECIAL_TOKEN = args.SPECIAL_TOKEN
        self.label2i = args.LABEL2I
        self.model = None
        self.tokenizer = None

    def train(self):
        self.tokenizer = load_tokenizer(self.args.pretrained_model_path, self.SPECIAL_TOKEN)
        pretrained_model, albertConfig = load_pretrained_model(self.args.pretrained_model_path, self.tokenizer, self.SPECIAL_TOKEN)

        train_set = GetDataset(self.args.train_path, self.tokenizer, self.args.max_length, self.SPECIAL_TOKEN, self.label2i, 'train')

        if self.args.dev_path:
            dev_dataset = GetDataset(self.args.dev_path, self.tokenizer, self.args.max_length, self.SPECIAL_TOKEN, self.label2i, 'dev')
            val_iter = get_dataloader(dev_dataset, batch_size=self.args.batch_size, shuffle=False)
        train_iter = get_dataloader(train_set, batch_size=self.args.batch_size)

        tag_num = len(self.label2i)
        albertfc = AlbertFC(albertConfig, pretrained_model, tag_num)
        self.model = albertfc.to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=0.1)

        best_val_loss = float('inf')
        for epoch in range(self.args.epochs):
            self.model.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                self.model.zero_grad()

                label = item['label']
                input_ids = item['input_ids']
                attention_mask = item['input_mask']
                segment_ids = item['segment_ids']

                label = label.to(DEVICE)
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                segment_ids = segment_ids.to(DEVICE)

                out = self.model(input_idx=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

                item_loss = criterion(out, label)
                acc_loss += item_loss.item()
                item_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                optimizer.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss / len(train_iter)))

            if self.args.dev_path:
                val_loss = self.validate(val_iter=val_iter, criterion=criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # save model
                    torch.save(self.model.state_dict(), self.args.model_path)
                    # torch.save(self.model, self.args.model_path)
                    logger.info('save model : {}'.format(self.args.model_path))
                logger.info('val_loss: {}, best_val_loss: {}'.format(val_loss, best_val_loss))

            scheduler.step()

    def predict_tail(self, head, rel, entity_list, topk=3):
        self.model.eval()

        tail_list = []
        for tail in entity_list:
            tmp_triple = [head, rel, tail, 1]
            if tmp_triple not in tail_list:
                tail_list.append(tmp_triple)
                tmp_dict = GetDataset.convert_examples_to_features(head, rel, tail, 1, self.tokenizer, self.args.max_length)
                singlg_label = tmp_dict['label']
                single_input_ids = tmp_dict['input_ids']
                single_input_mask = tmp_dict['input_mask']
                single_segment_ids = tmp_dict['segment_ids']

                single_input_ids = single_input_ids.unsqueeze(0)
                single_input_mask = single_input_mask.unsqueeze(0)
                single_segment_ids = single_segment_ids.unsqueeze(0)

                single_input_ids = single_input_ids.to(DEVICE)
                single_input_mask = single_input_mask.to(DEVICE)
                single_segment_ids = single_segment_ids.to(DEVICE)

                with torch.no_grad():
                    out = self.model(input_idx=single_input_ids, attention_mask=single_input_mask, token_type_ids=single_segment_ids)
                    print('predict tail : {}, preds : {}'.format(tail, out.cpu().data))

            # vec_predict = self.model(input_idx=input_ids, attention_mask=attention_mask, entity_mask=entity_mask)[0]
            # soft_predict = torch.softmax(vec_predict, dim=0)
            # predict_prob, predict_index = torch.max(soft_predict.cpu().data, dim=0)
            # i2label = {value: key for key, value in self.label2i.items()}
            # predict_class = i2label[predict_index]
            # predict_prob = predict_prob.item()
            # return predict_prob, predict_class

    def load(self):
        self.tokenizer = load_tokenizer(self.args.pretrained_model_path, self.SPECIAL_TOKEN)
        albertConfig = load_config(self.args.pretrained_model_path, self.tokenizer)
        albert_model = build_model(albertConfig)
        tag_num = len(self.label2i)
        self.model = AlbertFC(albertConfig, albert_model, tag_num)
        # self.model = torch.load(self.args.model_path, map_location=DEVICE)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=DEVICE))
        logger.info('loading model {}'.format(self.args.model_path))
        self.model = self.model.to(DEVICE)

    def validate(self, val_iter, criterion):
        self.model.eval()
        with torch.no_grad():
            labels = np.array([])
            predicts = np.array([])
            val_loss = 0.0
            for dev_item in tqdm(val_iter):
                label = dev_item['label']
                input_ids = dev_item['input_ids']
                attention_mask = dev_item['input_mask']
                segment_ids = dev_item['segment_ids']

                label = label.to(DEVICE)
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                segment_ids = segment_ids.to(DEVICE)

                out = self.model(input_idx=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
                loss = criterion(out, label)

                val_loss += loss.item()

                # p,r,f1 metrics
                prediction = torch.max(torch.softmax(out, dim=1), dim=1)[1]
                pred_y = prediction.cpu().data.numpy().squeeze()
                target_y = label.cpu().data.numpy()
                labels = np.append(labels, target_y)
                predicts = np.append(predicts, pred_y)
            report = get_score(labels, predicts)

            print('dev dataset len:{}'.format(len(val_iter)))
            logger.info('dev_score: {}'.format(report))
        return val_loss / len(val_iter)
