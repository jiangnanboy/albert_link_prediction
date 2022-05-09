import os
import argparse

from link_prediction.module import LP

def get_entity(file_path):
    entities = set()
    with open(file_path, 'r', encoding='utf-8') as f_read:
        for line in f_read:
            entities.add(line.strip())
    return list(entities)

if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    print("Base path : {}".format(path))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pretrained_model_path',
        default=os.path.join(path, 'model/pretrained_model'),
        type=str,
        required=False,
        help='The path of pretrained model!'
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join(path, 'model/lp_pytorch_model.bin'),
        type=str,
        required=False,
        help="The path of model!",
    )
    parser.add_argument(
        '--SPECIAL_TOKEN',
        default={"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]"},
        type=dict,
        required=False,
        help='The dictionary of special tokens!'
    )
    parser.add_argument(
        '--LABEL2I',
        default={'0': 0,
                 '1': 1,},
        type=dict,
        required=False,
        help='The dictionary of label2i!'
    )
    parser.add_argument(
        "--train_path",
        default=os.path.join(path, 'data/train.sample.csv'),
        type=str,
        required=False,
        help="The path of training set!",
    )
    parser.add_argument(
        '--dev_path',
        default=os.path.join(path, 'data/test.sample.csv'),
        type=str,
        required=False,
        help='The path of dev set!'
    )
    parser.add_argument(
        '--test_path',
        default=None,
        type=str,
        required=False,
        help='The path of test set!'
    )
    parser.add_argument(
        '--log_path',
        default=None,
        type=str,
        required=False,
        help='The path of Log!'
    )
    parser.add_argument("--epochs", default=100, type=int, required=False, help="Epochs!")
    parser.add_argument(
        "--batch_size", default=32, type=int, required=False, help="Batch size!"
    )
    parser.add_argument('--step_size', default=50, type=int, required=False, help='lr_scheduler step size!')
    parser.add_argument("--lr", default=0.0001, type=float, required=False, help="Learning rate!")
    parser.add_argument('--clip', default=5, type=float, required=False, help='Clip!')
    parser.add_argument("--weight_decay", default=0, type=float, required=False, help="Regularization coefficient!")
    parser.add_argument(
        "--max_length", default=100, type=int, required=False, help="Maximum text length!"
    )
    parser.add_argument('--train', default='false', type=str, required=False, help='Train or predict!')
    args = parser.parse_args()
    train_bool = lambda x:x.lower() == 'true'
    lp = LP(args)
    if train_bool(args.train):
        lp.train()
    else:
        lp.load()
        entities = get_entity('../data/entities.txt')
        predict_result = lp.predict_tail('科学', '包涵', entities)
        predict_result = sorted(predict_result.items(), key=lambda x: x[1], reverse=True)
        print(predict_result[:10])

        predict_result = lp.predict_tail('编译器', '外文名', entities)
        predict_result = sorted(predict_result.items(), key=lambda x: x[1], reverse=True)
        print(predict_result[:10])
