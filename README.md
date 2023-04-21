# albert-fc for LP(Link Prediction)，链接预测

## 概述
链接预测是一种根据KG中已存在的实体去预测缺失事实的任务，是一种有前途的、广泛研究的、旨在解决KG的不完整的任务。

## 方法

利用huggingface/transformers中的albert+fc进行知识图谱链接预测。

利用albert加载中文预训练模型，后接前馈网络。利用albert预训练模型进行fine-tune。

albert预训练模型下载：

链接：https://pan.baidu.com/s/1GSuwd6z_YG1oArnHVuN0eQ 

提取码：sy12

整个流程是：

- 数据经albert后获取最后的隐层hidden_state=768
- 根据albert的last hidden_state，获取[CLS]的hidden_state经一层前馈网络进行分类

![image](https://raw.githubusercontent.com/jiangnanboy/albert_link_prediction/master/image/albert-lp.png)

 ## 数据说明

数据形式见data/

训练数据示例如下，其中各列为`头实体`、`关系`、`尾实体`。

```
科学,包涵,自然、社会、思维等领域
科学,外文名,science
科学,拼音,kē xué
科学,中文名,科学
科学,解释,发现、积累的真理的运用与实践
```

## 训练和预测见（examples/test_lp.py）

```
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
```

## 项目结构
- data
    - test.sample.csv
    - train.sample.csv
- examples
    - test_lp.py #训练及预测
- model
    - pretrained_model #存放预训练模型和相关配置文件
        - config.json
        - pytorch_model.bin
        - vocab.txt
- link_prediction
    - dataset.py
    - model.py
    - module.py
- utils
    - log.py

## 参考
- [transformers](https://github.com/huggingface/transformers)
- [KG-BERT: BERT for Knowledge Graph Completion](https://arxiv.org/pdf/1909.03193.pdf)
