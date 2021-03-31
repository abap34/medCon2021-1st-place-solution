# ECG-classification

[全国医療AIコンテスト](https://www.kaggle.com/c/ai-medical-contest-2021/overview)の、チーム🦾😢のコードです。

大幅にリファクタリングしています。

発表時に使用したスライドは、
![https://docs.google.com/presentation/d/1zEko7m5alvMbFL09ImxFmKA8heCwzVzMXy1fQay__lM/edit?usp=sharing](https://cdn.discordapp.com/attachments/826098877458415638/826098891237490688/unknown.png)

https://docs.google.com/presentation/d/1zEko7m5alvMbFL09ImxFmKA8heCwzVzMXy1fQay__lM/edit?usp=sharing

で閲覧可能です。

# ファイル構成


`logs/`以下に各モデルの学習ログが、
`src/`以下に訓練・推論用途 のコードが、
`src/models/`以下に各モデルの定義が置かれています。

```
.
├── LICENSE
├── README.md
├── logs
│   ├── lstm
│   │   ├── losses
│   │   │   ├── fold-0-log.csv
│   │   │   ├── fold-0.png
…
│   │   │   ├── fold-4-log.csv
│   │   │   └── fold-4.png
│   │   ├── models
│   │   │   ├── checkpoint
│   │   │   ├── fold-0.ckpt.data-00000-of-00001
│   │   │   ├── fold-0.ckpt.index
…
│   │   │   ├── fold-4.ckpt.data-00000-of-00001
│   │   │   └── fold-4.ckpt.index
│   │   └── params.yaml
│   ├── pseudo_lstm
…
│   ├── resnet_1
...
│   ├── resnet_2
…
│   └── wavenet
...
├── requirements.txt
└── src
    ├── models
    │   ├── core_model.py
    │   ├── lstm.py
    │   ├── resnet_1.py
    │   ├── resnet_2.py
    │   └── wavenet.py
    ├── predict.py
    ├── pseudo_train.py
    ├── train.py
    └── utils.py
```

# Usage


```python3 train.py model_name=wavenet epoch=25 batch_size=8```