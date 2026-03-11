# word2dec-demo-NJU-NLP

这是南京大学人工智能学院开设的自然语言处理课的第一次课程作业。项目内容是实现word2dec的基本算法。

仓库结构（计划中）：

```
word2vec_project/
├── README.md              # 项目说明、运行方式、结果总结
├── data/
│   └── preprocess.py      # 数据加载与预处理
├── model/
│   ├── skipgram.py        # Skip-Gram + Negative Sampling 实现
│   └── vocabulary.py      # 词表构建
├── train.py               # 训练入口
├── evaluate.py            # 评估（相似度、对比）
├── visualize.py           # t-SNE 可视化
├── compare_gensim.py      # Gensim 对比
├── results/
│   ├── training_loss.png
│   ├── tsne_visualization.png
│   └── evaluation_results.txt
└── requirements.txt
```