# Word2Vec Skip-Gram 全套项目：三天作战计划

> 总工时预估：30-36 小时（每天 10-12 小时，请睡觉）

---

## Day 1：地基 — 数据处理 + Skip-Gram 核心实现

### Step 0：阅读核心文献（1.5h）

必读 3 篇：
1. Mikolov et al., 2013, "Efficient Estimation of Word Representations in Vector Space"
2. Mikolov et al., 2013, "Distributed Representations of Words and Phrases and their Compositionality"（负采样的出处）
3. Levy & Goldberg, 2014, "Neural Word Embedding as Implicit Matrix Factorization"

**不需要精读，重点看：**
- 负采样的损失函数公式（论文原始表述）
- α=0.75 在论文中是怎么提出的（你会发现作者没给太多解释）
- 下采样公式的原始定义

**产出检查点：**
- [ ] 能用自己的话解释为什么负采样能近似 full softmax
- [ ] 知道 α=0.75 的来源，理解为什么这是个值得研究的点

### Step 1.1 环境与数据准备（1.5h）

**做什么：**
- 下载天池数据集 https://tianchi.aliyun.com/dataset/119797
- 探索数据：看格式、文本量、语言（中/英）、噪声情况
- 安装依赖：numpy, matplotlib, jieba（如果是中文）, scipy, gensim（后面对比用）

**产出检查点：**
- [ ] 能打印出数据前 10 条
- [ ] 知道总共多少条文本、大约多少词

### Step 1.2 文本预处理（2h）

**做什么：**
- 清洗：去除 HTML 标签、特殊符号、数字（看情况）
- 分词：英文用 split/nltk，中文用 jieba
- 构建词表：统计词频，设定最小词频阈值（建议 min_count=5）过滤低频词
- 构建 word2index 和 index2word 映射

**关键代码结构：**
```
class Vocabulary:
    def __init__(self, min_count=5)
    def build(self, corpus)      -> 构建词表
    def word_freq()               -> 返回词频字典
    vocab_size                    -> 词表大小
```

**产出检查点：**
- [ ] 词表大小在合理范围（通常几千到几万）
- [ ] 能通过 word2index["的"] 拿到 id，能反查回来

### Step 1.3 训练样本生成（2h）

**做什么：**
- 实现 skip-gram 样本生成器：给定中心词，生成 (center, context) 词对
- 窗口大小 window_size=5（动态窗口：每次随机取 1~window_size）
- 输出格式：`[(center_id, context_id), ...]`

**关键代码结构：**
```
def generate_training_pairs(corpus, word2index, window_size=5):
    pairs = []
    for sentence in corpus:
        for i, center in enumerate(sentence):
            # 动态窗口
            dynamic_window = random.randint(1, window_size)
            for j in range(i - dynamic_window, i + dynamic_window + 1):
                if j != i and 0 <= j < len(sentence):
                    pairs.append((word2index[center], word2index[sentence[j]]))
    return pairs
```

**产出检查点：**
- [ ] 能打印出 10 个样本对，人眼看是否合理（中心词和上下文确实相关）
- [ ] 样本总数量级：几十万到几百万

### Step 1.4 Skip-Gram 基础版 — Full Softmax（3-4h）

**做什么：**
- 实现两个嵌入矩阵：W_center (V×D), W_context (V×D)
- 前向传播：center_vec → 与所有 context_vec 做点积 → softmax → 交叉熵损失
- 反向传播：手推梯度，更新两个矩阵
- 用 SGD 优化

**核心公式：**
```
前向：
  score_j = W_context[j] · W_center[i]    对所有 j
  P(j|i) = exp(score_j) / Σ exp(score_k)
  loss = -log P(context|center)

梯度：
  对 W_center[i]:  ∂L/∂W_center[i] = Σ_j (P(j|i) - 1_{j=context}) * W_context[j]
  对 W_context[j]: ∂L/∂W_context[j] = (P(j|i) - 1_{j=context}) * W_center[i]
```

**关键代码结构：**
```
class SkipGramBasic:
    def __init__(self, vocab_size, embedding_dim)
    def forward(self, center_id, context_id)   -> loss
    def backward()                              -> 更新参数
    def train(self, pairs, epochs, lr)          -> 训练循环，记录 loss
```

**产出检查点：**
- [ ] 在 **小数据**（示例文本，几百词）上 loss 能稳定下降
- [ ] 打印 loss 曲线，肉眼确认在收敛
- [ ] ⚠️ 此版本在大数据上会极慢（softmax 遍历整个词表），这是预期行为，不要在大数据上跑这个版本

### Step 1.5 数值梯度检验（0.5h）

**做什么：**
- 实现数值梯度：对每个参数微扰 ε=1e-5，计算 (L(θ+ε) - L(θ-ε)) / 2ε
- 与你的解析梯度对比，相对误差应 < 1e-5

**这一步非常重要！** 如果跳过，Day 2 加负采样时梯度错了你会完全找不到 bug。

**产出检查点：**
- [ ] 相对误差 < 1e-5，打印结果截图保存

---

## Day 1 结束时你应该有的东西：
1. ✅ 数据预处理 pipeline
2. ✅ 词表构建完成
3. ✅ 训练样本生成器
4. ✅ 基础 skip-gram 在小数据上跑通，loss 下降
5. ✅ 梯度检验通过

---

## Day 2：核心优化 — 负采样 + 工程提速

### Step 2.1 负采样实现（3-4h）⭐ 最关键的一步

**为什么：** Full softmax 复杂度 O(V)，负采样降到 O(K)，K 一般取 5-15。

**做什么：**
- 构建负采样概率分布：P(w) = freq(w)^0.75 / Σ freq(w)^0.75
- 预计算采样表（unigram table，长度 1e6-1e8），用于快速采样
- 修改损失函数和梯度

**负采样公式：**
```
loss = -log σ(W_context[pos] · W_center[i])
       - Σ_{neg samples} log σ(-W_context[neg] · W_center[i])

其中 σ(x) = 1 / (1 + exp(-x))  即 sigmoid

梯度：
  对正样本 context:
    ∂L/∂W_context[pos] = (σ(score) - 1) * W_center[i]
    ∂L/∂W_center[i]   += (σ(score) - 1) * W_context[pos]

  对每个负样本 neg:
    ∂L/∂W_context[neg] = σ(score_neg) * W_center[i]
    ∂L/∂W_center[i]   += σ(score_neg) * W_context[neg]
```

**关键代码结构：**
```
class SkipGramNegSampling:
    def __init__(self, vocab_size, embedding_dim, neg_samples=10)
    def build_neg_sample_table(self, word_freq)  -> 采样表
    def sample_negatives(self, center_id, k)     -> k 个负样本 id
    def forward(self, center_id, pos_id, neg_ids) -> loss
    def backward()                                -> 更新参数
```

**产出检查点：**
- [ ] 在小数据上 loss 下降
- [ ] 对负采样也做数值梯度检验！
- [ ] 训练速度相比 full softmax 有明显提升

### Step 2.2 高频词下采样（1h）

**做什么：**
- 实现 Mikolov 的下采样公式：P(discard) = 1 - sqrt(t / freq(w))，t=1e-5
- 在生成训练样本时，高频词（如"的"、"the"）有概率被跳过

**产出检查点：**
- [ ] "的"、"的"等高频词的丢弃概率 > 0.9
- [ ] 低频词的丢弃概率 ≈ 0

### Step 2.3 学习率线性衰减（0.5h）

**做什么：**
```
lr = initial_lr * (1 - current_step / total_steps)
lr = max(lr, initial_lr * 0.0001)  # 设下限，别衰减到 0
```

**产出检查点：**
- [ ] 打印 lr 变化曲线，确认线性下降

### Step 2.4 在真实数据上训练（2-3h）

**做什么：**
- 用天池数据集完整训练
- 超参数建议起点：embedding_dim=100, window=5, neg_samples=10, lr=0.025, epochs=5
- 训练过程中每隔 N 步打印 loss
- 保存训练好的词向量到文件（numpy .npy 或纯文本格式）

**⚠️ 速度优化提示：**
- 尽量用 numpy 向量化，避免 python for 循环
- 如果太慢，可以减少语料（取前 10 万条）或降低 embedding_dim
- 预计在 10 万条级别的语料上，单 epoch 需要 10-30 分钟（视实现效率而定）

**产出检查点：**
- [ ] 完整训练跑完，loss 整体呈下降趋势
- [ ] 词向量文件已保存

### Step 2.5：词类比评估实现（1h）

**做什么：**
- 实现 analogy 评估：a - b + c ≈ d
- 例如：king - man + woman ≈ queen

```python
def analogy(a, b, c, W, word2index, index2word, topk=5):
    """a is to b as c is to ?"""
    vec = W[word2index[b]] - W[word2index[a]] + W[word2index[c]]
    # 归一化
    vec = vec / np.linalg.norm(vec)
    W_norm = W / np.linalg.norm(W, axis=1, keepdims=True)
    scores = W_norm @ vec
    # 排除输入词
    exclude = {word2index[a], word2index[b], word2index[c]}
    top_ids = np.argsort(scores)[::-1]
    results = []
    for idx in top_ids:
        if idx not in exclude:
            results.append((index2word[idx], scores[idx]))
        if len(results) == topk:
            break
    return results
```

**产出检查点：**
- [ ] 函数能跑通
- [ ] 结果不一定完美，但函数逻辑正确


---

## Day 2 结束时你应该有的东西：
1. ✅ 负采样版 skip-gram 实现
2. ✅ 高频词下采样
3. ✅ 学习率衰减
4. ✅ 在真实数据上训练完成，词向量已保存

---

## Day 3：展示层 — 评估 + 可视化 + 对比

### Step 3.1 词相似度评估（2h）

**做什么：**
- 实现 cosine_similarity(vec_a, vec_b)
- 实现 most_similar(word, topk=10)：给定词，返回最相似的 k 个词
- 手动测试几组有意义的词（如"国王"→ 应该出现"皇帝"等）

**如果是英文数据，可以用标准评估集：**
- WordSim-353 或 SimLex-999
- 计算你的模型在这些数据集上的 Spearman 相关系数

**关键代码：**
```
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def most_similar(word, W, word2index, index2word, topk=10):
    vec = W[word2index[word]]
    scores = W @ vec / (np.linalg.norm(W, axis=1) * np.linalg.norm(vec))
    top_ids = np.argsort(scores)[::-1][1:topk+1]
    return [(index2word[i], scores[i]) for i in top_ids]
```

**产出检查点：**
- [ ] most_similar 结果看起来有一定语义合理性
- [ ] 如果有评估集：Spearman 系数 > 0.3 就算可以（手写实现能到这个水平已经不错）

### Step 3.2 t-SNE 可视化（1.5h）

**做什么：**
- 选取高频词 top 100-200 个
- 用 sklearn.manifold.TSNE 降到 2D
- 用 matplotlib 画散点图 + 标注词

**产出检查点：**
- [ ] 图上能看到一些语义聚类（如数字聚在一起、动词聚在一起）
- [ ] 图片保存为 word_vectors_tsne.png

### Step 3.3 Gensim 对比实验（1.5h）

**做什么：**
- 用 gensim.models.Word2Vec 在同样数据上训练
- 同样的超参数（embedding_dim, window, neg_samples）
- 对比两者在词相似度任务上的表现

**关键代码：**
```
from gensim.models import Word2Vec

model = Word2Vec(sentences, vector_size=100, window=5,
                 min_count=5, sg=1, negative=10, epochs=5)
```

**产出检查点：**
- [ ] 对比表格：你的实现 vs gensim，在相似度评估上的分数
- [ ] 分析差距来源（训练技巧、数据处理差异等）

### Step 3.4 升级为：围绕研究问题的系统实验（2.5h）

**不再是随意调参，而是有目的的对比实验。**

以方向 A 为例：

| 实验编号 | α 值 | 评估：全量 SimScore | 评估：高频词 SimScore | 评估：低频词 SimScore | 类比准确率 |
|---------|:---:|:---:|:---:|:---:|:---:|
| Exp-1 | 0.50 | ? | ? | ? | ? |
| Exp-2 | 0.75 | ? | ? | ? | ? |
| Exp-3 | 1.00 | ? | ? | ? | ? |
| Gensim baseline | 0.75 | ? | ? | ? | ? |

**额外分析：**
- 画出三组 α 对应的采样概率分布图（一张图说明 α 如何改变采样偏向）
- 按词频分桶（高/中/低频），分别统计各桶内词向量的平均 cosine 相似度
- 如果低频词在 α=0.5 时明显更好，这就是你的核心发现

### Step 3.5 训练 Loss 曲线整理（0.5h）

**做什么：**
- 用 matplotlib 画出完整的训练 loss 曲线
- 标注学习率变化（如果有衰减）
- 保存为 training_loss.png

### Step 3.6 代码整理与打包（1-1.5h）

**做什么：**
- 代码组织成清晰的模块结构：

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

- 写一个简洁的 README，包含：项目简介、运行方式、核心结果

**产出检查点：**
- [ ] 任何人 clone 下来能跑通
- [ ] README 清晰明了

### 新增 Step 3.7：写迷你论文（2-3h）

**这是你 Day 3 最重要的任务。**

- 用 LaTeX（ACL 短论文模板）写 4-6 页报告
- 不需要写得像正式论文那么精雕细琢，但结构要完整
- 所有图表直接从 `experiments/results/` 引用
- 重点打磨 Results & Analysis 部分——这是教授判断你研究能力的核心

**产出检查点：**
- [ ] PDF 生成成功
- [ ] 图表清晰，有 caption
- [ ] Related Work 引用了 4-5 篇论文

---

## Day 3 结束时你应该有的东西：
1. ✅ 词相似度评估结果
2. ✅ t-SNE 可视化图
3. ✅ Gensim 对比数据
4. ✅ 超参数消融实验
5. ✅ 整洁的代码仓库 + README

---

## 风险预警 & 应急方案

| 风险 | 症状 | 应急方案 |
|------|------|----------|
| 梯度写错 | loss 不降或爆炸 | 回退到 Step 1.5 做数值梯度检验 |
| 训练太慢 | 单 epoch > 1 小时 | 缩小语料到 5 万条，降维到 50 |
| 相似度结果差 | most_similar 输出不相关 | 先确认 loss 在下降；检查预处理是否合理 |
| 时间不够 | Day 2 还在调 bug | 砍掉 3.4 超参数实验，优先保 3.1 + 3.2 + 3.6 |

## 最低可上简历标准（如果时间真的不够）

按优先级排序，至少完成前 4 项：
1. ✅ Skip-Gram + Negative Sampling 手动实现
2. ✅ 在真实数据上训练完成
3. ✅ most_similar 功能可用
4. ✅ 训练 loss 曲线
5. ⬜ t-SNE 可视化（加分项）
6. ⬜ Gensim 对比（加分项）
7. ⬜ 超参数消融实验（锦上添花）