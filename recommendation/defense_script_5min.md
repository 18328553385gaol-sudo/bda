# 5-Minute Defense Script

大家好，我这个项目叫做 **Smart Scouting Recommendation System**。  
它的目标是做一个足球球探辅助系统，帮助用户完成三个任务：

- 找相似球员
- 做引援排序
- 查看球员画像

我希望这个项目不只是一个单独的模型，而是一个从数据、训练到前端展示都比较完整的原型系统。

---

首先，在建模上，我没有直接上来就用深度学习，而是先做了一个 baseline。

baseline 的作用有两个。  
第一，它本身就是一种可解释的推荐方法。因为足球球员已经有很多结构化统计特征，比如射门、xG、关键传球、抢断、带球推进、活动范围等等，这些特征本来就可以直接用来衡量球员风格。  
第二，它可以作为后续深度学习模型的对照组，帮助我判断 learned model 是否真正带来了新的价值。

在 baseline 中，我选了 12 个核心特征，覆盖进攻、推进、防守和空间活动四个方面。  
然后我分别尝试了 cosine similarity、euclidean similarity，以及 PCA 加 cosine similarity 三种方式来构建相似度矩阵。  
给定一个目标球员之后，系统会在这个特征空间里找到最相似的候选人，并且可以选择是否只保留同位置球员。

---

在 baseline 之上，我又训练了一个 autoencoder，用来学习球员的低维 embedding。

我选择 autoencoder 的原因是，这个任务更适合做无监督表示学习，而不是监督分类。  
我们并没有明确标签告诉模型哪些球员是相似的，所以更合理的做法是让模型从球员统计特征中自己学习一种压缩表示。

模型结构比较轻量。  
输入是 12 维特征，中间有一个 16 维隐藏层，最后压缩成 6 维 latent embedding。  
训练目标不是预测标签，而是重构原始特征，也就是输入和输出是同一个球员的特征向量。  
损失函数使用 MSE，优化器使用 Adam，同时加入了 validation split 和 early stopping 来保证训练稳定。

---

在训练过程中，我遇到的一个关键问题是数据预处理。

一开始，本地 `processed_features.parquet` 训练是正常的，但当我把训练链路接到 BigQuery 特征表以后，扩大样本规模时出现了 `loss = nan`。  
后面排查发现，本地训练用的 `processed_features.parquet` 并不是 BigQuery 表的直接导出，而是经过了额外 preprocessing：

- 缺失值填补为 0
- 按 `position_group` 做 z-score normalization

所以后来我把训练脚本中的 preprocessing 逻辑改成和 notebook 一致。  
这样做之后，BigQuery 模式下训练恢复稳定，validation loss 能够正常下降。  
这说明这次优化不仅是把数据接上了云，而是让本地训练流程和云端数据流程真正对齐了。

---

在系统功能上，我做了三个 Streamlit 页面。

第一个是 **Similar Player Recommendation**。  
这个页面会同时展示 baseline 推荐结果和 autoencoder embedding 推荐结果，并提供 overlap、position purity 这些对比指标。  
它的价值在于，让我可以同时观察“手工特征空间”和“学习到的 embedding 空间”下，相似球员推荐有什么差异。

第二个是 **Recruitment Ranking**。  
它不是问“谁和某个球员最像”，而是问“如果我要引援，这个位置上谁最符合我的需求”。  
这里我把底层特征聚合成 attacking、progression、defensive、spatial 四个维度，然后允许用户自己调权重，得到一份引援排序名单。  
它更像一个可解释的球探打分系统。

第三个是 **Player Profile Dashboard**。  
这个页面会展示球员的基本资料、市场价值历史和热力图。  
它的作用是把推荐结果进一步解释清楚，让用户不仅知道“推荐了谁”，还能看到这个球员是谁、价值变化怎么样、在场上主要活动在哪里。

---

在工程层面，这个项目还有一个重要升级，就是开始接入 BigQuery。

`Player Profile Dashboard` 现在已经支持直接从 BigQuery 实时查询球员资料、身价历史和热力图。  
训练脚本也已经支持直接从 BigQuery 特征表读取训练数据。  
这使得项目从原来主要依赖本地 parquet 文件的 demo，逐步升级成一个能够连接云端数据源的原型系统。

---

最后总结一下，我认为这个项目的价值不只是训练了一个 autoencoder，而是完成了一个完整的球探分析闭环：

- 有 baseline，可解释
- 有 learned model，可做表示学习
- 有引援排序，可支持业务决策
- 有画像页，可解释推荐结果
- 有 BigQuery 集成，开始接近真实数据工作流

后续如果继续优化，我会优先考虑三件事：

- 扩大训练数据规模
- 继续分析 embedding 相似度分布
- 把整个 Streamlit 应用进一步部署到云端

谢谢。
