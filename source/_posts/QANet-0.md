---
title: QANet
date: 2018-09-30 18:01:32
categories:
- NLP
- machine-reading-comprehension
tags:
- NLP
- machine-reading-comprehension
---

## QANet
#### QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension


### 特点
1. 移除RNN
2. 使用卷积捕捉局部信息
3. 自注意力机制捕捉全局信息

### 模型结构
<!--more-->
![image](https://ws1.sinaimg.cn/large/006tNc79ly1fvrpc8wjfzj30rk0zw43g.jpg)
1. Embedding层
2. Embedding Encoder层
3. Context-Query Attention层
4. Model encoder层
5. Output层

#### Embedding层
利用词向量和字向量拼接的方式获得最终的词向量：
文章最大长度设为600
- 词向量
    > 预训练好的，300维的Glove
    > embedding后维度为$600\times300$
- 字向量：
    > 预处理时将每个单词截断或者是padding到16个字符
    > 每个字符表示成200维的向量 （embedding后维度变成$600\times16\times200$）
    > 做卷积filter=128,kernel=5 (卷积后维度为$600\times12\times128$)
    > 沿行做最大池化，选择所有字符中最大的向量作为单词的最终字符向量。（维度变成$600\times128$）
    
- 拼接，再经过两层Highway Network得到最终的单词向量表示，维度为$600\times328$

#### Embedding Encoder层
提取Context和query中的信息
每一个Encoder块是由卷积、Self-Attention、全连接层组成。
输入向量维数是d=328，输出d=128.
![image](https://ws2.sinaimg.cn/large/006tNc79ly1fvrpsx6edxj30k80xsmzy.jpg)
**1. Position encoding**
纯Attention模型无法捕捉序列的顺序，如果将K,V按行打乱顺序（相当于打乱句子中的次序），Attention的结果还是一样的。如果没有序列顺序信息，Attention模型顶多是一个非常精妙的“词袋模型”而已。Position Embedding是位置信息的唯一来源。

$PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})$
$PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})$
pos(i+k)可以表达成pos(i)的线性组合，对相对距离的表示非常有利
把输入加上position embedding的结果作为输出

**2. 深度可分离卷积**
经典卷积：卷积核在所有输入通道上进行卷积，并综合所有输入通道情况得到卷积结果（池化）
深度可分离卷积：卷积核对每个输入通道分别做卷积，然后对得到的新的feature map先concat起来再使用$1 \times 1$的卷积核将输出通道混合。
![image](https://ws1.sinaimg.cn/large/006tNc79ly1fvrpu79woaj310a0huq5v.jpg)
![image](https://ws2.sinaimg.cn/large/006tNc79ly1fvrpulkoc4j30uy06gwfs.jpg)

优点：把空间特征学习和通道特征学习分开，这样可以提高泛化能力和卷积效率，避免参数冗余。（Xception的基础）


用3个$3\times3$（1通道）的卷积核分别与输入的3通道的数据做卷积，得到3个feature map，然后用256个$1 \times 1$大小的卷积核（3通道）在这3个feature map上进行卷积运算，将3个通道的信息进行融合。

**3. 自注意力机制**
[Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
序列编码：RNN要逐步递归才能获得全局信息，因此一般要双向RNN才比较好；CNN事实上只能获取局部信息，是通过层叠来增大感受野；Attention的思路最为粗暴，它一步到位获取了全局信息。
![image](https://ws2.sinaimg.cn/large/006tNc79ly1fvrpvist54j30og0ss0uz.jpg)
$Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = softmax\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V}$
query和key做內积并softmax,来得到query和value之间的相似度，然后加权求和。

**Self Attention**

如果Q=K=V，那么就称为Self Attention，它的意思是直接将每个词与原来的每个词进行比较，计算出表示。也就是在序列内部做Attention，寻找序列内部的联系。

**Muilti-Head Attention**

这个是Google提出的新概念，是Attention机制的完善。不过从形式上看，它其实就再简单不过了，就是把Q,K,V通过参数矩阵映射一下，然后再做Attention，把这个过程重复做h次，结果拼接起来就行了
![image](https://ws1.sinaimg.cn/large/006tNc79ly1fvrpwc63ldj30hc0kkq4k.jpg)

所谓“多头”（Multi-Head），就是指多做几次同样的事情（参数不共享），然后把结果拼接。
一方面，从直觉上多次attention操作可以捕获更多的信息，另一方面，先进行的投影操作能把QKV映射到不同空间，也许能发现更多特征。

**4. 层归一化+残差连接**

在encoder block中，每个子层都用了layernorm和残差连接：
$Output = f(layernorm(x)) + x$
其中 f 表示encoder block 中的子层，如 depth conv, self-attention, feed-forward等。layernorm() 表示 layer normalization。

![image](https://ws3.sinaimg.cn/large/006tNc79ly1fvrpwq23s1j314m0cs0up.jpg)
BN：针对一个minibatch的输入样本，计算均值和方差，基于计算的均值和方差来对某一层神经网络的输入X中每一个case进行归一化操作。

LN: 同层神经元输入拥有相同的均值和方差，不同的输入样本有不同的均值和方差；而BN中则针对不同神经元输入计算均值和方差，同一个minibatch中的输入拥有相同的均值和方差。因此，LN不依赖于mini-batch的大小和输入sequence的深度，


#### Context-Query Attention Layer
发现context query 之间的联系，并在词的层面上，解析出query, context中关键的词语。
1. 首先计算context和query每对词之间的相似度，搞成一个相似度矩阵S，用的是BiDAF的算法：
$S_{i,j} = f(q,c ) = W_0[q,c,q\odot c]$
2. 计算context-to-query的attention A：
$A = softmax(S, axis=row) \cdot Q^T \quad \in R^{n\times d}$

> 实现中，context的长度是600，question长度是100，经过embedding encoder之后分别变成$S=600\times128$和$Q=100\times128$的表示。$S\cdot Q^T$维度是$600\times100$

> 每行是一个context中的单词w，元素值是所有的query单词对于当前文档单词w的注意力分配权值
> 用$A$的每一行去乘以Q去表达单词w
> 这样就得到了用query表达context的结果
3. 计算query-to-context的attention B：
$B = A\cdot softmax(S,axis=column)^T \cdot C^T$
(这里采用了DCN的coattention) [Dynamic Coattention Networks For Question Answering](https://arxiv.org/abs/1611.01604)
#### Model Encoder layer
从全局的层面来考虑context与query之间的关系。
输入是3个关于Context的矩阵信息：
原始Context：$C \in \mathcal{R}^{n\times d}$
Context的Attention: $A \in \mathcal{R}^{n\times d}$
Context的Coattention:$B \in \mathcal{R}^{n \times d}$ 
每个单词的编码信息是上面三个矩阵的拼接
$f(w) = [c, a, c \odot a, c \odot b]$
一个有7个Encoder-Block，每个Encoder-Block：2个卷积层、Self-Attention、FFN。其它参数和Embedding Encoder一样。
一共有3个Model-Encoder，共享所有参数。输出依次为$M_0,M_1,M_2$.

#### Output layer
解析answer在context中的位置
$pos^{start} = softmax(W_{start} [M_0; M_1]),\quad  pos^{end} = softmax(W_{end}[M_0; M_2])$

#### Loss function
$L(\theta) = -  \frac{1}{N}\sum_{i}^N\left[\log(p_{y_i^{start}}^{start}) + \log(p_{y_i^{end}}^{end})\right]$