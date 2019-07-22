###[Attention Is All You Need]论文笔记
主流的序列到序列模型都是基于含有encoder和decoder的复杂的循环或者卷积网络。而性能最好的模型在encoder和decoder之间加了attentnion机制。本文提出一种新的网络结构，摒弃了循环和卷积网络，仅基于attention机制。  

self-attention是一种attention机制，它是在单个序列中计算每个位置与其他不同位置关系从而计算序列。Transformer是第一个完全依靠self-attention机制来计算输入和输出表示。

###模型架构
![](http://ww4.sinaimg.cn/large/006tNc79ly1g57r7e58dkj30910e9tb0.jpg) 
####encoder
编码器由6个完全相同的层堆叠而成，每一层有两个子层，第一层是multi-head self-attention机制，第二层是简单的、位置完全连接的前馈神经网络。对每个子层都使用残差网络连接[必要性]，接着进行Layer Normalization。也就是说，每个子层的输出都是LayerNorm(x + Sublayer(x)), 其中Sublayer(x)是具体子层的具体实现函数。为了方便残差连接，模型中所有子层以及嵌入层产生的输出维度都是dmodel = 512。  
####decoder
解码器同样由N=6个完全相同的层堆叠而成。除了编码器的两个子层之外，在解码器中还插入第三个子层, 该层对编码器的输出进行multi-head self-attention（中间部分),与编码器类似，解码器每个子层采用残差连接，并加LayerNormalization。在解码器中的self-attention子层需要修改，因为后面位置是不可见。[修改方式？]

####attention
attention可以描述为将query和一组key-value对映射到输出，query,key, value都是向量。输出就是value的加权和，每部分的权重通过query和key之间点积或其他运算而来。
![](http://ww3.sinaimg.cn/large/006tNc79ly1g57ry47a4lj30la0c6q6h.jpg)
#####Scaled Dot-Product Attention（缩放的点积attention)
图二左边所示为Scaled Dot-Product Attention。输入包含query、dk维的keys和dv维的values。我们通过计算query和所有的keys的点积，每一个再除以根号dk，最后使用softmax获取每一个value的权重。[为什么除以dk?]
在实际中，可以通过使用矩阵相乘的方式同时计算一组query,只需将query，keys,values打包成一个矩阵Q，K， V即可。  
![](http://ww1.sinaimg.cn/large/006tNc79ly1g57s8ky2lfj30fn01imxn.jpg)
有两种attention方法，一种是加法[需要调研]，另一种是点积。  
加法是使用含有一个隐藏层的前馈神经网络，与加法attention相比，点积在时间和空间上都很高效，因为他可以通过矩阵方式实现优化。  
#####Multi-Head Attention


#### 基于位置的前馈神经网络


###参考
[1] [Dissecting BERT Part 1: The Encoder](https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3) 
