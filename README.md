Yang Yang, De-Chuan Zhan, Yin Fan, Yuan Jiang, and Zhi-Hua Zhou. Deep Learning for Fixed Model Reuse. Proceedings of the 31st AAAI Conference on Artificial Intelligence (AAAI-2017), San Francisco, CA. 2017.

## FMR模型

基本实现与论文中一致，实验上略有不同。

使用的数据集为wiki数据集，原文中也有，但是原文中是image模型为迁移目标模型，text提供固定特征。而此处为image提供固定特征，text为目标模型。

主要有几个可调参数，代码中并未完全封装。

base_model可以是任意torch.nn.Module，提供可训练的特征，在本代码中为一个文本CNN模型。

每次knockdown的特征数量为m，以及每隔多少epoch进行一次kd，这两个参数对FMR的效果比较重要。

FMR的几个loss中，分类的权重一般不调，对于权重的限制loss L_{reg}由于在weight decay中已经有体现，可以为0，也可以作为参数调整。对于固定特征的重构loss权重可调，但是实验中没有调。

该代码直接运行得到以下一组实验结果

FMR 0.563 0.573 0.597 0.575 0.553 0.585 average = 0.57433
no FRM 0.567 0.569 0.559 0.585 0.541 0.538 average = 0.55983

可以看到平均情况下使用FMR比不使用要略好，于论文中基本一致，FMR的具体优劣还要看数据集而定。

数据处理方面只要dataloader能够将数据和特征对应输出就行。

数据集下载地址：http://www.svcl.ucsd.edu/projects/crossmodal/、

## 关于数据处理

由于wiki是一个文本图像双模态的数据，图像没什么需要特殊处理的，文本需要下载预训练的词向量。

在本实验中，图像采用vgg19提取特征，特征维度为4096，可以实现处理好。

文本模型采用TextCNN，将图像特征作为固定特征训练文本模型。注意：与文章中正好相反，文章中提取了文本特征，训练图像模型。

数据集参考dataset.py

代码中image vgg19 bn.pkl即为预处理好的图像特征。

data all.json为预处理好的数据信息，主要是每个数据的文本和标签，文本已经处理成了整数的形式（每个整数和一个单词对应），与模型中的embedding一一对应。

对于文本中有多个句子的情况，直接拼接。

如果要用到别的数据集上，dataset的实现和一般的数据没有多大不同，只要固定特征和原始数据对应即可。
