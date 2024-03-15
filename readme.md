# 动态寻找子网络的代码


# 20240314实验记录



## SABERT + SSA 的structure path 
拼接SSABERT的输出，和SSA的structure path的平均池化表示，在hu 的数据集上的response selection上优化的很快

只有structure path的平均池化表示，性能略弱

## 只使用 SSA的structure path的表示进行response selection，优化进行的很慢，目前不知道结果如何


| Type     | P@1 R2     | P@1 R10    |description|
| -------- | -------- | -------- |-------- |
|S-BERT | 行1，列2 | 行1，列3 |           |
|S-BERT & SA structure path (cat output of S-bert and path mean pooling) | 行2，列2 | 行2，列3 |
|S-BERT only SSA structure path | 行2，列2 | 行2，列3 |
| only SSA structure path | 行3，列2 | 行3，列3 |
————————————————
看起来with S-BERT with SSA structure path cat（cat S-BERT的输出和structure path 的性能最好）

实验结果表明还是凭借s-bert的输出和ssa 输出的structure path 的平均池化的拼接效果最好。只使用SSA structure path的效果最差

