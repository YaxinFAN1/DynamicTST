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

# parsing
 
 Sabert + SSA很难识别关系，only sabert效果好一些，但是也很差 基本rel 性能是70不到

 最后还得是BERT+GRU+SSA
 
 bert SSA with GRU

 bert SSA without GRU
 80.38 57.22
 80.47 57.00

 # 加不加GRU几乎没啥影响


 # addressee recognition 

 hu 

 bert + SSA withoutGRU dev 上的效果最好是 Pat1 is 0.9114266891149989, SessAcc is 0.66 因为只跑了五轮，可能结果还能继续更好，大概Pat1 能跑到94左右
 SAbert +SSA 几乎优化很慢


 # Speaker identification
 看起来一对多的优化性能并不是很好，测试集合上大概只有40 P@1
BERT+SSA 表现，无论一对多还是多对一表现好像都很差
(性能差的原因是除了bug，是用的hu_AR的训练集训练，然后测试的hu_si)
20240316晚上八点修复了bug。last speaker 使用None进行mask。
 ## SABERT only的表现，一对一

 # 20240319

 只找ST的子网络，比找BERT+ST的子网络效果要好