# GR_CNN

基于卷积神经网络（CNN）的手势识别（石头、剪刀、布）

## 文件树

│  load_model.py  #加载模型、测试模型
│  main.py #主文件
│  README.md
│  run.sh #调参脚本
│
├─.vscode
│      settings.json
│
├─gitImg
│      img.png
│      others.png
│      paper.png
│      rock.png
│      scissors.png
│
├─imgCollect
│      collectCode.cpp #采集数据集代码
│
├─model
│      CNN.pth #训练好的模型
│
├─results #结果文件夹
│  ├─epoch_100_lr_0.001_batch_size_train_128_2021-11-22 12-42-10
│  │      CNN.pth
│  │      log.txt
│  │      loss.png
│  │      Test_acc.png
│  │
│  ├─epoch_150_lr_0.001_batch_size_train_128_2021-11-22 14-19-51
│  │      CNN.pth
│  │      log.txt
│  │      loss.png
│  │      Test_acc.png
│  │
│  └─epoch_60_lr_0.001_batch_size_train_128_2021-11-22 09-28-19
│          CNN.pth
│          log.txt
│          loss.png
│          Test_acc.png
│
├─src
│      model.py #CNN模型
│      __init__.py
│
└─utils
datasets.py #加载数据集
evaluate.py #测试代码
others.py
train.py #训练代码
__init__.py

## 训练结果

参数：epoch 60, batch_size 128, lr 0.0001

![img](gitImg\loss.png "损失图")
![img](gitImg\Test_acc.png "测试准确率图")

## 测试效果

![gif](gitImg\1.gif "测试结果")
"# GR_CNN" 
"# GR_CNN" 
"# GR_CNN" 
