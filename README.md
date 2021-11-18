# 分数阶TBCNN（TBCNN-CAPUTO）模型实现

通过将tbcnn和caputo微分结合，用于JavaScript恶意代码检测

- tbcnn代码来自https://github.com/crestonbunch/tbcnn/blob/master/classifier/tbcnn/network.py

## 目录介绍

- ast2vec 训练词嵌入向量
- compare 传统的机器学习算法，用于比较模型效果
- data 存放数据及相关处理代码
- logs 存放tensorflow日志
- models 存放tensorflow模型
- metrics 度量指标
- tbcnn 模型实现
    - caputo 训练tbcnn-caputo模型
    - integer 训练tbcnn模型

## 训练步骤

1. 安装依赖包

    ```
   pip3 install -r requirments.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 
   ```

2. 下载样本

    ```bash
    cd data
    python3 download_sample.py
    ```
   
    样本存入`data/samples/black`和`data/samples/white`目录中
   
3. 提取js代码

    ```bash
    cd data
    python3 extract_javascript.py
    ```
   
5. 解析语法树

    ```bash
    cd data
    python3 parse_ast.py
    ```
   
5. 训练词嵌入向量

    ```bash
    cd ast2vec
    python3 train.py
    ```
   
6. 训练tbcnn-caputo

    ```bash
    cd tbcnn/caputo
    python3 train.py
    ```

## 测试与比较

### roc

```bash
cd metrics
python3 roc.py
```

### 分类指标矩阵

```bash
cd matrics
python3 matrix.py
```

# 论文链接

- [tbcnn](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.740.9656&rep=rep1&type=pdf)
- [Convolutional neural networks with fractional order gradient method](https://arxiv.org/pdf/1905.05336)

