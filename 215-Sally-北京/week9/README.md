### 深度学习开源框架
1. 深度学习框架包含5个核心组件：
    1. 张量（Tensor）
    2. 基于张量的各种操作（Operation）
        1. 矩阵乘法
        2. 卷积
        3. 池化
        4. LSTM
    3. 计算图（Computation Graph）
        1. 概念：表示计算过程的流程图
        2. 作用：分析、理解神经网络
    4. 自动微分（Automatic Differentiation）工具
    5. BLAS、cuBLAS、cuDNN等拓展包
2. 主流深度学习框架4大阵营，前两个用的最多
    1. TensorFlow，前端框架Keras，背后巨头Google；
        1. TensorFlow里也有Keras
    2. PyTorch，前端框架FastAI，背后巨头Facebook；
    3. MXNet，前端框架Gluon，背后巨头Amazon；
        1. nlp领域，如讯飞在用
    4. Cognitive Toolkit (CNTK)，前端框架Keras或Gluon，背后巨头Microsoft。
        1. 推荐场景使用
3. 国内深度学习框架，前两个用的最多
    1. 华为MindSpore
        1. 支持端、边、云独立的和协同的统一训练和推理框架。2020年3月28日，正式开源（半开源，未完全开源）
        2. 仅支持华为生态
    2. 百度PaddlePaddle
        1. PaddlePaddle 100% 都在Github上公开，没有内部版本。PaddlePaddle能够应用于自然语言处理、图像识别、推荐引擎等多个领域，其优势在于开放的多个领先的预训练中文模型。
    3. 阿里巴巴XDL (X-Deep Learning)
        1. 阿里妈妈将把其应用于自身广告业务的算法框架XDL (X-DeepLearning)进行开源。XDL主要是针对特定应用场景如广告的深度学习问题的解决方案，是上层高级API框架而不是底层框架。XDL需要采用桥接的方式配合使用 TensorFlow 和 MXNet 作为单节点的计算后端，XDL依赖于阿里提供特定的部署环境。
    4. 小米MACE：
        1. 它针对移动芯片特性进行了大量优化，目前在小米手机上已广泛应用，如人像模式、场景识别等。该框架采用与 Caffe2 类似的描述文件定义模型，因此它能非常便捷地部署移动端应用。目前该框架为 TensorFlow 和 Caffe 模型提供转换工具，并且其它框架定义的模型很快也能得到支持。
4. 注意区分框架、模型、算法
    1. 任何框架都可以实现特定的模型
    2. 99.9的Layer都被主流框架支持
    3. 深度学习框架和模型、算法没有直接联系，没有一一对应的关系
    4. 一个框架训练出的模型不能在另一个框架上去推理，因为存储格式不兼容
5. 深度学习框架的标准化--ONNX，
    1. 解决框架间不兼容的问题，把不同框架训练出的模型都转成此第三方格式
    2. 开放神经网络交换（ONNX，“Open Neural Network Exchange”）：ONNX最初由微软和Facebook联合发布，后来亚马逊也加入进来，并发布了V1版本，宣布支持ONNX的公司还有AMD、ARM、华为、 IBM、英特尔、Qualcomm等。
    3. ONNX是一个表示深度学习模型的开放格式。它使用户可以更轻松地在不同框架之间转移模型。例如，它允许用户构建一个PyTorch模型，然后使用MXNet运行该模型来进行推理。
