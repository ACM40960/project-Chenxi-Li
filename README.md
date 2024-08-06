# 图像分类模型

本项目旨在使用预训练的 DenseNet121 模型对图像进行分类，区分黑熊（black bear）和纽芬兰犬（newfoundland）。

## 项目简介

本项目使用 TensorFlow 和 Keras 库构建和训练深度学习模型，利用预训练的 DenseNet121 模型进行图像分类任务。项目包含数据预处理、模型训练、模型评估和结果可视化等部分。

## 安装依赖

请确保已安装以下库：

```sh
pip install tensorflow pandas numpy matplotlib scikit-learn seaborn colorama
```

## 数据集说明

将黑熊和纽芬兰犬的图像分别放在 black bear 和 newfoundland 目录下。未见数据集放在 unseen_data 目录下。

## 模型训练

模型训练的核心步骤包括加载数据、创建数据生成器、构建模型和训练模型。

## 模型评估

训练完成后，我们使用以下方法评估模型：

绘制训练和验证曲线
<img src="https://github.com/ACM40960/project-Chenxi-Li/blob/main/images/training_history.png" alt="Model Structure" width="800" height="300"/>

生成混淆矩阵
<img src="https://github.com/ACM40960/project-Chenxi-Li/blob/main/images/confusion_matrix.png" alt="Model Structure" width="800" height="500"/>

绘制 ROC 曲线
<img src="https://github.com/ACM40960/project-Chenxi-Li/blob/main/images/roc_curve.png" alt="Model Structure" width="600" height="500"/>
## 结果展示



## 分类报告

|               | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| black bear    | 1.00      | 1.00   | 1.00     | 35      |
| newfoundland  | 1.00      | 1.00   | 1.00     | 40      |
|               |           |        |          |         |
| accuracy      |           |        | 1.00     | 75      |
| macro avg     | 1.00      | 1.00   | 1.00     | 75      |
| weighted avg  | 1.00      | 1.00   | 1.00     | 75      |


## 模型结构
<img src="https://github.com/ACM40960/project-Chenxi-Li/raw/main/images/model_structure.png" alt="Model Structure" width="400" height="600"/>

## 分类结果
未见数据集上的分类结果：

## 使用方法
克隆项目：
git clone https://github.com/your_username/your_project.git
cd your_project

安装依赖：
pip install -r requirements.txt

运行训练代码：
python train_model.py

使用保存的模型进行预测：
python predict.py --data_path path_to_unseen_data

## 贡献
欢迎贡献者！如果你有任何改进建议或发现了问题，请提交 issue 或 pull request。

## 许可证
本项目基于 MIT 许可证进行发布。
