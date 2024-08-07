# Black Bear vs Newfoundland Image Classification

<img src="https://github.com/ACM40960/project-Chenxi-Li/blob/main/images/densenet121_logo_.png" alt="DenseNet121 Logo" width="400" height="300"/>


## Introduction

This project aims to classify images of black bears and Newfoundland dogs using a deep learning model built on DenseNet121. The project involves data loading, model building, training, evaluation, and visualization of results.

The core algorithm is a transfer learning model based on DenseNet121, and BatchNormalization and Dropout layers are added to it to improve the generalization ability of the model and prevent overfitting.

## 模型结构
<img src="https://github.com/ACM40960/project-Chenxi-Li/raw/main/images/model_structure.png" alt="Model Structure" width="400" height="600"/>

The model uses the pre-trained DenseNet121 architecture for classifying images of black bears and Newfoundland dogs. It starts with an input layer for images of shape (224, 224, 3), followed by DenseNet121 which outputs features of shape (7, 7, 1024).

A BatchNormalization layer stabilizes and accelerates training, followed by a Dropout layer to prevent overfitting. The output is then flattened into a one-dimensional vector.

A Dense layer with 128 units and ReLU activation processes the flattened features. Another Dropout layer provides additional regularization. The final Dense layer with 2 units and softmax activation outputs the probability distribution for the two classes.

This architecture combines DenseNet121's feature extraction with custom layers to improve stability, prevent overfitting, and ensure effective classification.

## Data

data/black bear: Contains images of black bears.

data/newfoundland: Contains images of Newfoundland dogs.

unseen_data: Contains unseen images for model prediction.

## 模型训练

模型训练的步骤包括加载数据、创建数据生成器、构建模型和训练模型。

1.加载数据

从指定目录中加载黑熊和纽芬兰犬的图像，并将其分为训练集、验证集和测试集。
```python
# 加载数据并分割为训练集、验证集和测试集
black_bear_dir = "data/black bear"
newfoundland_dir = "data/newfoundland"
# 加载并组合数据
filepaths, labels = load_data(black_bear_dir, 'black bear') + load_data(newfoundland_dir, 'newfoundland')
df = pd.DataFrame({'filepath': filepaths, 'label': labels})
# 分割数据集
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)
```

2.创建数据生成器

使用 ImageDataGenerator 创建训练、验证和测试数据的生成器，进行数据增强和预处理。
```python
# 创建数据生成器
train_generator, val_generator, test_generator = create_generators(train_df, val_df, test_df)

```
3.构建模型

基于预训练的 DenseNet121 模型，构建一个自定义的顺序模型。DenseNet121 作为基础模型，并添加 BatchNormalization 层、Dropout 层和全连接层。
```
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = build_model(base_model, num_classes=len(train_generator.class_indices))
```
4.训练模型

使用训练和验证数据生成器训练模型，并使用 EarlyStopping 回调函数防止过拟合。
```
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(train_generator, validation_data=val_generator, epochs=15, callbacks=[early_stopping])
```

## 模型评估
The model's performance is evaluated on the validation set, with metrics including loss and accuracy.
```
validation_loss, validation_accuracy = model.evaluate(val_generator)
print("Validation Loss:", validation_loss)
print("Validation Accuracy:", validation_accuracy)
model.save('model/my_model.h5')
```

Visualization：

Training History: Accuracy and loss plots for training and validation sets.
<img src="https://github.com/ACM40960/project-Chenxi-Li/blob/main/images/training_history.png" alt="Model Structure" width="800" height="300"/>

Confusion Matrix: Heatmap to show the confusion matrix.
<img src="https://github.com/ACM40960/project-Chenxi-Li/blob/main/images/confusion_matrix.png" alt="Model Structure" width="800" height="500"/>

ROC Curve: ROC curves and AUC scores for each class.


<img src="https://github.com/ACM40960/project-Chenxi-Li/blob/main/images/roc_curve.png" alt="Model Structure" width="600" height="500"/>


分类报告

|               | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| black bear    | 0.98      | 1.00   | 0.99     | 43      |
| newfoundland  | 1.00      | 0.99   | 0.99     | 75      |
|               |           |        |          |         |
| accuracy      |           |        | 0.99     | 118     |
| macro avg     | 0.99      | 0.99   | 0.99     | 118     |
| weighted avg  | 0.99      | 0.99   | 0.99     | 118     |



## Usage

Classifying Unseen Images
A function is provided to preprocess and classify images from the unseen dataset, displaying the predictions.

<img src="https://github.com/ACM40960/project-Chenxi-Li/blob/main/images/unseen_data_predictions.png" alt="Model Structure" width="1000" height="700"/>

## 使用方法
克隆项目：
```bash
git clone https://github.com/ACM40960/project-Chenxi-Li.git
cd project-Chenxi-Li
```
安装依赖：
```sh
pip install tensorflow pandas numpy matplotlib scikit-learn seaborn colorama
```

运行代码
在Jupyter Notebook中，打开 Final Project(pure code).ipynb 文件，并按照顺序运行所有单元格。

## 贡献
欢迎贡献者！如果你有任何改进建议或发现了问题，请提交 issue 或 pull request。

## 许可证
本项目基于 MIT 许可证进行发布。
