# Black Bear vs Newfoundland Image Classification

<img src="https://github.com/ACM40960/project-Chenxi-Li/blob/main/images/densenet121_logo_.png" alt="DenseNet121 Logo" width="400" height="300"/>


## Introduction

This project aims to classify images of black bears and Newfoundland dogs using a deep learning model built on DenseNet121. The project involves data loading, model building, training, evaluation, and visualization of results.

### Dataset
- Data structure
  - data/Black bear
  - data/Newfoundland

Dataset is split as follows:

| Dataset | proportion |
| ------- |--------|
| Train | 0.64    |
 | Test | 0.16 |
 | Validation | 0.20   |

### Methodology

In this project, we utilize a transfer learning model based on DenseNet121 for the classification of images into two classes: Black Bear and Newfoundland. The core algorithm leverages the pre-trained DenseNet121 architecture as a feature extractor, combined with additional layers to enhance the model's performance. BatchNormalization and Dropout layers are added to it to improve the generalization ability of the model and prevent overfitting.

#### Model Architecture
<img src="https://github.com/ACM40960/project-Chenxi-Li/raw/main/images/model_structure.png" alt="Model Structure" width="400" height="600"/>



- The model is built on the pre-trained DenseNet121 architecture, which is used as a feature extractor.

- Additional layers are added on top of DenseNet121:

  - BatchNormalization Layer: Normalizes the output to stabilize and accelerate training.

  - Dropout Layer: Prevents overfitting by randomly setting a fraction of input units to zero.

  - Flatten Layer: Converts the feature maps into a one-dimensional vector.

  - Dense Layer (128 units): Fully connected layer with ReLU activation for further processing.

  - Dropout Layer: Additional dropout for regularization.

  - Output Dense Layer (2 units): Fully connected layer with softmax activation to output the probability distribution for the two classes.


## Model Traning

- Data Loading

Images of black bears and Newfoundland dogs are loaded and split into training, validation, and test sets.

```python
# Load and split data into training, validation, and test sets
black_bear_dir = "data/black bear"
newfoundland_dir = "data/newfoundland"

filepaths, labels = load_data(black_bear_dir, 'black bear') + load_data(newfoundland_dir, 'newfoundland')
df = pd.DataFrame({'filepath': filepaths, 'label': labels})

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)
```

- Creating Data Generators

Use ImageDataGenerator to create data generators for training, validation, and test sets, performing data augmentation and preprocessing.
```python
# Create data generators
train_generator, val_generator, test_generator = create_generators(train_df, val_df, test_df)
```
- Model Construction

Construct a custom sequential model based on the pre-trained DenseNet121 model. DenseNet121 is used as the base model, and additional BatchNormalization, Dropout, and Dense layers are added to enhance generalization and prevent overfitting.
```
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential

# Load the pre-trained DenseNet121 model without the top layer
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

# Create a custom model on top of the DenseNet121 base
model = Sequential([
    base_model,
    BatchNormalization(),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(2, activation='softmax')  # Output layer with softmax activation for 2 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
- Model Training

Train the model using the training and validation data generators, and employ an EarlyStopping callback to prevent overfitting.
```
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(train_generator, validation_data=val_generator, epochs=15, callbacks=[early_stopping])
```

## Model Evaluation
The model's performance is evaluated on the validation set, with metrics including loss and accuracy.
```
validation_loss, validation_accuracy = model.evaluate(val_generator)
print("Validation Loss:", validation_loss)
print("Validation Accuracy:", validation_accuracy)
```
### Visualization：
In this project, several visualizations are used to analyze the training process. Below are the performances of the visualizations included in this project:

#### Training History: Accuracy and loss plots for training and validation sets.
<img src="https://github.com/ACM40960/project-Chenxi-Li/blob/main/images/training_history.png" alt="Model Structure" width="800" height="300"/>

#### Confusion Matrix: Heatmap to show the confusion matrix.
<img src="https://github.com/ACM40960/project-Chenxi-Li/blob/main/images/confusion_matrix.png" alt="Model Structure" width="800" height="500"/>

#### ROC Curve: ROC curves and AUC scores for each class.

<img src="https://github.com/ACM40960/project-Chenxi-Li/blob/main/images/roc_curve.png" alt="Model Structure" width="600" height="500"/>


#### Classification Report

|               | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| black bear    | 0.98      | 0.98   | 0.98     | 43      |
| newfoundland  | 0.99      | 0.99   | 0.99     | 75      |
|               |           |        |          |         |
| accuracy      |           |        | 0.98     | 118     |
| macro avg     | 0.98      |0.98    | 0.98     | 118     |
| weighted avg  | 0.98      | 0.98   | 0.98     | 118     |



## Visualizing Predictions on Unseen Data:

A function is provided to preprocess and recognize images from the unseen dataset, displaying the predictions.

<img src="https://github.com/ACM40960/project-Chenxi-Li/blob/main/images/unseen_data_predictions.png" alt="Model Structure" width="1000" height="700"/>

## Usage

Clone the repository:
```bash
git clone https://github.com/ACM40960/project-Chenxi-Li.git
cd project-Chenxi-Li
```
Install dependencies:
```sh
pip install tensorflow pandas numpy matplotlib scikit-learn seaborn colorama
```

Run the code:
Open the Final Project(pure code).ipynb file in Jupyter Notebook and run all cells in order.

## Contributing
Contributions are welcome! If you have any suggestions for improvements or find any issues, please submit an issue or a pull request.

## License
This project is licensed under the MIT License.
