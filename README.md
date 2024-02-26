# Transfer Learning for Image Classification

This README demonstrates the use of transfer learning for image classification using a pre-trained MobileNetV2 model.

## Dataset and Preprocessing

### Dataset
- The dataset is provided in a CSV file (`bald_people.csv`) containing image paths and corresponding labels.
- Images are stored in the directory `./images/`.

### Preprocessing
- Image data augmentation techniques are applied using `ImageDataGenerator` to increase the diversity of the training dataset.
- The dataset is split into training and validation sets with a validation split of 20%.

## Model Architecture

- The pre-trained MobileNetV2 model is used as the base model with weights pre-trained on the ImageNet dataset.
- The weights of the base model are frozen to prevent them from being updated during training.
- A custom classification head is added on top of the base model, consisting of a Flatten layer followed by two Dense layers with ReLU and softmax activations, respectively.

## Training

- The model is compiled with the Adam optimizer and categorical cross-entropy loss function.
- The training is performed using the `fit` method with the training and validation generators.

## Parameters and Hyperparameters

- Number of Classes: 6
- Image Size: (224, 224)
- Batch Size: 42
- Epochs: 20

## Dependencies

- pandas
- Keras
