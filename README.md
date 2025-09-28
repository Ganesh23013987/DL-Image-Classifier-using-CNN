# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Define the problem
Classify handwritten digits (0–9) using the MNIST dataset.

### STEP 2: Import libraries and dataset
Import required libraries such as TensorFlow/Keras, NumPy, and Matplotlib.
Load the MNIST dataset using keras.datasets.mnist.load_data().


### STEP 3: Preprocess the data
Normalize the image pixel values (scale from 0-255 to 0-1).
Reshape the images to match CNN input shape.


### STEP 4: Build the CNN model
Initialize a Sequential model.
Add convolutional layers with activation (ReLU), followed by pooling layers.
Flatten the output and add Dense layers.
Use a softmax layer for classification.

### STEP 5: Compile and train the model
Compile the model with an optimizer (e.g., Adam), loss function (e.g., categorical crossentropy), and metrics (accuracy).
Train the model using training data and validate using validation split or test data.


### STEP 6: Evaluate and visualize results
Evaluate the model on test data and print accuracy.
Plot training/validation loss and accuracy curves.
Optionally, display a confusion matrix or sample predictions.



## PROGRAM

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

train_model(model, train_loader)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: Kanagavel A K')
    print('Register Number: 212223230096')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name: Kanagavel A K')
    print('Register Number: 212223230096')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name: Kanagavel A K')
    print('Register Number: 212223230096')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

test_model(model,test_loader)

import matplotlib.pyplot as plt
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
    class_names = dataset.classes

    # Display the image
    print('Name: Kanagavel A K')
    print('Register Number: 212223230096')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')


predict_image(model, image_index=80, dataset=test_dataset)
```

### Name: GANESH D

### Register Number: 212223240035

### OUTPUT

## Training Loss per Epoch
<img width="461" height="72" alt="Screenshot 2025-09-23 135441" src="https://github.com/user-attachments/assets/ca0689f7-efed-44f0-b284-ed0926a2dbfc" />




## Confusion Matrix
<img width="442" height="135" alt="image" src="https://github.com/user-attachments/assets/f125332a-4457-4e0c-90e6-d636b16b8d4c" />


<img width="886" height="614" alt="image" src="https://github.com/user-attachments/assets/87f7cb72-cacf-4119-a917-f265fe82eb27" />

## Classification Report
<img width="615" height="347" alt="image" src="https://github.com/user-attachments/assets/80f47e65-82ed-4515-9ad5-ee0aa1dfb60a" />


### New Sample Data Prediction
<img width="540" height="502" alt="image" src="https://github.com/user-attachments/assets/83f228c6-9da6-4a5e-8511-a93bdd2b121c" />



## RESULT
The CNN model achieved high accuracy on the MNIST dataset, with training and validation losses showing good convergence. The classification report and confusion matrix confirmed strong performance across all digits, with minimal misclassifications. Overall, the model performs reliably in handwritten digit recognition.
