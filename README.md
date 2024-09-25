🌟 CIFAR-100 Classification Using CNN, ANN, and Transfer Learning 🌟
This repository demonstrates image classification using both Convolutional Neural Networks (CNN) and Artificial Neural Networks (ANN) on the CIFAR-100 dataset. It also showcases transfer learning and fine-tuning using a pre-trained model. 🚀


📊 Dataset
The CIFAR-100 dataset consists of 60,000 32x32 color images in 100 classes, with 600 images per class. There are 50,000 training images and 10,000 test images.

🎯 Class Distribution
The CIFAR-100 dataset contains 100 classes such as apple, airplane, bicycle, etc. Each image is 32x32 in RGB format.

🧠 Models
🔢 ANN (Artificial Neural Network):

The ANN model is built using fully connected Dense layers and is trained on the CIFAR-100 dataset.
The architecture consists of a simple feed-forward network with ReLU activation followed by Softmax for classification.
🌀 CNN (Convolutional Neural Network):

The CNN model uses multiple convolutional layers followed by max-pooling and fully connected layers for classification.
The architecture includes:
Conv2D layers
MaxPooling2D
Fully connected Dense layers with ReLU and Softmax activation.
🌍 Transfer Learning:

Utilizes a pre-trained ResNet50 model from ImageNet.
Fine-tuned on the CIFAR-100 dataset by adding custom classification layers and training on the new dataset.
📁 Project Structure

📦 cifar100-cnn-ann
 ┣ 📜 cnn_ann_cifar100.py   # CNN, ANN, and Transfer Learning code
 ┣ 📜 README.md             # Project documentation
 ┣ 📜 requirements.txt      # Dependencies

🔧 Requirements
Install the necessary packages using pip:
pip install -r requirements.txt

Dependencies include:

TensorFlow
NumPy
Matplotlib
Pandas
🚀 Training and Evaluation
Training the ANN Model:

The ANN model is trained for 40 epochs using the Adam optimizer.
We visualize training and validation loss to track model performance.
Training the CNN Model:

The CNN model is trained for 25 epochs.
We evaluate the model on the CIFAR-100 test set and measure accuracy and loss.
Transfer Learning with ResNet50:

Fine-tuned on CIFAR-100 by retraining the top layers of ResNet50 for 15 epochs.
🧪 Results
ANN Model:
🎯 Test Accuracy: <0.4252 >

CNN Model:
🎯 Test Accuracy: <0.3179>

Transfer Learning with ResNet50:
🎯 Test Accuracy: <0.5491>

📊 Visualizations
CIFAR-100 Sample Images
