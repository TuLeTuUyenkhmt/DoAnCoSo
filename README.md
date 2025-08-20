**Overview**
This repository contains the source code, documentation, and results from my project titled "Exploring Convolutional Neural Networks" (Tìm Hiểu Mạng Nơ-ron Tích Chập). The project provides a comprehensive study of neural networks, focusing on Convolutional Neural Networks (CNNs) for image processing and classification tasks. It combines theoretical foundations with practical experiments to demonstrate the evolution and application of CNN architectures.
The project was completed under the supervision of Dr. Nguyen Van Duy at Phenikaa University, Faculty of Information Technology, in August 2025.

**Theoretical Foundations:**
Introduction to Perceptrons and basic neural network concepts.
Gradient Descent optimization methods for single and multi-variable functions.
Neural Network architectures: Feed-forward networks, weight initialization, regularization, loss functions, and backpropagation.
Convolutional Neural Networks: Layers (convolutional, pooling, fully connected), activation functions, and historical development (e.g., LeNet-5, AlexNet, VGG-16, GoogLeNet/Inception, ResNet-50, DenseNet).


**Practical Experiments:**
Dataset: COVID-Classifier (images of COVID-19 X-rays classified into categories like Normal, COVID, Pneumonia).
Models Implemented: ResNet and VGGNet-16.
Training: Models trained for 300 epochs.
Results: Comparative analysis showing accuracy, loss curves, and performance metrics (e.g., ResNet achieved higher accuracy than VGGNet-16 on the test set).
Tools: Python with TensorFlow/Keras (or PyTorch), NumPy, Matplotlib for visualization.

**Technologies Used**
Programming Language: Python
Libraries: TensorFlow/Keras (for model building and training), NumPy, Pandas, Matplotlib/Seaborn (for data processing and visualization)
Environment: Jupyter Notebook for experiments
Data Handling: Image preprocessing, augmentation, and splitting into train/test sets

**Results and Insights**
ResNet outperformed VGGNet-16 in terms of convergence speed and final accuracy (detailed in the thesis PDF).
The project highlights challenges like overfitting and the benefits of advanced architectures (e.g., residual connections in ResNet).
Visualizations include loss/accuracy curves, confusion matrices, and sample predictions.

**How to Run**
Clone the repository: git clone https://github.com/yourusername/cnn-research-project.git
Install dependencies: pip install -r requirements.txt
Download the COVID-Classifier dataset (link in data/README.md).
Run the Jupyter Notebook: jupyter notebook experiments.ipynb for training and evaluation.

**Skills Demonstrated**
Deep Learning fundamentals and CNN implementation.
Model optimization, evaluation, and comparison.
Data preprocessing and visualization.
Research and technical writing
