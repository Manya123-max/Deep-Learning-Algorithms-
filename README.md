# Deep Learning Algorithms

An advanced collection of Deep Learning implementations developed during my Master's degree program. This repository demonstrates proficiency in neural networks, computer vision, sequence modeling, and generative models using TensorFlow/Keras and PyTorch.

## 📚 About This Repository

This repository contains comprehensive implementations of state-of-the-art deep learning algorithms, from fundamental neural networks to advanced architectures like GANs and Transformers. Each notebook includes theoretical concepts, practical implementations, and real-world applications.

## 🧠 Deep Learning Architectures Implemented

### 1. **Feedforward Neural Networks**
- **[DeepLab1.ipynb](DeepLab1.ipynb)** - Introduction to Neural Networks
  - Perceptrons and activation functions
  - Backpropagation algorithm
  - Multi-layer perceptrons (MLP)
  - Forward and backward propagation

### 2. **Advanced Neural Network Techniques**
- **[DeepLab2.ipynb](DeepLab2.ipynb)** - Optimization & Training
  - Gradient descent variants (SGD, Adam, RMSprop)
  - Batch normalization
  - Dropout and early stopping
  - Learning rate scheduling

### 3. **Convolutional Neural Networks (CNN)**
- **[DeepLab3_CNN.ipynb](DeepLab3_CNN.ipynb)** - Computer Vision
  - Convolutional layers and pooling
  - Feature extraction and visualization
  - Image classification tasks
  - Architecture design principles

### 4. **Regularization Techniques**
- **[DeepLab4_Regularization.ipynb](DeepLab4_Regularization.ipynb)** - Preventing Overfitting
  - L1/L2 regularization
  - Dropout strategies
  - Data augmentation
  - Cross-validation techniques

### 5. **Transfer Learning**
- **[DeepLab5_TransferLearning.ipynb](DeepLab5_TransferLearning.ipynb)** - Pre-trained Models
  - VGG, ResNet, Inception architectures
  - Fine-tuning strategies
  - Feature extraction
  - Domain adaptation

### 6. **Autoencoders**
- **[DeepLab6_Encoders.ipynb](DeepLab6_Encoders.ipynb)** - Dimensionality Reduction
  - Vanilla autoencoders
  - Denoising autoencoders
  - Sparse autoencoders
  - Latent space representation

- **[DL_7_StackedAE.ipynb](DL_7_StackedAE.ipynb)** - Deep Autoencoders
  - Stacked autoencoder architecture
  - Layer-wise pre-training
  - Deep feature learning
  - Reconstruction analysis

### 7. **Generative Adversarial Networks (GAN)**
- **[DepLab8_GAN.ipynb](DepLab8_GAN.ipynb)** - Generative Models
  - Generator and discriminator networks
  - Adversarial training
  - Mode collapse handling
  - Image synthesis

### 8. **Recurrent Neural Networks (RNN)**

#### Long Short-Term Memory (LSTM)
- **[Deep9_LSTM_PartA.ipynb](Deep9_LSTM_PartA.ipynb)** - LSTM Fundamentals
  - LSTM cell architecture
  - Forget, input, and output gates
  - Sequence prediction
  - Time series forecasting

- **[Deep9_LSTM_PartB.ipynb](Deep9_LSTM_PartB.ipynb)** - Advanced LSTM
  - Bidirectional LSTM
  - Stacked LSTM layers
  - Attention mechanisms
  - Text generation

#### Gated Recurrent Unit (GRU)
- **[Deep10_GRU.ipynb](Deep10_GRU.ipynb)** - Simplified RNN Architecture
  - GRU cell structure
  - Update and reset gates
  - LSTM vs GRU comparison
  - Sequence modeling

## 🛠️ Technologies & Frameworks

### Core Libraries
- **TensorFlow 2.x** - Deep learning framework
- **Keras** - High-level neural networks API
- **PyTorch** - Dynamic computational graphs
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation

### Visualization & Analysis
- **Matplotlib** - Plotting and visualization
- **Seaborn** - Statistical graphics
- **TensorBoard** - Training visualization
- **Plotly** - Interactive plots

### Additional Tools
- **OpenCV** - Computer vision operations
- **PIL/Pillow** - Image processing
- **scikit-learn** - Preprocessing and metrics
- **Google Colab** - Cloud computing platform

## 📊 Key Concepts Mastered

### Neural Network Fundamentals
- Activation functions (ReLU, Sigmoid, Tanh, Softmax)
- Loss functions (Cross-entropy, MSE, MAE)
- Optimization algorithms (SGD, Adam, Adagrad)
- Backpropagation and gradient descent

### Computer Vision
- Convolutional operations
- Pooling strategies (Max, Average, Global)
- Object detection and classification
- Image segmentation basics

### Sequence Modeling
- Time series prediction
- Natural language processing
- Sentiment analysis
- Text generation and translation

### Generative Models
- Latent space manipulation
- Adversarial training dynamics
- Image generation and enhancement
- Style transfer concepts

### Model Optimization
- Hyperparameter tuning
- Regularization techniques
- Transfer learning strategies
- Model compression

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow keras torch torchvision numpy pandas matplotlib seaborn opencv-python pillow scikit-learn
```

### Running the Notebooks

1. **Clone the repository**
```bash
git clone https://github.com/Manya123-max/Deep-Learning-Algorithms-.git
cd Deep-Learning-Algorithms-
```

2. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

3. **Or open directly in Google Colab**
- Each notebook is optimized for Colab
- GPU acceleration recommended for training
- Runtime → Change runtime type → GPU

## 📈 Learning Progression

```
DeepLab1 (Neural Networks Basics)
    ↓
DeepLab2 (Optimization)
    ↓
DeepLab3 (CNNs)
    ↓
DeepLab4 (Regularization)
    ↓
DeepLab5 (Transfer Learning)
    ↓
DeepLab6 & DL_7 (Autoencoders)
    ↓
DepLab8 (GANs)
    ↓
Deep9 (LSTM) → Deep10 (GRU)
```

## 💡 Project Highlights

### 1. **Convolutional Neural Networks**
- Implemented custom CNN architectures for image classification
- Achieved 90%+ accuracy on benchmark datasets
- Explored different filter sizes and pooling strategies

### 2. **Transfer Learning**
- Fine-tuned pre-trained models (VGG16, ResNet50)
- Demonstrated significant reduction in training time
- Applied to custom datasets with limited data

### 3. **Generative Adversarial Networks**
- Built generator and discriminator from scratch
- Explored training dynamics and stability
- Generated synthetic images

### 4. **Sequence Models (LSTM/GRU)**
- Implemented bidirectional LSTM for sentiment analysis
- Compared LSTM vs GRU performance
- Applied to time series forecasting

### 5. **Autoencoders**
- Created stacked autoencoders for feature learning
- Implemented denoising autoencoders
- Visualized latent space representations

## 🎯 Applications & Use Cases

- **Computer Vision**: Image classification, object detection
- **Natural Language Processing**: Sentiment analysis, text generation
- **Time Series**: Stock price prediction, weather forecasting
- **Generative AI**: Image synthesis, style transfer
- **Anomaly Detection**: Fraud detection using autoencoders
- **Recommendation Systems**: Collaborative filtering with neural networks

## 📝 Repository Structure

```
Deep-Learning-Algorithms-/
│
├── Fundamentals/
│   ├── DeepLab1.ipynb (Neural Networks)
│   └── DeepLab2.ipynb (Optimization)
│
├── Computer Vision/
│   ├── DeepLab3_CNN.ipynb
│   ├── DeepLab4_Regularization.ipynb
│   └── DeepLab5_TransferLearning.ipynb
│
├── Generative Models/
│   ├── DeepLab6_Encoders.ipynb
│   ├── DL_7_StackedAE.ipynb
│   └── DepLab8_GAN.ipynb
│
├── Sequence Models/
│   ├── Deep9_LSTM_PartA.ipynb
│   ├── Deep9_LSTM_PartB.ipynb
│   └── Deep10_GRU.ipynb
│
└── README.md
```

## 🔬 Research & Experimentation

Each notebook includes:
- ✅ Theoretical background and mathematics
- ✅ Step-by-step implementation
- ✅ Visualization of results
- ✅ Performance metrics and evaluation
- ✅ Hyperparameter tuning experiments
- ✅ Comparative analysis

## 🏆 Skills Demonstrated

- Deep understanding of neural network architectures
- Proficiency in TensorFlow and Keras
- Ability to implement research papers
- Strong debugging and optimization skills
- Experience with GPU-accelerated training
- Knowledge of best practices in deep learning
- Capability to work with large-scale datasets

