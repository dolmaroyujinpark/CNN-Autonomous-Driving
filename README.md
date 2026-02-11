# CNN-Autonomous-Driving

**Research Project**: Convolutional Neural Network (CNN) based Autonomous Driving Steering Angle Prediction

## 📋 Project Overview

This is a **research project** focused on developing a steering angle prediction model for autonomous driving using Convolutional Neural Networks (CNN). The project explores deep learning techniques to predict vehicle steering angles from image data.

**연구용 프로젝트**: 본 프로젝트는 합성곱 신경망(CNN)을 활용한 자율주행 조향각 예측 연구 프로젝트입니다. 이미지 데이터로부터 차량의 조향각을 예측하는 딥러닝 기술을 연구합니다.

## 🎯 Research Focus

- **Steering Angle Prediction**: Development of CNN models to predict steering angles from visual input
- **Deep Learning for Autonomous Driving**: Exploring neural network architectures for self-driving applications
- **Image-based Control**: Research on image processing and control systems for autonomous vehicles

## 🚀 Features

- Custom CNN architectures for steering angle prediction
- Image preprocessing and dataset management utilities
- Training scripts with various model configurations (4train, 6train, 8train variants)
- Dataset creation and labeling tools
- Support for image classification and cropping operations

## 📁 Project Structure

```
.
├── 4train_upgrade2.py      # Training script with 4-layer configuration
├── 6train_upgrade2.py      # Training script with 6-layer configuration
├── 8train_upgrade2.py      # Training script with 8-layer configuration
├── Dataset.py              # Dataset handling utilities
├── dataset_make.py         # Dataset creation script
├── csvmake_yujin2.py       # CSV label file generation
├── crop_images.py          # Image cropping utility
├── image_class.py          # Image classification tools
├── binary2.py              # Binary classification script
└── sign_capture.py         # Sign detection and capture
```

## 🛠️ Environment Setup

### Requirements

- Python 3.x
- PyTorch
- torchvision
- pandas
- numpy
- Pillow (PIL)
- matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/dolmaroyujinpark/CNN-Autonomous-Driving.git
cd CNN-Autonomous-Driving

# Install dependencies
pip install torch torchvision pandas numpy pillow matplotlib
```

## 💻 Usage

### Training a Model

```bash
# Train with 4-layer configuration
python 4train_upgrade2.py

# Train with 6-layer configuration
python 6train_upgrade2.py

# Train with 8-layer configuration
python 8train_upgrade2.py
```

### Preparing Dataset

```bash
# Create dataset
python dataset_make.py

# Generate CSV labels
python csvmake_yujin2.py

# Crop images
python crop_images.py
```

## 📊 Model Architecture

The project implements various CNN architectures with different layer depths:
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for steering angle prediction
- Support for multi-class classification (straight, left, right, stop)

## 📝 Dataset Format

The project uses CSV files for label management with the following structure:
- Column 1: Image file path
- Column 2: Label (steering command: straight, left, right, stop)

Images are preprocessed to standardized dimensions (e.g., 70x320 or 411x231).

## 🔬 Research Notes

This is an **experimental research project** aimed at exploring CNN-based approaches for autonomous driving. The models and techniques are under active development and refinement.

## 👨‍💻 Developer

**Developer**: dolmaroyujinpark

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This research project builds upon various open-source autonomous driving and deep learning resources.

---

**Note**: This is a research and educational project. It is not intended for deployment in real-world autonomous driving systems.
