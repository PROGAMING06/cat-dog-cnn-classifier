# Cat-Dog CNN Classifier

A Convolutional Neural Network (CNN) image classification project that predicts whether an uploaded image is a **Cat** or **Dog**.

The trained model is integrated into a **Streamlit web application**, allowing users to upload an image and receive a prediction with a confidence score.

## 🚀 Live Demo

👉 **[Try the Cat-Dog CNN Classifier](https://your-streamlit-demo-link.streamlit.app/)**

> **Note:** Replace the link above with your actual Streamlit deployment link after deploying the application.

## Project Overview

This project uses **PyTorch** to build and train a CNN for binary image classification.

The CNN consists of:

* Two convolutional layers
* ReLU activation functions
* Max pooling layers
* A fully connected layer
* Dropout regularization
* An output layer with two classes: Cat and Dog

Dropout with a probability of `0.5` was added to help reduce overfitting.

## Model Architecture

```text
Input Image (3 × 64 × 64)
        ↓
Convolutional Layer (3 → 16)
        ↓
ReLU
        ↓
Max Pooling
        ↓
Convolutional Layer (16 → 32)
        ↓
ReLU
        ↓
Max Pooling
        ↓
Flatten
        ↓
Fully Connected Layer (8192 → 64)
        ↓
ReLU
        ↓
Dropout (0.5)
        ↓
Fully Connected Layer (64 → 2)
        ↓
Cat / Dog
```

## Data Preprocessing

Images are resized to:

```text
64 × 64 pixels
```

Training images use data augmentation:

* Random horizontal flip
* Random rotation up to 10 degrees
* Random brightness adjustment

Both training and test images are converted to tensors and normalized using:

```text
Mean = (0.5, 0.5, 0.5)
Standard Deviation = (0.5, 0.5, 0.5)
```

## Dataset

The dataset is divided into:

* **80% training data**
* **20% testing data**

A fixed random seed of `42` is used for reproducibility.

## Model Training

The model is trained using:

* **Loss Function:** Cross Entropy Loss
* **Optimizer:** Adam
* **Learning Rate:** `0.001`
* **Number of Epochs:** `15`

The final training accuracy was:

```text
82.06%
```

The test accuracy was:

```text
80.64%
```

## Streamlit Application

The Streamlit application allows users to:

1. Upload a `.jpg`, `.jpeg`, or `.png` image.
2. View the uploaded image.
3. Process the image using the trained CNN.
4. Display whether the image is predicted to be a Cat or Dog.
5. Display the model's confidence percentage.

## Project Structure

```text
cat-dog-cnn-classifier/
│
├── app.py
├── model.py
├── model.pkl
├── cnn(1).ipynb
├── app.ipynb
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/cat-dog-cnn-classifier.git
```

Move into the project directory:

```bash
cd cat-dog-cnn-classifier
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app.py
```

## Technologies Used

* Python
* PyTorch
* Torchvision
* Streamlit
* Pillow
* Jupyter Notebook

## Author

**Odion Promise**

Computer Science
