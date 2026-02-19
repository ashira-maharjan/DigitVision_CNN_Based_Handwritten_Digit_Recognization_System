# DigitVision_CNN_Based_Handwritten_Digit_Recognization_System

> DigitVision is an end-to-end Deep Learning project that accurately recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with PyTorch and deployed as an interactive web application using Streamlit.

This project demonstrates the complete machine learning lifecycle from data preprocessing and model training to evaluation, visualization, deployment, and user interaction handling.

![License](https://img.shields.io/badge/license-MIT-green) ![Version](https://img.shields.io/badge/version-1.0.0-blue) ![Language](https://img.shields.io/badge/language-Python-yellow) 
##  Project Information

- **👤 Author:** ashira-maharjan
- **📦 Version:** 1.0.0
- **📄 License:** MIT
- **🌐 Website:** [https://digitvisionrecognization.streamlit.app/](https://digitvisionrecognization.streamlit.app/)
- **📂 Repository:** [https://github.com/ashira-maharjan/DigitVision_CNN_Based_Handwritten_Digit_Recognization_System](https://github.com/ashira-maharjan/DigitVision_CNN_Based_Handwritten_Digit_Recognization_System)

## Project Structure 
```markdown 
DigitVision_CNN_Based_Handwritten_Digit_Recognization_System/
│
├── model/
│   ├── mnist_cnn.pth
│
├── data/MNIST/raw
│
├── src
|   |── evaluate.py
|   |── model.py
|   |── train.py
|
├── notebook 
|
├── uploads
│
├── app.py
├── app1.py
├── requirements.txt
└── README.md
```

## Model Architecture
- Convolutional Layers
- ReLU Activation 
- MaxPooling

The model is trained on handwritten digit dataset (like MNIST format).

## Streamlit Web App Features

The deployed web application allows users to:

-  Draw a digit using canvas
- Upload a digit image
- Get instant prediction
- View prediction confidence
-  Automatically save:

## Installation 
```python 
git clone https://github.com/ashira-maharjan/DigitVision_CNN_Based_Handwritten_Digit_Recognization_System.git
cd DigitVision_CNN_Based_Handwritten_Digit_Recognization_System
```

Instal dependencies 
```python  
pip install -r requirements.txt
```

Run Application 
```python 
streamlit run app.py
```

## Learning Outcomes

This project demonstrates:

- End-to-End Deep Learning Workflow

- CNN Implementation from Scratch

- Model Evaluation Techniques

- Data Visualization

- Model Deployment with Streamlit

- Handling User Inputs & Saving Data