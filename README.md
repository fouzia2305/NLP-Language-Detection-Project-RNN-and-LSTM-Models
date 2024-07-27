# NLP Language Detection Using RNN and LSTM

## Overview
This project aims to develop a robust language detection model to classify text into 17 different languages. The model is built using advanced neural network techniques, specifically Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks, leveraging Python and popular libraries like TensorFlow, Keras, and Scikit-learn.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Development](#model-development)
- [Results](#results)

## Technologies Used
- Python
- TensorFlow
- Keras
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

## Dataset
The multilingual text dataset includes samples in 17 different languages. The dataset was preprocessed and feature extraction was performed using `CountVectorizer`.

## Model Development
- **Data Preparation**: Preprocessed and cleaned multilingual text datasets using `CountVectorizer` for feature extraction.
- **Model Construction**:
  - Developed and fine-tuned a language detection model using both RNN and LSTM networks.
  - Implemented `EarlyStopping` to prevent overfitting.
- **Training & Optimization**: Achieved a test accuracy of 96.5% with the RNN model and 97.4% with the LSTM model.
- **Evaluation**: Generated classification reports and confusion matrices to assess model performance.

## Results
The LSTM model demonstrated superior performance compared to the RNN model, achieving higher accuracy and better generalization capabilities. This superior performance is attributed to the LSTM's ability to capture long-term dependencies in the data more effectively than standard RNNs, making it better suited for language detection tasks where context and sequential information are crucial.

Detailed visualizations of performance metrics and confusion matrices are included in the results.
