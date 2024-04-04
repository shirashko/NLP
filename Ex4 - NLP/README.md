# Text Classification with Transformer Models

This repository contains implementations and comparisons of three different models for the task of text classification on a subset of the 20newsgroups dataset. The models include traditional logistic regression with TFIDF encoding, fine-tuned Transformer models, and zero-shot classification using a pre-trained transformer.

## Overview

The goal of this exercise is to understand the application and effectiveness of simple Transformer architectures in the text classification domain and compare them to a baseline log-linear model.

## Models

1. **Log-linear Classifier**: A traditional machine learning approach using Logistic Regression with Term Frequency-Inverse Document Frequency (TFIDF) vector encoding.
2. **Fine-tuned Transformer**: Utilizes a fine-tuned `distilroberta-base` model for text classification.
3. **Zero-shot Classification**: Employs the `cross-encoder/nli-MiniLM2-L6-H768` model for classification without fine-tuning on the task-specific data.

## Requirements

- Python programming language with packages:
  - `transformers`
  - `scikit-learn`
- The 20newsgroups dataset.

## Installation

Install the necessary Python packages:

```bash
pip install transformers scikit-learn
```

For `scikit-learn`, you can use the pre-installed version on the aquarium computers if you are working in that environment.

## Data

We focus on four topics from the 20newsgroups dataset: `'comp.graphics'`, `'rec.sport.baseball'`, `'sci.electronics'`, and `'talk.politics.guns'`.

## Classification Tasks

1. Run a Log-linear classifier and plot accuracy as a function of data portion used: full dataset, half, and 10%.
2. Fine-tune the Transformer model and report average loss and validation accuracy across epochs for different data portions.
3. Perform zero-shot classification using the pretrained pipeline and report accuracy.
4. Comparative analysis of the models with insights on their performance, sensitivity to training set size, pros, and cons.

## Usage

To run the classification tasks and models:

```bash
python ex4.py
```

Make sure to adjust the script to load and preprocess the data correctly and that you follow the specific instructions within each section of the provided `ex5.py` skeleton file.

## Results

Include the following in your PDF file:

- Plots and accuracy metrics for all models.
- A detailed comparison of model performance.
- Discussion on the models' performance, including pros and cons, particularly for zero-shot classification.

## Repository Contents

- `ex4.py`: Source code for the classification models.
- `README.md`: This file with instructions and details about the exercise.
- `results.pdf`: The PDF file with detailed results and answers.
