# Natural Language Processing: Exercise 2

This repository contains the second set of exercises for the Natural Language Processing course, focusing on hidden Markov models (HMMs), Viterbi algorithms, and Parts-of-Speech (POS) tagging using the Brown Corpus.

## Overview

The exercises cover theoretical and practical applications of HMMs in NLP. It includes constructing language models to determine the sequence of states in biological data, extending and modifying the Viterbi algorithm for n-gram taggers, and implementing an HMM POS tagger with various smoothing techniques.

## Theoretical Component

The theoretical part addresses complex questions about the behavior of language models and the Viterbi algorithm in scenarios such as:
- Biological sequence analysis
- Modifications of n-gram models and predictions
- Smoothing methods and their impact on language model probabilities

## Practical Component

In the practical part, we implement HMM POS taggers with tasks such as:
- **Baseline Model**: Establishing a baseline using the most likely tag for each word.
- **Bigram HMM Tagger**: Constructing and applying a bigram HMM for POS tagging.
- **Add-One Smoothing**: Implementing smoothing techniques to enhance the model performance.
- **Pseudo-Words Technique**: Utilizing pseudo-words for unknown or low-frequency words in the training set to improve the handling of unknown words.
- **Performance Analysis**: Evaluating error rates and investigating errors using confusion matrices.

### Dataset

The dataset used is the “news” category from the Brown corpus, available via the NLTK toolkit.

### Environment Setup

Ensure that you have the following packages installed:

- NLTK: For loading and processing the Brown corpus.
- SpaCy (optional): For additional language processing tasks if required.

### Installation

Install the required packages using the following commands:

```bash
pip install nltk
nltk.download('brown')
```

## Usage

To run the exercises, navigate to the source code directory and execute the Python scripts:

```bash
python pos_tagger.py
```

## Submission Guidelines

Please include:
- A `.pdf` file with your answers to the theoretical questions.
- A `.py` or `.ipynb` file with your implementation of the practical tasks.
- A `README.txt` file with instructions on running your code and any other necessary documentation.
