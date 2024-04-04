# Natural Language Processing - Exercise 1

The first exercise for the Natural Language Processing course. In this exercise, we delve into the construction and application of unigram and bigram language models. We explore the theoretical underpinnings of language models and put theory into practice by implementing models in Python.

## Theoretical Overview

The exercise begins with a theoretical exploration of bigram language models, non-Markovian models, the construction of a spelling corrector, and advanced smoothing methods. We analyze the probability distributions of word sequences and the impact of various smoothing techniques on these distributions.

### Key Topics

- Bigram language models and their properties
- Differences between Markovian and non-Markovian models
- Building a spelling corrector focused on contextual word use
- Good-Turing smoothing method and its effect on unseen words

## Practical Implementation

The practical part of the exercise involves hands-on tasks such as:

1. **Training Language Models**: Implementing unigram and bigram models using a maximum likelihood estimator based on the Wikitext-2 dataset.
2. **Sentence Prediction**: Extending a given sentence using the most probable next word according to the bigram model.
3. **Probability and Perplexity Calculation**: Evaluating the likelihood and complexity of given sentences using the trained bigram model.
4. **Linear Interpolation Smoothing**: Estimating a new model by smoothing between the bigram and unigram models and re-evaluating the sentences.

### Prerequisites

- Python programming language
- SpaCy package for language processing
- Huggingface’s `datasets` package

### Installation

Before running the scripts, ensure you have the necessary Python environment and packages installed:

```bash
pip install spacy
python -m spacy download en_core_web_sm
pip install datasets
```

## Running the Scripts

To execute the language model scripts:

```bash
python nlp_language_models.py
```

Make sure you have the `wikitext-2-raw-v1` dataset downloaded and accessible by the script.

## Submission

Alongside your `.pdf` file with the theoretical answers, include a `.py` or `.ipynb` file with your code implementations.

---

Replace the `python nlp_language_models.py` with the actual script name you used for the language models implementation. Ensure you provide additional instructions or comments within your code to make the functionality clear for those who may review or assess your work.
