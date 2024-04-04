import numpy as np
import spacy
from datasets import load_dataset
from collections import Counter
from multiprocessing import Pool
from math import log


# Constants
NLP_MODEL = "en_core_web_sm"
DATASET_NAME = "wikitext"
DATASET_VERSION = "wikitext-2-raw-v1"
DATASET_SPLIT = "train"



# Initialize SpaCy model
# The 'disable' parameter is used to skip loading the Named Entity Recognition (ner)
# and Dependency Parsing (parser) components of the model.
# This improves loading and processing speed when these features are not needed.
language_model = spacy.load(NLP_MODEL, disable=["ner", "parser"])


class UnigramModel:
    def __init__(self):
        self.probabilities = {}  # dictionary of lemmas and their log probabilities

    def train(self, lemmas_list):
        """
        Train the unigram model on the provided lemmas list.
        :param lemmas_list: a list of lists, where each inner list contains lemmas of a line in a text
        """
        lemmas_count = Counter()
        for lemma_line in lemmas_list:
            lemmas_count.update(lemma_line)

        total_lemmas = sum(lemmas_count.values())
        self.probabilities = {lemma: log(count / total_lemmas) for lemma, count in lemmas_count.items()}

    def get_unigram_log_probability(self, lemma):
        """
        Get the log probability of a lemma.
        :param lemma: the lemma to get the probability for
        :return: log probability of the lemma
        """
        return self.probabilities.get(lemma, -np.inf)  # if the lemma is not in the dictionary, meaning it was not
        # seen in the training, return -inf


class BigramModel:
    def __init__(self):
        self.probabilities = {}
        self.start_token = "<START>"

    def train(self, lemmas_lines):
        """
        Train the bigram model on the provided lemmas lines.
        :param lemmas_lines: a list of lists, where each inner list contains lemmas of a line
        """
        bigram_counts = Counter()
        lemmas_counts = Counter()
        for line in lemmas_lines:
            bigrams = list(zip([self.start_token] + line, line))
            bigram_counts.update(bigrams)
            lemmas_counts.update([self.start_token] + line[:-1])  # Remove the last lemma because it doesn't have a
            # next lemma and for the MLE we need to count the number of times each lemma appears as the first lemma of
            # a bigram.

        self.probabilities = {bigram: log(count / lemmas_counts[bigram[0]]) for bigram, count in bigram_counts.items()}

    def predict_next_word(self, sentence, process_line_func):
        """
        Predict the next word in a sentence using the bigram model.
        :param sentence: a string representing a sentence
        :param process_line_func: a function to process the sentence into lemmas
        :return: the most probable next word in the sentence according to the bigram model
        """
        lemmas = process_line_func(sentence)
        last_lemma = lemmas[-1] if lemmas else self.start_token
        candidate_bigrams = {bigram: prob for bigram, prob in self.probabilities.items() if bigram[0] == last_lemma}

        if not candidate_bigrams:
            return "No prediction available"  # word didn't appear in train set as a first word of a bigram

        most_probable_bigram = max(candidate_bigrams, key=candidate_bigrams.get)  # (last_lemma, the predicted lemma)
        return most_probable_bigram[1]

    def compute_sentence_log_probability(self, sentence, process_line_func):
        """
        Compute the log probability of a sentence using the bigram model.
        :param sentence: a string representing a sentence
        :param process_line_func: a function to process the sentence into lemmas
        :return: the log probability of the sentence
        """
        lemmas = process_line_func(sentence)
        bigrams = list(zip([self.start_token] + lemmas, lemmas))
        return sum(self.get_bigram_log_probability(bigram) for bigram in bigrams)

    def compute_perplexity(self, sentences, process_line_func):
        """
        Compute perplexity for a set of sentences.
        :param sentences: a list of strings representing sentences
        :param process_line_func: a function to process the lines into lemmas
        :return: the perplexity of the sentences
        """
        # Calculate the total number of lemmas in the test data and add the number of start tokens to each sentence to
        # the lemmas count
        total_text = ".".join(sentences)
        num_of_lemmas = len(process_line_func(total_text))

        # Calculate the log prob to get each sentence in the test data and divide by the number of lemmas
        total_log_prob = sum(self.compute_sentence_log_probability(sentence, process_line_func) for sentence in
                             sentences)
        return np.exp(-total_log_prob / num_of_lemmas)

    def get_bigram_log_probability(self, bigram):
        """
        Get the log probability of a lemma.
        :param bigram: the (lemma, lemma) bigram to get the probability for
        :return: log probability of the lemma
        """
        return self.probabilities.get(bigram, -np.inf)  # if the bigram is not in the dictionary, meaning it was not
        # seen in the training, return -inf


class InterpolatedModel:
    def __init__(self, unigram_model, bigram_model, lambda_bigram=2/3, lambda_unigram=1/3):
        self.unigram_model = unigram_model
        self.bigram_model = bigram_model
        self.lambda_bigram = lambda_bigram
        self.lambda_unigram = lambda_unigram

    def compute_interpolated_log_probability(self, bigram):
        """
        Compute the interpolated probability of a bigram using linear interpolation.
        :param bigram: the bigram tuple
        :return: interpolated probability of the bigram
        """
        bigram_prob = self.bigram_model.get_bigram_log_probability(bigram)
        unigram_prob = self.unigram_model.get_unigram_log_probability(bigram[1])

        # Handling cases where either or both probabilities are -inf (log(0))
        bigram_prob_exp = np.exp(bigram_prob) if bigram_prob > -np.inf else 0
        unigram_prob_exp = np.exp(unigram_prob) if unigram_prob > -np.inf else 0

        interpolated_prob = (self.lambda_bigram * bigram_prob_exp) + (self.lambda_unigram * unigram_prob_exp)
        return np.log(interpolated_prob) if interpolated_prob != 0 else -np.inf

    def compute_sentence_log_probability(self, sentence, process_line_func):
        """
        Compute the log probability of a sentence using the interpolated model.
        :param sentence: a string representing a sentence
        :param process_line_func: a function to process the sentence into lemmas
        :return: the log probability of the sentence
        """
        lemmas = process_line_func(sentence)
        bigrams = list(zip([self.bigram_model.start_token] + lemmas, lemmas))

        return sum(self.compute_interpolated_log_probability(bigram) for bigram in bigrams)

    def compute_perplexity(self, sentences, process_line_func):
        """
        Compute perplexity for a set of sentences.
        :param sentences: a list of strings representing sentences
        :param process_line_func: a function to process the lines into lemmas
        :return: the perplexity of the sentences
        """
        # Calculate the total number of lemmas in the test data and add the number of start tokens to each sentence to
        # the lemmas count
        total_text = ".".join(sentences)
        num_of_lemmas = len(process_line_func(total_text))

        # Calculate the log prob to get each sentence in the test data and divide by the number of lemmas
        total_log_prob = sum(self.compute_sentence_log_probability(sentence, process_line_func) for sentence in
                             sentences)
        return np.exp(-total_log_prob / num_of_lemmas)


def process_line_to_filtered_lemmas(line):
    """
    Process a line using SpaCy, filters out non-alpha tokens, and extracts lemmas.
    :param line: a string line from the dataset
    :return: a list of lemmas extracted from the line
    """
    doc = language_model(line)
    return [token.lemma_ for token in doc if token.is_alpha]


def convert_data_to_filtered_lemmas(dataset):
    """
    Process the dataset to extract lemmas from each line using SpaCy and parallel processing.
    :param dataset: a list of text lines from the dataset
    :return: a list of lists, where each inner list contains the lemmas of a line from the dataset
    """

    filtered_dataset = [line for line in dataset if line.strip()]

    # process each line of the dataset in parallel
    with Pool() as pool:
        lemmas_list = pool.map(process_line_to_filtered_lemmas, filtered_dataset)
    # filter out empty lines
    return [line for line in lemmas_list if len(line) > 0]


def main():

    # 1 train unigram and bigram models over dataset
    print("Question 1: training a unigram and bigram models over dataset")
    # Load dataset
    train_dataset = load_dataset(DATASET_NAME, DATASET_VERSION, split=DATASET_SPLIT)
    lemmas_list = convert_data_to_filtered_lemmas(train_dataset['text'])

    # Initialize and train models
    unigram_model = UnigramModel()
    unigram_model.train(lemmas_list)

    bigram_model = BigramModel()
    bigram_model.train(lemmas_list)

    interpolated_model = InterpolatedModel(unigram_model, bigram_model)

    # 2 predict next word of a sentence
    print("\nQuestion 2: predict next word of a sentence")
    sentence_to_predict_next_word_for = "I have a house in"
    next_word = bigram_model.predict_next_word(sentence_to_predict_next_word_for, process_line_to_filtered_lemmas)
    print(f"Next word prediction of the sentence \"{sentence_to_predict_next_word_for}\": {next_word}")

    # 3 compute sentence log probability and perplexity for each sentence in the array
    print("\nQuestion 3: compute sentence log probability and perplexity to evaluate the bigram model")
    sentences = ["Brad Pitt was born in Oklahoma", "The actor was born in USA"]
    for sentence in sentences:
        log_prob = bigram_model.compute_sentence_log_probability(sentence, process_line_to_filtered_lemmas)
        print(f"Bigram log probability of the sentence \"{sentence}\" is: {log_prob}")

    perplexity = bigram_model.compute_perplexity(sentences, process_line_to_filtered_lemmas)
    print(f"Bigram model perplexity over test set with sentences {sentences} is: {perplexity}")

    # 4. Interpolated model - compute interpolated bigram probabilities using linear interpolation
    print("\nQuestion 4: Interpolated model - compute interpolated bigram probabilities using linear interpolation")
    for sentence in sentences:
        interpolated_log_prob = interpolated_model.compute_sentence_log_probability(sentence,
                                                                                    process_line_to_filtered_lemmas)
        print(f"Interpolated log probability of the sentence \"{sentence}\" is: {interpolated_log_prob}")

    interpolated_perplexity = interpolated_model.compute_perplexity(sentences,process_line_to_filtered_lemmas)
    print(f"Interpolated model perplexity over test set with sentences {sentences} is: {interpolated_perplexity}")


if __name__ == '__main__':
    main()
