import re
from collections import defaultdict, Counter
import nltk
import numpy as np
from matplotlib import pyplot as plt

CORPUS = "brown"  # Corpus to use in the exercise
CATEGORY = "news"  # Category in the brown corpus to use in the exercise


def load_tagged_corpus():
    """
    Loads tagged sentences from the Brown corpus based on the specified category.
    :return: List of tagged sentences.
    """
    nltk.download(CORPUS)
    return nltk.corpus.brown.tagged_sents(categories=CATEGORY)


def split_train_test(processed_sentences, split_ratio=0.9):
    """
    Splits the processed sentences into training and test sets based on the provided split ratio.
    :param processed_sentences - List of processed sentences.
    :param split_ratio - Ratio at which to split the sentences. Default is 0.9 (90% training, 10% test).
    :return: Tuple containing train_sentences and test_sentences.
    """
    split_index = int(len(processed_sentences) * split_ratio)
    train_sentences = processed_sentences[:split_index]
    test_sentences = processed_sentences[split_index:]
    return train_sentences, test_sentences


def filter_tag(word):
    """
    Filter the tag to keep only alphabetical characters and '-' characters.
    Also remove the prefix of the word if it contains a '+' or a '-' and make no changes if the word contains no
    alphabetical characters
    :param word: The tag to filter
    :return: The filtered tag
    """
    # Keep special case for '--' which is a valid tag
    if word == "--":
        return word

    # Get the prefix of the word if it contains a '+' or a '-'
    if '+' in word or '-' in word:
        # get the prefix of the word
        word = word[:word.find('+')] if '+' in word else word[
                                                         :word.find('-')]

    # Return the input string if it contains no alphabetical characters
    if not any(char.isalpha() for char in word):
        return word

    # Use regular expression to keep only alphabetical characters. This is done for not distinguish between AT, AT$.
    result = re.sub(r'[^a-zA-Z]', '', word)  # Keep only alphabetical characters
    return result


def compute_error_rate(errors, total):
    """
    Computes the error rate.
    :param errors: Number of errors.
    :param total: Total number of instances.
    :return: Error rate.
    """
    return errors / total if total > 0 else 0  # Avoid division by zero


def process_complex_tags(tagged_sentences):
    """
    Processes complex tags in the tagged sentences, replacing them with simplified tags.
    :param tagged_sentences: List of tagged sentences.
    :return: List of processed sentences with simplified tags.
    """
    processed_sentences = []
    for sentence in tagged_sentences:
        # filtering the tag of each word in the sentence
        processed_sent = [(word, filter_tag(tag)) for word, tag in sentence]
        processed_sentences.append(processed_sent)
    return processed_sentences


def display_evaluation(model_name, known_word_error_rate, total_error_rate, unknown_word_error_rate):
    """
    Displays the evaluation results of a POS tagger model.
    :param model_name: Name of the model.
    :param known_word_error_rate: Error rate for known words.
    :param total_error_rate: Total error rate.
    :param unknown_word_error_rate: Error rate for unknown words.
    """
    print(f"################### {model_name} POS Tagger Results ###################")
    print(f"Error rate for known words: {known_word_error_rate:.5f}")
    print(f"Error rate for unknown words: {unknown_word_error_rate:.5f}")
    print(f"Total error rate: {total_error_rate:.5f}")


class MLEMostLikelyTagModel:
    """
    A baseline model that assigns the most likely tag to each word in the training set using Maximum Likelihood
    """

    def __init__(self):
        """
        Initializes the MostLikelyTagModel which computes and stores the most likely tag for each word in the training
        set.
        """
        self.words_to_most_likely_tags = {}  # A dictionary mapping words to their most likely tag
        self.most_likely_tag_for_unknown_words = "NN"

    def train(self, train_sentences):
        """
        Trains the model on the provided training sentences. Computes the most likely tag for each word.
        :param train_sentences: List of processed training sentences with tags.
        """
        word_tag_frequency = defaultdict(Counter)  # A dictionary mapping words to their tag frequencies
        for sentence in train_sentences:
            for word, tag in sentence:
                word_tag_frequency[word][tag] += 1

        for word, tag_frequency in word_tag_frequency.items():  # For each word, get the most frequent tag
            self.words_to_most_likely_tags[word] = tag_frequency.most_common(1)[0][0]

    def evaluate(self, test_sentences):
        """
        Evaluates the model on the test sentences. Computes the error rates for known words, unknown words,
        and the total error rate.
        :param test_sentences: List of processed test sentences with tags.
        :return: Tuple: Known word error rate, unknown word error rate, total error rate.
        """
        total_errors, known_words_errors, unknown_words_errors, total_known_words, total_unknown_words = (
            self._evaluate_sentences(test_sentences))
        # computing the error rates for known words, unknown words, and the total error rate
        known_word_error_rate = compute_error_rate(known_words_errors, total_known_words)
        unknown_word_error_rate = compute_error_rate(unknown_words_errors, total_unknown_words)
        total_error_rate = compute_error_rate(total_errors, total_known_words + total_unknown_words)

        return known_word_error_rate, unknown_word_error_rate, total_error_rate

    def _evaluate_sentences(self, test_tagged_corpus):
        """
        Evaluates the model on the test sentences. Computes the error rates for known words, unknown words,
        and the total error rate.
        :param test_tagged_corpus: List of processed test sentences with tags.
        :return: Tuple: Total errors, known word errors, unknown word errors, total known words, total unknown words.
        """
        total_errors = known_word_errors = unknown_word_errors = total_known_words = total_unknown_words = 0

        for test_sentence in test_tagged_corpus:
            for word, actual_tag in test_sentence:
                is_prediction_correct, is_known = self._evaluate_word(word, actual_tag)
                # updating the counters
                total_errors += not is_prediction_correct
                known_word_errors += not is_prediction_correct and is_known
                unknown_word_errors += not is_prediction_correct and not is_known
                total_known_words += is_known
                total_unknown_words += not is_known

        return total_errors, known_word_errors, unknown_word_errors, total_known_words, total_unknown_words

    def _evaluate_word(self, word, actual_tag):
        """
        Evaluates a single word's predicted tag against its actual tag.
        :param word: The word to evaluate.
        :param actual_tag: The actual tag of the word.
        :return: Tuple (is_prediction_correct, is_known):
        """
        predicted_tag = self.words_to_most_likely_tags.get(word, self.most_likely_tag_for_unknown_words)
        is_prediction_correct = (predicted_tag == actual_tag)
        is_known = word in self.words_to_most_likely_tags  # checking if the word is known or not
        return is_prediction_correct, is_known

    def predict_sentence_pos_tags(self, sentence):
        """
        Predicts the POS tags for a sentence.
        :param sentence: A list of words in a sentence.
        :return: A list of tuples, each containing a word and its predicted POS tag.
        """
        predicted_tags = []
        for word in sentence:
            predicted_tag = self.words_to_most_likely_tags.get(word, self.most_likely_tag_for_unknown_words)
            predicted_tags.append((word, predicted_tag))

        return predicted_tags


class BigramHMMTagger:
    """
    A Hidden Markov Model (HMM) that uses bigram transition and emission probabilities to predict the most likely
    sequence of tags for a given sequence of words. estimates the emission probabilities and transition probabilities
    using Maximum Likelihood Estimation (MLE).
    """

    def __init__(self, vocabulary):
        """
        Initializes the BigramHMMTagger which computes and stores the transition and emission probabilities
        for a bigram Hidden Markov Model based on the training data.
        """
        # Model parameters
        self.transition_probs = defaultdict(Counter)
        self.emission_probs = defaultdict(Counter)

        self.vocabulary = vocabulary

        # Additional data structures to store known words and tags and their counts
        self.known_words = set()
        self.known_tags = set()
        self.emission_counts = defaultdict(Counter)

        self.unknown_word_tag = "NN"  # When seeing unknown words we will assign them the tag "NN"
        self.start_tag = "<START>"
        self.stop_tag = "<STOP>"

        self.is_smoothed = False
        self.use_pseudo_words = False

        # rules for categorizing words based on specific characteristics or patterns they exhibit

    def train(self, train_tagged_corpus):
        """
        Trains the model on the provided training sentences. Computes the bigram transition and emission probabilities.
        :param train_tagged_corpus: List of processed training sentences with tags.
        """
        # initializing counters for tag pairs and tag-word pairs
        transition_counts = defaultdict(Counter)
        emission_counts = defaultdict(Counter)

        # counting the tag pairs and tag-word pairs to compute the transition and emission probabilities
        for sentence in train_tagged_corpus:
            prev_tag = self.start_tag
            for word, tag in sentence:
                transition_counts[prev_tag][tag] += 1
                emission_counts[tag][word] += 1
                self.known_words.add(word)
                self.known_tags.add(tag)
                prev_tag = tag

            # At the end of each sentence, we need to count the transition from the last tag to the stop tag
            transition_counts[prev_tag][self.stop_tag] += 1

        self.emission_counts = emission_counts  # storing the emission counts for later use for smoothing
        # computing the transition and emission probabilities using MLE
        self.transition_probs = self._compute_mle_transition_probs(transition_counts)
        self.emission_probs = self._compute_mle_emission_probs()

    def _compute_mle_emission_probs(self):
        """
        Computes emission probabilities using Maximum Likelihood Estimation (MLE).
        :param emission_counts: Counts of how often each word is emitted by each tag.
        :return: A dictionary mapping tags to their emission probabilities.
        """
        emission_probs = {}
        for tag, word_frequencies in self.emission_counts.items():
            total_emissions = sum(self.emission_counts[tag].values())  # total number of word emissions for the tag
            emission_probs[tag] = {word: frequency / total_emissions for word, frequency in word_frequencies.items()}
        return emission_probs

    @staticmethod
    def _compute_mle_transition_probs(transition_counts):
        """
        Computes transition probabilities using Maximum Likelihood Estimation (MLE).
        :param transition_counts: Counts of tag transitions.
        :return: A dictionary mapping tags to their transition probabilities.
        """
        tags_transitions_probs = {}
        for current_tag, following_tags_frequencies in transition_counts.items():
            total_transitions = sum(following_tags_frequencies.values())
            tags_transitions_probs[current_tag] = {
                next_tag: transition_count / total_transitions
                for next_tag, transition_count in following_tags_frequencies.items()
            }
        return tags_transitions_probs

    @staticmethod
    def _compute_predict_tags(states, viterbi_table):
        """
        Computes the most probable sequence of tags and its probability using the Viterbi table.
        This method backtracks from the last state in the Viterbi table to construct the most likely sequence of tags
        for the given sentence based on the calculated probabilities.
        :param states: A list of possible states (tags) in the HMM model.
        :param viterbi_table: The Viterbi table containing probabilities and back pointers.
        :return: A tuple containing the maximum probability of the most likely tag sequence (max_prob) and the sequence.
        """
        max_prob, last_tag = max((viterbi_table[-1][state]["prob"], state) for state in states)
        predicted_tags = [last_tag]
        previous = viterbi_table[-1][last_tag]["prev"]
        for t in range(len(viterbi_table) - 2, -1, -1):
            predicted_tags.insert(0, previous)  # prepend for efficiency
            previous = viterbi_table[t][previous]["prev"]
        return max_prob, predicted_tags

    def add_one_laplace_emission_probability_smoothing(self):
        """
        Adds one Laplace smoothing to the emission probabilities.
        """
        smoothed_emission_probs = {}
        for tag, word_frequencies in self.emission_counts.items():
            # The total emissions for a tag are increased by the vocabulary size for smoothing
            # How many times did this tag emit words + |v|*1
            total_emissions = sum(self.emission_counts[tag].values()) + len(self.vocabulary)

            smoothed_emission_probs[tag] = {}
            for word in self.vocabulary:
                # Apply add-one smoothing. Use get method to handle words that did not appear with this tag
                frequency = word_frequencies.get(word, 0)
                smoothed_emission_probs[tag][word] = (frequency + 1) / total_emissions

        self.emission_probs = smoothed_emission_probs
        self.is_smoothed = True

    def cancel_emission_probability_smoothing(self):
        """
        Cancels the emission probability smoothing.
        """
        self.emission_probs = self._compute_mle_emission_probs()
        self.is_smoothed = False

    def viterbi_algorithm(self, sentence):
        """
        Applies the Viterbi algorithm to predict the most likely sequence of tags for a given sequence of words.
        :param sentence: The sequence of words.
        :return: List of str: The most likely sequence of tags.
        """
        states = list(self.known_tags)  # list of possible states (tags) in the HMM model
        viterbi_table = [{}]  # Initializing the viterbi table

        # Consider the transition from start tag of to the first tag to evaluate the likelihood of the tag to appear
        # in the start of a sentence. Meaning the probability of sentence in length 0 to get to some state from start
        # tag state
        for state in states:
            emission_prob = self.get_emission_prob(sentence[0], state)
            prob_tag_to_appear_in_start = self.transition_probs[self.start_tag].get(state, 0)
            viterbi_table[0][state] = {"prob": prob_tag_to_appear_in_start * emission_prob, "prev": self.start_tag}

        for t in range(1, len(sentence)):
            viterbi_table.append({})
            for state in states:
                # Get the path with the highest probability, and the maximize previous state
                max_tag_transition_prob, prev_state_selected = max((viterbi_table[t - 1][prev_state]["prob"] *
                                                                    self.transition_probs[prev_state].get(state, 0),
                                                                    prev_state) for prev_state in states)

                emission_prob = self.get_emission_prob(sentence[t], state)

                # If the previous word was unknown, and the path to prev state=self.unknown_word_tag was zero, then
                # prev_state_selected might not be the path that it's previous state is self.unknown_word_tag.
                viterbi_prob = max_tag_transition_prob * emission_prob
                is_known_word = sentence[t - 1] in self.known_words
                viterbi_prev = prev_state_selected if (is_known_word or self.is_smoothed) else self.unknown_word_tag
                viterbi_table[t][state] = {"prob": viterbi_prob, "prev": viterbi_prev}

        # Consider the transition from last tag of to the stop tag to evaluate the likelihood of a tag to appear at the
        # end of a sentence
        for state in states:
            viterbi_table[-1][state]["prob"] *= self.transition_probs[state].get(self.stop_tag, 0)

        # Construct the final list of tags
        max_prob, predicted_tags = self._compute_predict_tags(states, viterbi_table)

        return predicted_tags, max_prob

    def get_emission_prob(self, word, state):
        """
        Computes the emission probability of a word given a state (tag).
        :param word: the word to compute the emission probability for.
        :param state: the state (tag) to compute the emission probability for.
        :return:
        """
        if not (word in self.known_words) and not self.is_smoothed:
            return 0 if state != self.unknown_word_tag else 1

        return self.emission_probs[state].get(word, 0)

    def evaluate(self, test_sentences):
        """
        Evaluates the model on the test sentences. Computes the error rates for known words, unknown words,
        and the total error rate.
        :param test_sentences: List of processed test sentences with tags.
        :return: Tuple: Known word error rate, unknown word error rate, total error rate.
        """
        original_test_test = test_sentences
        if self.use_pseudo_words:
            test_sentences = self.apply_pseudo_words_to_test_sentences(test_sentences)

        # getting the number of errors for known words, unknown words, and the total number of errors, also the total
        # number of known words and unknown words
        total_errors = known_word_errors = unknown_word_errors = total_known_words = total_unknown_words = 0

        for original_sentence, sentence in zip(original_test_test, test_sentences):
            words, actual_tags = zip(*sentence)  # getting the words and tags of the sentence
            predicted_tags, _ = self.viterbi_algorithm(words)  # getting the predicted tags of the sentence

            original_words, _ = zip(*original_sentence)

            for original_word, word, actual_tag, predicted_tag in zip(original_words, words, actual_tags,
                                                                      predicted_tags):
                if self.use_pseudo_words:
                    # Because we only added pseudo words to the original words, and not removed the original words,
                    # we need to evaluate the original word, not the pseudo word.
                    is_prediction_correct, is_known = self.evaluate_word(original_word, actual_tag, predicted_tag)
                else:
                    is_prediction_correct, is_known = self.evaluate_word(word, actual_tag, predicted_tag)

                # updating the counters
                total_errors += not is_prediction_correct
                known_word_errors += not is_prediction_correct and is_known
                unknown_word_errors += not is_prediction_correct and not is_known
                total_known_words += is_known
                total_unknown_words += not is_known

        known_word_error_rate = compute_error_rate(known_word_errors, total_known_words)
        unknown_word_error_rate = compute_error_rate(unknown_word_errors, total_unknown_words)
        total_error_rate = compute_error_rate(total_errors, total_known_words + total_unknown_words)

        return known_word_error_rate, unknown_word_error_rate, total_error_rate

    def apply_pseudo_words_to_test_sentences(self, test_sentences):
        """
        Applies pseudo words transformation to each word in the test sentences if use_pseudo_words is True.

        Args:
            test_sentences (List[List[Tuple[str, str]]]): The test sentences with their original tags.

        Returns:
            List[List[Tuple[str, str]]]: The modified test sentences with pseudo words applied.
        """
        modified_test_sentences = []
        for sentence in test_sentences:
            modified_sentence = [(self.get_word_when_pseudo_words_is_on(word), tag) for word, tag in sentence]
            modified_test_sentences.append(modified_sentence)

        return modified_test_sentences

    def evaluate_word(self, word, actual_tag, predicted_tag):
        """
        Evaluates a single word's predicted tag against its actual tag.
        :param word: The word to evaluate.
        :param actual_tag: The actual tag of the word.
        :param predicted_tag: The predicted tag of the word.
        :return: Tuple (is_prediction_correct, is_known):
        """
        is_prediction_correct = (predicted_tag == actual_tag)
        is_known = word in self.known_words  # checking if the word is known or not
        return is_prediction_correct, is_known

    def apply_pseudo_word_classes(self):
        """
        Apply pseudo words to the emission probabilities and update emission counts accordingly.
        """
        pseudo_words_vocabulary = set()
        pseudo_emission_probs = defaultdict(Counter)
        updated_emission_counts = defaultdict(lambda: defaultdict(int))  # Default to 0 for any new tag-word combination

        for state in self.emission_probs:
            for word in self.emission_probs[state]:
                pseudo_word = self.get_word_when_pseudo_words_is_on(word)
                self.known_words.add(pseudo_word)  # Update the set with pseudo words
                # Accumulate emission probabilities for pseudo words
                pseudo_emission_probs[state][pseudo_word] += self.emission_probs[state][word]
                # Similarly, update emission counts for pseudo words
                updated_emission_counts[state][pseudo_word] += self.emission_counts[state].get(word, 0)
                pseudo_words_vocabulary.add(pseudo_word)

        self.emission_probs = pseudo_emission_probs
        self.emission_counts = updated_emission_counts  # Update emission counts to reflect pseudo words
        self.use_pseudo_words = True
        self.vocabulary = pseudo_words_vocabulary

    def is_frequent_word_in_train_set(self, word, threshold=5):
        """
        Checks if a word is considered frequent in the training set, based on a specified threshold.

        Args:
            word (str): The word to check for frequency.
            threshold (int): The minimum number of occurrences for a word to be considered frequent. Default is 5.

        Returns:
            bool: True if the word is frequent, False otherwise.
        """
        # Calculate the total frequency of the word across all tags
        total_frequency = sum(self.emission_counts[tag].get(word, 0) for tag in self.emission_counts)

        # Check if the word's total frequency across all tags is greater than or equal to the threshold
        return total_frequency >= threshold

    def get_word_when_pseudo_words_is_on(self, word):
        """
        Enhances the replacement of words with pseudo words based on more detailed characteristics.
        This helps in better handling low-frequency and unknown words by grouping them into
        informative categories.
        :param word: The word to replace.
        :return: The pseudo word.
        """
        if self.is_frequent_word_in_train_set(word):
            return word

        # Start with more specific cases, then move to more general ones
        if word[0].isupper() and not word.isupper():  # First word is capitalized and not all words
            return "<CAPITALIZED>"
        if word.isupper():
            return "<ACRONYM>"  # All uppercase, likely an acronym
        if "-" in word:
            return "<HYPHENATED>"  # Hyphenated words often are compound adjectives or nouns
        if word.endswith("ing"):
            return "<GERUND>"  # Gerunds or present participles
        if word.endswith("ed"):
            return "<PAST>"  # Past tense verbs
        if word.endswith("s") and not word.endswith("ss"):
            return "<PLURAL>"  # Plural nouns (simple heuristic, not perfect)
        if word.endswith("able") or word.endswith("ible"):
            return "<ADJECTIVE>"  # Adjectives ending in -able or -ible
        if word.endswith("ly"):
            return "<ADVERB>"  # Adverbs ending in -ly
        if word.isdigit():
            return "<NUMERAL>"  # Purely numeric
        if "$" in word:
            return "<CURRENCY>"
        if "%" in word:
            return "<PERCENTAGE>"  # Percentage
        if any(char.isdigit() for char in word):
            return "<ALPHANUMERIC>"  # Contains numbers, suggesting an alphanumeric identification
        # More specific rules could be added here based on observation of the dataset

        # A fallback for words that don't fit the more specific categories
        return "<UNKNOWN>"  # A generic tag for words that don't fit other categories

    def create_confusion_matrix(self, test_set):
        """
        Generates a confusion matrix for the model's predictions on the test dataset.
        :param test_set: Test dataset with each sentence as a list of (word, tag) tuples.
        :return: Tuple (confusion_matrix, tags_indexer):
        confusion_matrix (numpy.ndarray) - 2D array where each element [i][j] is the count of
        times the tag i was predicted as tag j.
        tags_indexer (list) - List of all unique tags used as indices for the matrix.
        """
        tags_indexer = self._get_all_tags(test_set)  # getting all the tags used in the training and test sets
        confusion_matrix = np.zeros((len(tags_indexer), len(tags_indexer)))
        if self.use_pseudo_words:
            test_set = self.apply_pseudo_words_to_test_sentences(test_set)
        for sentence in test_set:
            words, actual_tags = zip(*sentence)  # getting the words and tags of the sentence
            predicted_tags, _ = self.viterbi_algorithm(words)  # getting the predicted tags of the sentence
            for word, actual_tag, predicted_tag in zip(words, actual_tags, predicted_tags):
                confusion_matrix[tags_indexer.index(actual_tag)][tags_indexer.index(predicted_tag)] += 1

        return confusion_matrix, tags_indexer

    def _get_all_tags(self, test_set):
        """
        Returns a list of all tags used in the training and test sets.
        :param test_set: Test dataset with each sentence as a list of (word, tag) tuples.
        :return: List of all unique tags used in the training and test sets.
        """
        unknown_tags = set()  # initializing a set all the tags used in the test set
        for sentence in test_set:
            for _, tag in sentence:
                unknown_tags.add(tag)
        all_tags = self.known_tags.union(unknown_tags)  # getting all the tags used in the training and test sets
        return list(all_tags)


def evaluate_model(model, test_sentences):
    """
    Evaluates the model on the test sentences and displays the results.
    :param model: The POS tagger model to evaluate.
    :param test_sentences: List of processed test sentences with tags.
    """
    return model.evaluate(test_sentences)


def get_model_evaluation(model, tagged_test_set, model_name):
    """
    Evaluates the model on the test set and displays the evaluation results.
    :param model:
    :param tagged_test_set:
    :param model_name:
    :return:
    """
    known_word_error_rate, unknown_word_error_rate, total_error_rate = evaluate_model(model, tagged_test_set)
    display_evaluation(model_name, known_word_error_rate, total_error_rate, unknown_word_error_rate)


def get_finite_vocabulary(sentences):
    """
    Extracts a set of unique words (vocabulary) from the given sentences.
    :param sentences: List of sentences, where each sentence is a list of (word, tag) tuples.
    :return: Set of unique words in the given sentences.
    """
    # getting the set of unique words in the sentences
    return {word for sentence in sentences for word, _ in sentence}


def display_confusion_matrix(confusion_matrix, known_tags_indexer):
    """
    Displays a heatmap of the confusion matrix for the POS tagger.
    :param confusion_matrix: The confusion matrix, where each element represents the count
        of predictions for a pair of true and predicted tags.
    :param known_tags_indexer: List of tags that index the rows and columns of the confusion matrix.
    """
    plt.imshow(confusion_matrix, cmap='hot')  # Use log scale to make the plot more readable
    plt.colorbar()
    plt.xticks(np.arange(len(known_tags_indexer)), known_tags_indexer, rotation=90, fontsize=2)
    plt.yticks(np.arange(len(known_tags_indexer)), known_tags_indexer, fontsize=2)
    plt.xlabel("Predicted tags")
    plt.ylabel("True tags")
    plt.title("Confusion Matrix of the Bigram HMM tag classifier with pseudo words and laplace add-one "
              "smoothing", fontsize=7, loc='center')
    plt.savefig("confusion_matrix.png", dpi=1000)
    plt.show()


def get_most_frequents_errors(confusion_matrix, known_tags_indexer, error_number=10):
    """
    Get the most frequent errors from the confusion matrix.
    :param confusion_matrix: The confusion matrix, where each element represents the count
        of predictions for a pair of true and predicted tags.
    :param known_tags_indexer: List of tags that index the rows and columns of the confusion matrix.
    :param error_number: The number of most frequent errors to return.
    :return: List of tuples, where each tuple contains the true tag, predicted tag, and the count of errors.
    """
    errors = dict()
    # iterating over the confusion matrix
    for i, true_tag in enumerate(known_tags_indexer):
        for j, predicted_tag in enumerate(known_tags_indexer):
            if i != j and confusion_matrix[i, j] > 0:  # if the true tag is not equal to the predicted tag
                errors[(true_tag, predicted_tag)] = confusion_matrix[i, j]
    # sort the errors by their count in descending order
    sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
    return sorted_errors[:error_number]


def plot_most_frequent_errors(most_frequent_errors):
    """
    Plots the most frequent errors from the confusion matrix.
    :param most_frequent_errors: List of tuples, where each tuple contains the true tag, predicted tag,
        and the count of errors.
    """
    x = [f"{error[0][0]} -> {error[0][1]}" for error in most_frequent_errors]
    y = [error[1] for error in most_frequent_errors]
    plt.bar(x, y)
    plt.xticks(fontsize=5)
    plt.title("Most frequent errors of the Bigram HMM tag classifier with pseudo words and laplace smoothing",
              fontsize=7, loc='center')
    plt.show()


def main():
    # download_corpus()
    tagged_corpus = load_tagged_corpus()
    corpus_vocabulary = get_finite_vocabulary(tagged_corpus)

    processed_tagged_corpus = process_complex_tags(tagged_corpus)
    train_sentences, test_sentences = split_train_test(processed_tagged_corpus)

    # Implementation of the most likely tag baseline
    most_likely_tag_model = MLEMostLikelyTagModel()
    most_likely_tag_model.train(train_sentences)
    evaluate_model(most_likely_tag_model, test_sentences)
    get_model_evaluation(most_likely_tag_model, test_sentences, "MLE Most Likely Tag Model")

    # Implementation of a bigram HMM tagger
    bigram_hmm_tagger = BigramHMMTagger(corpus_vocabulary)
    bigram_hmm_tagger.train(train_sentences)
    get_model_evaluation(bigram_hmm_tagger, test_sentences, "Bigram HMM Tagger")

    # Using Add-one smoothing
    bigram_hmm_tagger.add_one_laplace_emission_probability_smoothing()
    get_model_evaluation(bigram_hmm_tagger, test_sentences, "Bigram HMM Tagger with Add-One Smoothing")

    # Using pseudo-words
    bigram_hmm_tagger.cancel_emission_probability_smoothing()
    bigram_hmm_tagger.apply_pseudo_word_classes()
    get_model_evaluation(bigram_hmm_tagger, test_sentences, "Bigram HMM Tagger with Pseudo Words")

    bigram_hmm_tagger.add_one_laplace_emission_probability_smoothing()
    get_model_evaluation(bigram_hmm_tagger, test_sentences, "Bigram HMM Tagger with Pseudo Words and Add-One Smoothing")

    # Generating a confusion matrix for the bigram HMM tagger
    confusion_matrix, tags_indexer = bigram_hmm_tagger.create_confusion_matrix(test_sentences)
    display_confusion_matrix(confusion_matrix, tags_indexer)

    # Get the most frequent errors from the confusion matrix
    most_frequent_errors = get_most_frequents_errors(confusion_matrix, tags_indexer)

    # plot the most frequent errors
    plot_most_frequent_errors(most_frequent_errors)


if __name__ == "__main__":
    main()
