###################################################
# Exercise 4 - Natural Language Processing 67658  #
###################################################

# Constants
RANDOM_STATE = 21

import numpy as np
from matplotlib import pyplot as plt

# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }


def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=RANDOM_STATE)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=RANDOM_STATE)

    # train
    train_len = int(portion * len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


# Q1
def linear_classification(portion=1.):
    """
    Perform linear classification using Logistic Regression and TFIDF features.
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    LOGISTIC_REGRESSION_MAX_ITER = 1000
    TFIDF_MAX_FEATURES = 1000

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Fetch the data
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Initialize the TFIDF vectorizer
    tf = TfidfVectorizer(stop_words='english', max_features=TFIDF_MAX_FEATURES)

    # Fit and transform the train and test data to create TFIDF vectors
    x_train_tf, x_test_tf = tf.fit_transform(x_train), tf.transform(x_test)

    # Initialize and train the Logistic Regression model
    # random_state is chosen to make the model's behavior deterministic, ensuring reproducibility of results, and
    # max_iter is selected to give the optimization algorithm sufficient iterations to converge to a good
    # solution, improving the model's accuracy.
    lr = LogisticRegression(max_iter=LOGISTIC_REGRESSION_MAX_ITER)

    # Fit the model to the training data
    lr.fit(x_train_tf, y_train)

    # Make predictions on the test data
    y_pred = lr.predict(x_test_tf)

    # Calculate and return the classification accuracy
    return accuracy_score(y_test, y_pred)


# Q2
def transformer_classification(portion=1.):
    """
    Perform transformer-based classification using a pre-trained model.
    :param portion: portion of the data to use
    :return: classification accuracy and loss
    """
    TOKENIZER_MAX_LENGTH = 512
    TRANSFORMER_MODEL_NAME = 'distilroberta-base'
    TRAIN_EPOCHS = 3
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01

    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments)
    from datasets import load_metric

    # Define the dataset class
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokenizer):
            """
            Initializes the dataset with the given texts, labels, and tokenizer. These are the raw strings to classify
            :param texts: the input texts
            :param labels: the corresponding labels
            :param tokenizer: the tokenizer to use for tokenizing the input texts. This tokenizer is used to preprocess
            the texts by converting them into a numerical format that the model can understand.
            """
            # Arguments ensures that all sequences are truncated or padded to a uniform length (max_length)
            # self.encodings is a dictionary containing the tokenized input texts and their corresponding attention
            # masks, a binary mask indicating the presence of a token (1) or padding (0)
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=TOKENIZER_MAX_LENGTH)
            self.labels = labels

        def __getitem__(self, idx):
            """
            Allows PyTorch's DataLoader to access and iterate over the data efficiently
            :param idx: index of the sample
            :return: the item dictionary, which serves as a single, ready-to-process data point in the model,
            containing the tokenized input, its attention mask, and the label
            """
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            """
            Returns the total number of items (text samples) in the dataset.
            :return: the dataset's total size.
            """
            return len(self.labels)

    # Load the accuracy metric
    metric = load_metric("accuracy")

    # Function to compute metrics
    def compute_metrics(eval_pred):
        """
        Compute the accuracy metric for the given predictions.
        :param eval_pred: the evaluation predictions
        :return: the accuracy metric
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        res = metric.compute(predictions=predictions, references=labels)
        return metric.compute(predictions=predictions, references=labels)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    # Using a type of Transformer model that is part of the RoBERTa family of models, which are
    # optimized for performance and efficiency. The model is pre-trained on a large corpus of text data and fine-tuned
    # for sequence classification tasks, by adding a classification head on top of the pre-trained model, turning it
    # from a base model, which is good at understanding language in a general sense, into a model that is specialized
    # for the specific task of classifying text into different categories.
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_NAME, num_labels=len(category_dict))

    # Fetch and tokenize the data
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    train_dataset = Dataset(x_train, y_train, tokenizer)
    test_dataset = Dataset(x_test, y_test, tokenizer)

    #  Trainer goes hand-in-hand with the TrainingArguments class, which offers a wide range of options to customize
    #  how a model is trained. Together, these two classes provide a complete training API.
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",  # Evaluation is done at the end of each epoch.
        save_strategy="epoch",  # save at the end of each epoch
        learning_rate=LEARNING_RATE,
        save_total_limit=3,  # Only keep the last 3 models
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model in its current state (i.e., as it is after the last training epoch), testing the model's
    # performance on the test dataset.
    eval_results = trainer.evaluate()

    return eval_results['eval_accuracy'], eval_results['eval_loss']


# Q3
def zeroshot_classification(portion=1.):
    """
    Perform zero-shot classification using a pre-trained model.

    :param portion: portion of the data to use
    :return: classification accuracy
    """
    # The zero-shot-classification pipeline allows you to use models like cross-encoder/nli-MiniLM2-L6-H768 for
    # zero-shot classification tasks by providing the text to classify, along with a list of candidate labels.
    # The pipeline uses the underlying NLI transformer based model to infer which of these labels best fits the input
    # text, even though the model was not explicitly trained on these labels. Internally, it frames the task as a
    # series of NLI problems, essentially asking, "Given the input text as a premise, how likely is each candidate
    # label to be a true hypothesis?" The label with the highest entailment score is chosen as the classification result

    ZERO_SHOT_MODEL_NAME = 'cross-encoder/nli-MiniLM2-L6-H768'
    ZERO_SHOT_TASK = 'zero-shot-classification'

    from transformers import pipeline
    # Fetch the data
    _, _, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Load the zero-shot classification pipeline with the specified model
    clf = pipeline(ZERO_SHOT_TASK, model=ZERO_SHOT_MODEL_NAME)

    # Define candidate labels for classification
    candidate_labels = list(category_dict.values())

    # Map the category indices to their descriptions for easier interpretation of the results
    label_map = {i: desc for i, desc in enumerate(category_dict.values())}
    string_y_test = [label_map[label] for label in y_test]

    # Perform classification and collect predictions
    predictions = []
    for text in x_test:
        # Use the classifier to predict the label of each text
        output = clf(text, candidate_labels=candidate_labels, hypothesis_template="This text is about {}.")
        top_prediction_index = np.argmax(output['scores'])  # The index of the highest score
        predicted_label = output['labels'][top_prediction_index]
        predictions.append(predicted_label)

    # Calculate accuracy
    correct_predictions = sum([pred == true for pred, true in zip(predictions, string_y_test)])
    accuracy = correct_predictions / len(y_test)

    return accuracy


##### Utility functions #####

def plot_model_accuracy(portions, accuracy_scores, model_name='Model'):
    """
    Plot the model accuracy as a function of the portion of the data used.

    Parameters:
    - portions: A list of portions of the data used for training (e.g., [0.1, 0.5, 1.0]).
    - accuracy_scores: A list of accuracy scores corresponding to each portion.
    - model_name: A string representing the name of the model (for labeling purposes).
    """
    # Create the plot
    plt.figure(figsize=(8, 5))  # Set figure size
    plt.plot(portions, accuracy_scores, '-o', label=f'{model_name} Accuracy')  # Plot the line with markers

    # Adding title and labels
    plt.title(f'{model_name} Accuracy vs. Portion of Data Used')
    plt.xlabel('Portion of Data Used')
    plt.ylabel('Accuracy Score')
    plt.xticks(portions, [f'{p * 100}%' for p in portions])  # Convert portions to percentage on the x-axis

    # Adding a grid for better readability
    plt.grid(True)

    # Adding a legend
    plt.legend()

    # Show the plot
    plt.show()


##### Main #####

if __name__ == "__main__":
    portions = [0.1, 0.5, 1.]
    # Q1
    linear_classification_accuracy_scores = []
    print("Logistic regression results:")
    for p in portions:
        print(f"Portion: {p}")
        linear_classification_accuracy_scores.append(linear_classification(p))
        print(linear_classification_accuracy_scores[-1])

    # Create the plot
    plot_model_accuracy(portions, linear_classification_accuracy_scores, model_name='Logistic Regression')

    # # Q2
    transformer_accuracy_scores = []
    print("\nFine-tuning results:")
    for p in portions:
        accuracy, loss = transformer_classification(portion=p)
        transformer_accuracy_scores.append(accuracy)  # Only append accuracy
        print(f"Portion: {p}, Accuracy: {accuracy}, Loss: {loss}")

    # Now, transformer_accuracy_scores only contains accuracy scores, so plotting will only plot accuracy
    plot_model_accuracy(portions, transformer_accuracy_scores, model_name='Transformer Fine-tuning')

    # Q3
    print("\nZero-shot result:")
    # Since the zero-shot classification doesn't use the training data, we only need to call it once
    print(zeroshot_classification())
