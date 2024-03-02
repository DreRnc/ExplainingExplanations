import random
import os

def prepare_dataset(dataset, tokenize_mapping, sizes):
    """Prepare the dataset.
    This function takes a dataset and prepares it for training.

    Args:
        dataset (datasets.Dataset): the dataset to prepare.
        tokenize_mapping (function or tuple of functions): the function to tokenize the dataset.
        sizes (dict): the sizes of the dataset.

    Returns:
        tuple: the train, validation and test datasets.
    """
    training_set = dataset["train"]
    validation_set = dataset["validation"]
    test_set = dataset["test"]

    train = training_set.select(range(sizes['n_train']))
    valid = validation_set.select(range(sizes['n_val']))
    test = test_set.select(range(sizes['n_test']))

    if isinstance(tokenize_mapping, tuple):
        train_tokenized = train.map(tokenize_mapping[0], batched=True).with_format("torch")
        valid_tokenized = valid.map(tokenize_mapping[1], batched=True).with_format("torch")
        test_tokenized = test.map(tokenize_mapping[2], batched=True).with_format("torch")
    else:
        train_tokenized = train.map(tokenize_mapping, batched=True).with_format("torch")
        valid_tokenized = valid.map(tokenize_mapping, batched=True).with_format("torch")
        test_tokenized = test.map(tokenize_mapping, batched=True).with_format("torch")

    train_tokenized = train_tokenized.remove_columns(['label'])
    valid_tokenized = valid_tokenized.remove_columns(['label'])
    test_tokenized = test_tokenized.remove_columns(['label'])

    return train_tokenized, valid_tokenized, test_tokenized

def save_explanations(dataset):
    """Save the explanations.
    This function takes a dataset and saves the explanations to a file.

    Args:
        dataset (datasets.Dataset): the dataset to prepare.
    
    Returns:
        dirs (list of str): the list of directories.
    """
    if not os.path.exists("ex_files"):
        os.makedirs("ex_files")
    
    texts = []
    texts = [example["explanation_1"] for example in dataset["train"]]
    with open("ex_files/explanations_train.txt", "w", encoding="utf-8") as f:
        f.writelines(text + "\n" for text in texts)

    texts = []
    texts = [example["explanation_1"] for example in dataset["validation"]]
    with open("ex_files/explanations_val.txt", "w", encoding="utf-8") as f:
        f.writelines(text + "\n" for text in texts)

    texts = []
    texts = [example["explanation_1"] for example in dataset["test"]]
    with open("ex_files/explanations_test.txt", "w", encoding="utf-8") as f:
        f.writelines(text + "\n" for text in texts)

    return ["ex_files/explanations_train.txt", "ex_files/explanations_val.txt", "ex_files/explanations_test.txt"]

def save_shuffled_explanations(explanation_dir):
    """Save the shuffled explanations.

    Args:
        explanation_dir (str): directory of file with explanations to shuffle
    
    Returns:
        str: the path to the shuffled explanations file.
    """
    if not os.path.exists("ex_files"):
        os.makedirs("ex_files")

    # Read the explanations from the files
    with open(explanation_dir, "r", encoding="utf-8") as f:
        explanations_train = f.readlines()

    # Shuffle the explanations
    random.shuffle(explanations_train)

    # Save the shuffled explanations to a file
    with open("ex_files/shuffled_explanations_train.txt", "w", encoding="utf-8") as f:
        f.writelines(explanations_train)

    return "ex_files/shuffled_explanations_train.txt"

def retrieve_explanations(explanation_dir):
    """Retrieve the explanations from a txt file.
    Args:
        explanation_dirs (str): the directory of txt file with explanations.
    
    Returns:
        lst of str: the explanations.
    """

    # Read the explanations from the files
    with open(explanation_dir, "r", encoding="utf-8") as f:
        explanations= f.readlines()

    return explanations