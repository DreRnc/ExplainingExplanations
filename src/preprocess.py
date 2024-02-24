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

def save_shuffled_explanations(explanation_dirs):
    """Save the shuffled explanations.
    This function takes a list of explanation directories and saves the shuffled explanations to a file.

    Args:
        explanation_dirs (list of str): the list of directories.
    
    Returns:
        str: the path to the shuffled explanations file.
    """
    if not os.path.exists("ex_files"):
        os.makedirs("ex_files")

    # Read the explanations from the files
    with open(explanation_dirs[0], "r", encoding="utf-8") as f:
        explanations_train = f.readlines()
    with open(explanation_dirs[1], "r", encoding="utf-8") as f:
        explanations_val = f.readlines()
    with open(explanation_dirs[2], "r", encoding="utf-8") as f:
        explanations_test = f.readlines()

    # Shuffle the explanations
    random.shuffle(explanations_train)
    random.shuffle(explanations_val)
    random.shuffle(explanations_test)

    # Save the shuffled explanations to a file
    with open("ex_files/shuffled_explanations_train.txt", "w", encoding="utf-8") as f:
        f.writelines(explanations_train)
    with open("ex_files/shuffled_explanations_val.txt", "w", encoding="utf-8") as f:
        f.writelines(explanations_val)
    with open("ex_files/shuffled_explanations_test.txt", "w", encoding="utf-8") as f:
        f.writelines(explanations_test)

    return ["ex_files/shuffled_explanations_train.txt", "ex_files/shuffled_explanations_val.txt", "ex_files/shuffled_explanations_test.txt"]

def retrieve_explanations(explanation_dirs):
    """Retrieve the explanations.
    This function takes a list of explanation directories and retrieves the explanations.

    Args:
        explanation_dirs (list of str): the list of directories.
    
    Returns:
        dict: the explanations.
    """
    explanations = {}

    # Read the explanations from the files
    with open(explanation_dirs[0], "r", encoding="utf-8") as f:
        explanations["train"] = f.readlines()
    with open(explanation_dirs[1], "r", encoding="utf-8") as f:
        explanations["validation"] = f.readlines()
    with open(explanation_dirs[2], "r", encoding="utf-8") as f:
        explanations["test"] = f.readlines()

    return explanations