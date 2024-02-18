def prepare_dataset(dataset, tokenize_mapping, sizes):
    training_set = dataset["train"]
    validation_set = dataset["validation"]
    test_set = dataset["test"]

    train = training_set.select(range(sizes['n_train']))
    valid = validation_set.select(range(sizes['n_val']))
    test = test_set.select(range(sizes['n_test']))

    train_tokenized = train.map(tokenize_mapping, batched=True).with_format("torch")
    valid_tokenized = valid.map(tokenize_mapping, batched=True).with_format("torch")
    test_tokenized = test.map(tokenize_mapping, batched=True).with_format("torch")

    train_tokenized = train_tokenized.remove_columns(['label'])
    valid_tokenized = valid_tokenized.remove_columns(['label'])
    test_tokenized = test_tokenized.remove_columns(['label'])

    return train_tokenized, valid_tokenized, test_tokenized


