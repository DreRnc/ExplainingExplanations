def generate_batch_prompts_mnli(batch):
    """Generate the prompt for MNLI.

    Args:
        batch (dict of lists): a dictionary containing the lists of premises and hypotheses of length batch_size.

    Returns:
        list of str: the prompts.
    """
    prompts = []
    for i in range(len(batch['premise'])):
        prompt = "mnli hypothesis: " + batch['hypothesis'][i] + ' premise: ' + batch['premise'][i]
        prompts.append(prompt)
    return prompts

def generate_prompt_mnli(datapoint):
    """Generate the prompt for MNLI.

    Args:
        datapoint (dict): a dictionary containing the premise and hypothesis.

    Returns:
        str: the prompt.
    """
    prompt = "mnli hypothesis: " + datapoint['hypothesis'] + ' premise: ' + datapoint['premise']
    return prompt

def evaluate_output_mnli(output, label):
    """T5 outputs a string. We need to compare it to the label which is
    0, 1 or 2 (entailment, neutral, contradiction).
    """
    if output == 'entailment':
        output = 0
    elif output == 'neutral':
        output = 1
    elif output == 'contradiction':
        output = 2
    else:
        raise ValueError('Output not recognized')
    return output == label


""" def tokenize_function(example, tokenizer):
    prompts = generate_batch_prompts_mnli(example)
    l = ["entailment", "neutral", "contraddiction"]
    # Tokenize the premise (input) and label
    inputs = tokenizer(prompts, padding='max_length', truncation=True, max_length=128)
    labels = tokenizer([l[i] for i in example["label"]], padding="max_length", truncation=True)

    # Return a dictionary containing input and label tokens
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"],
    } """

def tokenize_function(examples, tokenizer):     
    """Tokenize the dataset. This function is passed to the map method.
    """
    prompts = generate_batch_prompts_mnli(examples)
    return tokenizer(prompts, padding='max_length', truncation=True, max_length=128)