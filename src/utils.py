import torch

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

def convert_labels(output):
    """Convert the output of the model "entailment", "neutral", "contradiction" to 0, 1, 2.

    Args:
        output (lst of str): the output of the model.

    Returns:
        torch.tensor: the labels.
    """
    labels = []
    for i in range(len(output)):
        if output[i] == "entailment":
            labels.append(0)
        elif output[i] == "neutral":
            labels.append(1)
        elif output[i] == "contradiction":
            labels.append(2)
        else: 
            raise ValueError("The output of the model is not valid.")
    return torch.tensor(labels)

def tokenize_function(examples, tokenizer):     
    """Tokenize the dataset. This function is passed to the map method.
    """
    prompts = generate_batch_prompts_mnli(examples)
    return tokenizer(prompts, padding='max_length', truncation=True, max_length=128)