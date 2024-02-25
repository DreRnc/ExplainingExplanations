import numpy as np
import random

def generate_batch_prompts_mnli(batch):
    """Generate the prompt with the MNLI pretrained format.

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
    """Generate the prompt with the MNLI pretrained format.

    Args:
        datapoint (dict): a dictionary containing the premise and hypothesis.

    Returns:
        str: the prompt.
    """
    prompt = "mnli hypothesis: " + datapoint['hypothesis'] + ' premise: ' + datapoint['premise']
    return prompt

def generate_batch_prompts(batch):
    """Generate the prompt for finetuning on eSNLI.

    Args:
        batch (dict of lists): a dictionary containing the lists of premises and hypotheses of length batch_size.

    Returns:
        list of str: the prompts.
    """
    prompts = []
    for i in range(len(batch['premise'])):
        prompt = "hypothesis: " + batch['hypothesis'][i] + ' premise: ' + batch['premise'][i]
        prompts.append(prompt)
    return prompts

def generate_prompt(datapoint):
    """Generate the prompt for finetuning on eSNLI.

    Args:
        datapoint (dict): a dictionary containing the premise and hypothesis.

    Returns:
        str: the prompt.
    """
    prompt = "hypothesis: " + datapoint['hypothesis'] + ' premise: ' + datapoint['premise']
    return prompt

def remove_explanation(label):
    """Convert the label to a number.
    Take in input a string and retrieve the word after 'label: '.
    Then convert it to number and return it.
    If the prediction string is not in the format 'label: <label> explanation: <explanation>',
    return -1.

    Args:
        label (str): the label.

    Returns:
        int: the number corresponding to the label.
    """
    try:
        return label.split('label: ')[1].split(' explanation: ')[0]
    except:
        return '-1'
    
def tokenize_function(example, tokenizer, use_mnli_format = False):
    """Tokenize mapping function.
    This function generates the promopt for the T5 model and tokenizes it.
    The label is the tokenization of the label class.
    
    Args:
        example (dict): the example to tokenize.
        tokenizer (transformers.PreTrainedTokenizer): the tokenizer.
        use_mnli_format (bool): whether to use the mnli format for prompts or not
        
    Returns:
        dict: the tokenized prompt and label.
    """
    if use_mnli_format:
        prompts = generate_batch_prompts_mnli(example)
    else:
        prompts = generate_batch_prompts(example)
    
    l = ["entailment", "neutral", "contradiction"]
    # Tokenize the premise (input) and label
    inputs = tokenizer(prompts, truncation=True, max_length=128)
    labels_tokenized = tokenizer([l[i] for i in example["label"]], truncation=True)

    # Return a dictionary containing input and label tokens
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels_tokenized["input_ids"],
    }

def tokenize_function_ex(example, tokenizer, explanations = None, use_mnli_format = False):
    """Tokenize mapping function.
    This function generates the promopt for the T5 model and tokenizes it.
    The label is the tokenization of the label and explanation in the following fromat:
    "label: <label> explanation: <explanation>"

    Args:
        example (dict): the example to tokenize.
        tokenizer (transformers.PreTrainedTokenizer): the tokenizer.
        explanations (list of str): the explanations to use. If None, the explanation_1 field of the example is used.
        use_mnli_format (bool): whether to use the mnli format for prompts or not
    
    Returns:
        dict: the tokenized prompt and label.
    """
    if use_mnli_format:
        prompts = generate_batch_prompts_mnli(example)
    else:
        prompts = generate_batch_prompts(example)
    
    l = ["entailment", "neutral", "contradiction"]
    # Tokenize the premise (input) and label
    inputs = tokenizer(prompts, truncation=True, max_length=128)
    label_classes = [l[i] for i in example["label"]]

    if explanations is None:
        explanations = example['explanation_1']
        
    labels_tokenized = tokenizer([f"label: {label} explanation: {explanation}" for label, explanation in zip(label_classes, explanations)], truncation=True)

    # Return a dictionary containing input and label tokens
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels_tokenized["input_ids"],
    }

def compute_metrics(eval_pred, pred_transform, metric):
    """Compute the metrics.

    Args:
        eval_pred (EvalPrediction): the predictions and labels.
        pred_transform (function): the function to transform the logits and labels.
        metric (datasets.Metric): the metric.

    Returns:
        dict: the computed metrics.

    """
    pred, labels = pred_transform(eval_pred)

    return metric.compute(predictions=pred, references=labels)

def convert_label_to_num_mnli(label):
    """Convert the label to a number.

    Args:
        label (str): the label.

    Returns:
        int: the number corresponding to the label.
    """
    if label == 'entailment':
        return 0
    elif label == 'neutral':
        return 1
    elif label == 'contradiction':
        return 2
    else:
        return -1

def eval_pred_transform_accuracy(eval_pred, tokenizer, remove_explanations_from_label = False, debugging = False):
    """Transform the logits and labels to compute the accuracy.

    Args:
        eval_pred (EvalPrediction): the predictions and labels.
        tokenizer (transformers.PreTrainedTokenizer): the tokenizer.

    Returns:
        tuple: predictions and labels in format (list of int).
    """
    if debugging:
        print('eval_pred.predictions.shape:', eval_pred.predictions.shape)
        print('eval_pred.label_ids.shape:', eval_pred.label_ids.shape)
        print('eval_pred.predictions:', eval_pred.predictions)
        print('eval_pred.label_ids:', eval_pred.label_ids)
    
    pred = eval_pred.predictions
    labels = eval_pred.label_ids

    # Remove the -100 padding put by the collator to make data of same length
    # -100 is chosen as padding as it is automatically ignored by the PyTorch losses
    pred = np.where(pred != -100, pred, tokenizer.pad_token_id)
    pred = tokenizer.batch_decode(pred, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    if debugging:
        print('Before removing explanations:\n', 'pred', pred[:5], '\n', 'labels:', labels[:5])

    if remove_explanations_from_label:
        pred = [remove_explanation(p) for p in pred]
        labels = [remove_explanation(l) for l in labels]

    if debugging:
        print('After removing explanations:\n', 'pred', pred[:5], '\n', 'labels:', labels[:5])

    pred_nums = [convert_label_to_num_mnli(p) for p in pred]
    labels_nums = [convert_label_to_num_mnli(l) for l in labels]

    num_invalid_answers = len([p for p in pred_nums if p not in [0,1,2]])
    if num_invalid_answers > 0:
        print('Number of predictions not in [entailment, neutral, contradiction]:', num_invalid_answers)

    return pred_nums, labels_nums