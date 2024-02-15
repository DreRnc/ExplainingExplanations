import torch
import numpy as np
from functools import partial
import evaluate 

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

def tokenize_function(example, tokenizer):
    """Tokenize mapping function.
    This function generates the promopt for the T5 model and tokenizes it.
    The label is the tokenization of the label class.
    
    Args:
        example (dict): the example to tokenize.
        tokenizer (transformers.PreTrainedTokenizer): the tokenizer.
        
    Returns:
        dict: the tokenized prompt and label.
    """

    prompts = generate_batch_prompts_mnli(example)
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

def tokenize_function_ex(example, tokenizer):
    """Tokenize mapping function.
    This function generates the promopt for the T5 model and tokenizes it.
    The label is the tokenization of the label and explanation in the following fromat:
    "label: <label> explanation: <explanation>"

    Args:
        example (dict): the example to tokenize.
        tokenizer (transformers.PreTrainedTokenizer): the tokenizer.
    
    Returns:
        dict: the tokenized prompt and label.
    """
    prompts = generate_batch_prompts_mnli(example)
    l = ["entailment", "neutral", "contradiction"]
    # Tokenize the premise (input) and label
    inputs = tokenizer(prompts, truncation=True, max_length=128)
    label_classes = [l[i] for i in example["label"]]
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

def eval_pred_transform_accuracy(eval_pred, tokenizer):
    """Transform the logits and labels to compute the accuracy.

    Args:
        eval_pred (EvalPrediction): the predictions and labels.
        tokenizer (transformers.PreTrainedTokenizer): the tokenizer.

    Returns:
        tuple: predictions and labels in format (list of int).
    """
    #print('eval_pred.predictions:', eval_pred.predictions)
    #print('eval_pred.label_ids:', eval_pred.label_ids)
    pred = eval_pred.predictions
    labels = eval_pred.label_ids

    pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
    pred_nums = [convert_label_to_num_mnli(p) for p in pred]

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels_nums = [convert_label_to_num_mnli(l) for l in labels]

    print('Number of predictions not in [entailment, neutral, contradiction]:', len([p for p in pred_nums if p not in [0, 1, 2]])) 
    print('Wrong predictions and labels:', [(p, l) for p, l in zip(pred, labels) if p != l])
    return pred_nums, labels_nums