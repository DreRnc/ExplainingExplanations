import torch
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


def tokenize_function(example, tokenizer):
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
    }

def compute_metrics(eval_pred, transform, metric):
    """Compute the metrics.

    Args:
        eval_pred (EvalPrediction): the predictions and labels.
        transform (function): the function to transform the logits and labels.
        metric (datasets.Metric): the metric.

    Returns:
        dict: the computed metrics.

    """
    pred, labels = transform(eval_pred) 
    return metric.compute(predictions=pred, references=labels)

def eval_pred_transform_accuracy(eval_pred, tokenizer):
    """Transform the logits and labels to compute the accuracy.

    Args:
        eval_pred (EvalPrediction): the predictions and labels.
        tokenizer (transformers.PreTrainedTokenizer): the tokenizer.

    Returns:
        tuple: predictions and labels.

    """
    pred_ids = eval_pred.predictions[0]
    labels = eval_pred.label_ids
    print('pred_ids', pred_ids)
    print('labels', labels)
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print('pred_str', pred_str)
    print('label_str', label_str)
    return pred_str, label_str

def preprocess_logits_argmax(logits, labels):
    """Pre-process the logits and labels to compute the metrics.

    Args:
        logits (list of torch.Tensor): the logits and the labels logits.
        labels (torch.Tensor): the labels.

    Returns:
        tuple: predictions and labels.

    """
    pred_ids = torch.argmax(logits[0], axis=-1)

    return pred_ids, labels

compute_accuracy = partial(compute_metrics, transform=eval_pred_transform_accuracy, metric = evaluate.load('accuracy'))