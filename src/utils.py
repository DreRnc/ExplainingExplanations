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
    #l = ["entailment", "neutral", "contraddiction"]
    # Tokenize the premise (input) and label
    inputs = tokenizer(prompts, padding='max_length', truncation=True, max_length=128)
    #labels = tokenizer([l[i] for i in example["label"]], padding="max_length", truncation=True)
    labels = [torch.reshape(torch.tensor(example["label"]), (-1, 1))]
    print('input_ids:', inputs["input_ids"])
    # Return a dictionary containing input and label tokens
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "label": labels
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
    print('eval_pred:', eval_pred)
    print('eval_pred.predictions:', eval_pred.predictions)
    print('eval_pred.predictions[0]:', eval_pred.predictions[0])
    print('eval_pred.label_ids:', eval_pred.label_ids)
    #pred, labels = pred_transform(eval_pred)
    pred = eval_pred.predictions[0]
    labels = eval_pred.label_ids
    return metric.compute(predictions=pred, references=labels)

def eval_pred_transform_accuracy(eval_pred, tokenizer):
    """Transform the logits and labels to compute the accuracy.

    Args:
        eval_pred (EvalPrediction): the predictions and labels.
        tokenizer (transformers.PreTrainedTokenizer): the tokenizer.

    Returns:
        tuple: predictions and labels in format (list of int).
    """
    pred_ids = eval_pred.predictions[1]
    label_ids = eval_pred.label_ids
    print('pred_ids:', pred_ids)
    print('label_ids:', label_ids)

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    print('pred_str:', pred_str)
    print('label_str:', label_str)

    # Convert the string labels to integers
    l = ["entailment", "neutral", "contraddiction"]
    pred = []
    for i in range(len(pred_str)):
        try:
            pred.append(l.index(pred_str[i]))
        except ValueError:
            pred.append(-1)
    labels = [l.index(label) for label in label_str]

    print('Number of predicted non valid labels:', pred.count(-1))

    return pred, labels

def preprocess_logits_argmax(logits, labels):
    """Pre-process the logits and labels to compute the metrics.

    Args:
        logits (list of torch.Tensor): the logits and the labels logits.
        labels (torch.Tensor): the labels.

    Returns:
        tuple: predictions and labels.

    """
    print('logits:', logits)
    pred_ids = torch.argmax(logits[0], dim=-1)
    print('pred_ids:', pred_ids)
    return pred_ids, labels