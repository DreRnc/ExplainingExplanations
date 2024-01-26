def generate_prompt_mnli(datapoint):
    return "mnli hypothesis: " + datapoint['hypothesis'] + ' premise: ' + datapoint['premise']

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
