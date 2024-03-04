# T5_Finetuning_for_NLI_Performance_Analysis

---

## Overview:

This repository contains code for finetuning a T5 transformer model for the analysis of performance in the context of the Natural Language Inference (NLI) task. Specifically, our focus is on generating explanations to understand if this approach can lead to improved performance. Additionally, we aim to gain insights into what aspects of the explanations the model utilizes.

The repository contains three notebooks:

1. **Finetuning Notebook**: `T5Training.ipynb`
2. **Qualitative Analysis Notebook**: `QualitativeAnalysis.ipynb`
3. **Plots Generation Notebook**: `plots.ipynb`

## Usage:

### 1. Finetuning Notebook:

- **File Name**: `T5Training.ipynb`
- **Description**: This notebook contains code for finetuning the T5 transformer model for the NLI task. It includes data preprocessing, model configuration, finetuning loop, and evaluation.
- **Usage**: 
  - Run each cell sequentially to finetune the T5 model.
  - Modify hyperparameters and configurations as needed.
  - The notebook saves the finetuned model weights and other necessary files for later use.

### 2. Qualitative Analysis Notebook:

- **File Name**: `QualitativeAnalysis.ipynb`
- **Description**: This notebook produces samples to conduct qualitative analysis on the finetuned T5 model for the NLI task.
- **Usage**: 
  - Load the finetuned model and necessary files.
  - Run the notebook to sample generated answers from the T5 model for qualitative analysis.

### 3. Plots Generation Notebook:

- **File Name**: `plots.ipynb`
- **Description**: This notebook generates plots based on the training results obtained from `T5Training.ipynb`, contained in the `results.txt` file. It includes visualizations of label accuracy curves and explanation generation quality.
- **Usage**: 
  - Ensure that the necessary files from the finetuning process are available.
  - Run the notebook to generate plots.

## Installation:

1. Clone the repository:
   ```bash
   git clone https://github.com/DreRnc/ExplainingExplanations
   poetry add