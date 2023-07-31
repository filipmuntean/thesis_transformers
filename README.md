# Bsc Thesis @VU Amsterdam

<!-- <img src="https://github-readme-stats.vercel.app/api/top-langs?username=filipmuntean&show_icons=true&locale=en&layout=compact&theme=chartreuse-dark" alt="ovi" /> -->
![Static Badge](https://img.shields.io/badge/PyTorch-1.1-brightgreen)

## Integrating a Recurrent Connection in a Pre-trained Transformer Model

This project investigates the integration of a recurrent connection within a pre-trained transformer model (DistilGPT-2) using PyTorch. The main goal is to explore whether adding a recurrent connection to a pure attention-based model is worth the expense of increased parameters, longer training times, and other architectural constraints.

## Dependencies

Before running the project, ensure the following dependencies are installed:

- pytorch >= 1.10
- torch
- tqdm
- numpy
- wandb

You can install these dependencies by using the following command:

```python setup.py bdist_wheel sdists ```

## Project Structure

The project is structured as follows:

### 1. Loading the data

This section involves loading and preprocessing the data before feeding it to the model.

### 2. Adding basic & multiheadself attention on top of the classifier

The initial step is to build a basic model as a starting point for further enhancements. In this step, basic and multi-head self-attention layers are added on top of the classifier model.

### 4. Adding a transformer on top of the classifier

Next, a complete transformer is added on top of the classifier to introduce a more complex model.

### 5. Building a generator transformer

A generator transformer is implemented in this section, enhancing the model's capabilities.

### 6. Adding a recurrence layer on top of the generator transformer

The main focus of this project, a recurrent connection, is integrated into the generator transformer.

### 7. Results

Finally, the results and metrics (loss, gradient clipping norm, and perplexity) are tracked using weights & biases for evaluation.

## Conclusion

This project aims to understand the impact of a recurrent connection in a pre-trained transformer model. By comparing the performance, training time, and architectural constraints, we can determine whether the addition of recurrent connections is beneficial in the context of attention-based models.


