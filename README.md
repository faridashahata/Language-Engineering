# DD2418 Language Engineering Project - Movie Summarization

This project is a part of the DD2418 Language Engineering course at KTH. The goal of the project is to build a narrative text summarizer by fine-tuning Transformers (namely, Torch T5 and BART models), to generate movie and TV episode summaries out of corresonding plots.

## Project Overview

The project focuses on training and evaluating Transformer-based models for movie summarization. The main steps involved in the project are:

1. Data Preparation: Preprocessing the Narrasum dataset, including cleaning, tokenization, and formatting the data according to the T5/BART model input requirements.

2. Model Training: Fine-tuning the T5 and BART models using the preprocessed data. The models can be trained on a GPU for improved performance (see train_X_with_cuda.py).

3. Evaluation: During the evaluation phase, we will assess the performance of the trained models for text summarization using the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics, which are commonly used for evaluating summarization tasks.

## Getting Started
To train and evaluate the models, you will need to install the required dependencies

Run the following command to install the dependencies:
``` bash
pip install -r requirements.txt
```

### Training the Models
Run this to prepare the data and start the training process for the T5 model:

``` bash
python t5/train_t5.py
```

And this to prepare the data and start the training process for the BART model:

``` bash
python bart/train_bart.py
```

### Evaluating the Models
Run this to evaluate the T5 model:

``` bash
python t5/testing_t5.py
```

And this to evaluate the BART model:

``` bash
python bart/testing_bart.py
```

### Adjust hyperparameters
The hyperparameters for the models can be adjusted in the config.py files in the t5 and bart folders.



### Resources
- [Torch documentation](https://pytorch.org/docs/stable/index.html): Official documentation for the Torch framework.
- [Hugging Face Transformers documentation](https://huggingface.co/transformers/): Documentation for the Transformers library, which provides pre-trained models and tools for natural language processing tasks.
- [Narrasum dataset](https://github.com/zhaochaocs/narrasum) The Narrasum dataset is a collection of movie plot summaries and their corresponding summaries. The dataset is used for training and evaluating the models.
- [ROUGE metrics](https://en.wikipedia.org/wiki/ROUGE_(metric)): The ROUGE metrics are used for evaluating the performance of the models for text summarization.
- [T5 model](https://arxiv.org/abs/1910.10683): The T5 model is a Transformer-based model that can be used for text summarization.
- [BART model](https://arxiv.org/abs/1910.13461): The BART model is a Transformer-based model that can be used for text summarization.
