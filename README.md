# DD2418 Language Engineering Project - Movie Summarization

This project is a part of the DD2418 Language Engineering course at KTH. The goal of the project is to fine-tune Torch T5 and BART models using the Narrasum dataset to improve the summarization of text describing movies.

## Project Overview

The project focuses on training and evaluating Transformer-based models for movie summarization. The main steps involved in the project are:

1. Data Preparation: Preprocessing the Narrasum dataset, including cleaning, tokenization, and formatting the data according to the T5/BART model input requirements.

2. Model Training: Fine-tuning the T5 and BART models using the preprocessed data. The models can be trained on a GPU for improved performance (see train_X_with_cuda.py).

3. Evaluation: During the evaluation phase, we will assess the performance of the trained models for text summarization using the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics, which are commonly used for evaluating summarization tasks.

### Resources
- [Torch documentation](https://pytorch.org/docs/stable/index.html): Official documentation for the Torch framework.
- [Hugging Face Transformers documentation](https://huggingface.co/transformers/): Documentation for the Transformers library, which provides pre-trained models and tools for natural language processing tasks.
- [Narrasum dataset](https://arxiv.org/abs/2212.01476): The Narrasum dataset is a collection of movie plot summaries and their corresponding summaries. The dataset is used for training and evaluating the models.
- [ROUGE metrics](https://en.wikipedia.org/wiki/ROUGE_(metric)): The ROUGE metrics are used for evaluating the performance of the models for text summarization.
- [T5 model](https://arxiv.org/abs/1910.10683): The T5 model is a Transformer-based model that can be used for text summarization.
- [BART model](https://arxiv.org/abs/1910.13461): The BART model is a Transformer-based model that can be used for text summarization.