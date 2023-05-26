import re

import pandas as pd
import torch
from torch.utils.data import (TensorDataset)
from transformers import T5Tokenizer

train_df = pd.read_json("./data/train.jsonl", lines=True)
test_df = pd.read_json("./data/test.jsonl", lines=True)
val_df = pd.read_json("./data/validation.jsonl", lines=True)

# Mean Girls example:
print("Sample document: ", train_df.iloc[33614].document)
print("Sample summary: ", train_df.iloc[33614].summary)

print("The shape of the training dataframe: ", train_df.shape)

# STEP 1: DATA PREPARATION :
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""  Before Using the T5 Tokenizer, we need to do the following:
     Add pad token before summaries
     Add "summarize: " before documents to summarize
     Add "</s>" to end of summary and document  """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
global MODEL_NAME
MODEL_NAME = 't5-large'
global BATCH_SIZE
BATCH_SIZE = 64


def prepare_data(df, threshold, tokenizer):
    # Lowercase the data:
    df['document'] = df['document'].apply(lambda x: x.lower())
    df['summary'] = df['summary'].apply(lambda x: x.lower())

    # Clean:
    df['document'] = df['document'].str.replace('-', ' ')
    df['summary'] = df['summary'].str.replace('-', ' ')

    df['document'] = df['document'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))
    df['summary'] = df['summary'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))

    # remove excess white spaces
    df['document'] = df['document'].apply(lambda x: " ".join(x.split()))
    df['summary'] = df['summary'].apply(lambda x: " ".join(x.split()))

    # Generate word counts:
    df['document_word_count'] = df['document'].apply(lambda x: len(x.split()))
    df['summary_word_count'] = df['summary'].apply(lambda x: len(x.split()))

    # df['summary_token_count'] = df['summary'].apply(lambda x: len(tokenizer.tokenize(x)))

    # Add "summarize" before document:
    df['document'] = 'summarize: ' + df['document']
    # Add pad token to summaries:
    df['summary'] = '<pad>' + df['summary']
    # Truncate data:
    new_df = df[df.document_word_count <= threshold].copy()
    # Generate token counts:
    new_df['doc_token_count'] = new_df['document'].apply(lambda x: len(tokenizer.tokenize(x)))

    # Truncate based on tokens:
    new_df = new_df[new_df.doc_token_count <= 1.5 * threshold]

    # Remove rows with no summary or document:
    new_df = new_df[new_df.document_word_count > 0]
    new_df = new_df[new_df.summary_word_count > 0]

    return new_df

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

""" 
The output of tokenizer is a dictionary containing two keys – input ids and attention mask. 
Input ids are the unique identifiers of the tokens in a sentence.
Attention mask is used to batch the input sequence together and indicate whether the token 
should be attended by our model or not.Token with attention mask value 0 means token will 
be ignored and 1 means tokens are important and will be taken for further processing.
Source: https://neptune.ai/blog/hugging-face-pre-trained-models-find-the-best#:~:text=The%20output%20of%20tokenizer%20is,by%20our%20model%20or%20not.
"""
def tokenize(df, tokenizer, max_len):
    input_ids = []
    attention_masks = []

    for document in df:
        encoded_dict = tokenizer.encode_plus(
                document,
                add_special_tokens=True,
                max_length=max_len,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

def prepare_dataset(df, tokenizer, threshold):
    # threshold = 1000
    df = prepare_data(df, threshold, tokenizer)

    DOC_MAX_LEN = int(1.5 * threshold)
    print("DOC_MAX_LEN", DOC_MAX_LEN)
    print("SUMMARY_MAX_LEN", threshold)
    doc_input_ids, doc_attention_masks = tokenize(df['document'].values, tokenizer, DOC_MAX_LEN)
    summary_input_ids, summary_attention_masks = tokenize(df['summary'].values, tokenizer, threshold)

    tensor_df = TensorDataset(doc_input_ids, doc_attention_masks, summary_input_ids, summary_attention_masks)
    return tensor_df

