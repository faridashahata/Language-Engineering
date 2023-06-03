import re
import pandas as pd
import torch
from torch.utils.data import (TensorDataset)
from transformers import T5Tokenizer
from config import *

device = torch.device("cuda")

train_df = pd.read_json(TRAIN_DATA_PATH, lines=True)
test_df = pd.read_json(TEST_DATA_PATH, lines=True)
val_df = pd.read_json(VAL_DATA_PATH, lines=True)

# Mean Girls example:
# print("Sample document: ", train_df.iloc[33614].document)
# print("Sample summary: ", train_df.iloc[33614].summary)

print("The shape of the training dataframe: ", train_df.shape)

def prepare_data(df, threshold, tokenizer):
    """
        Preprocesses the data by lowercasing, cleaning, adding "summarize: " before documents, and adding "<pad>" to summaries.

        Args:
            df (pandas.DataFrame): Dataframe containing the document and summary columns.
            threshold (int): Maximum word count threshold for truncating data.
            tokenizer (T5Tokenizer): Tokenizer used to tokenize the data.

        Returns:
            pandas.DataFrame: Processed dataframe.
    """
    df['document'] = df['document'].apply(lambda x: x.lower())

    df['summary'] = df['summary'].apply(lambda x: x.lower())

    # Clean:
    df['document'] = df['document'].str.replace('-', ' ')
    df['summary'] = df['summary'].str.replace('-', ' ')

    # Remove excess white spaces:
    df['document'] = df['document'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))
    df['summary'] = df['summary'].apply(lambda x: re.sub(r'\s([?.!"](?:\s|$))', r'\1', x))
    df['document'] = df['document'].apply(lambda x: " ".join(x.split()))
    df['summary'] = df['summary'].apply(lambda x: " ".join(x.split()))

    # Generate word counts:
    df['document_word_count'] = df['document'].apply(lambda x: len(x.split()))
    df['summary_word_count'] = df['summary'].apply(lambda x: len(x.split()))

    # Add "summarize" before document to adopt T5 format:
    df['document'] = 'summarize: ' + df['document']

    # Add pad token to summaries to adopt T5 format:
    df['summary'] = '<pad>' + df['summary']

    # Truncate data, remove rows exceeding threshold, inplace=True means we don't need to reassign df:
    df.drop(df[df.document_word_count > threshold].index, inplace=True)
    print("The shape of the truncated dataframe: ", df.shape)
    # Remove rows with no summary or document:
    df.drop(df[df.document_word_count <= 0].index, inplace=True)
    print("The shape of the truncated dataframe after removing document wc < 0: ", df.shape)
    df.drop(df[df.summary_word_count <= 0].index, inplace=True)
    print("The shape of the truncated dataframe after removing summary wc < 0: ", df.shape)

    # Generate token counts:
    df['doc_token_count'] = df['document'].apply(lambda x: len(tokenizer.tokenize(x)))

    # Truncate based on tokens, as this is sometimes larger that word count
    df = df[df.doc_token_count <= 1.5 * threshold]

    return df


def tokenize(df, tokenizer, max_len):
    """
        Tokenizes the documents using the given tokenizer.

        Args:
            df (pandas.DataFrame): Dataframe containing the document and summary columns.
            tokenizer (T5Tokenizer): Tokenizer used to tokenize the data.
            max_len (int): Maximum length of the tokenized sequences.

        Returns:
            torch.Tensor: Tokenized input IDs.
            torch.Tensor: Attention masks.
        """
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
    """
        Prepares the dataset by preprocessing the data, tokenizing the documents and summaries, and creating a TensorDataset.

        Args:
            df (pandas.DataFrame): Dataframe containing the document and summary columns.
            tokenizer (T5Tokenizer): Tokenizer used to tokenize the data.
            threshold (int): Maximum word count threshold for truncating data.

        Returns:
            TensorDataset: Prepared dataset.
    """

    df = prepare_data(df, threshold, tokenizer)

    DOC_MAX_LEN = int(1.5 * threshold)
    print("DOC_MAX_LEN: ", DOC_MAX_LEN)
    print("SUMMARY_MAX_LEN: ", threshold)
    doc_input_ids, doc_attention_masks = tokenize(df['document'].values, tokenizer, DOC_MAX_LEN)
    summary_input_ids, summary_attention_masks = tokenize(df['summary'].values, tokenizer, threshold)

    tensor_df = TensorDataset(doc_input_ids, doc_attention_masks, summary_input_ids, summary_attention_masks)
    return tensor_df
