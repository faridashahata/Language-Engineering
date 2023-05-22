import pandas as pd
import os
import re

import pandas as pd
import torch
from torch.utils.data import (TensorDataset)
from tqdm import tqdm
# from pytorch_transformers import AdamW, WarmupLinearSchedule, T5Tokenizer
from transformers import T5Tokenizer

os.getcwd()
os.chdir("/Users/faridashahata/Desktop/Language Engineering/Project")
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

def prepare_data(df, threshold):

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

    # Add "summarize" before document:
    df['document'] = 'summarize: ' + df['document']

    # Add pad token to summaries:
    df['summary'] = '<pad>' + df['summary']

    # Add " </s>" to end of each document and summary:
    # df['document'] = df['document'] + " </s>"
    # df['summary'] = df['summary'] + " </s>"





    # Truncate data:
    df = df[df.document_word_count <= threshold]

    # Remove rows with no summary or document:
    df = df[df.document_word_count > 0]
    df = df[df.summary_word_count > 0]
    
    return df

# train_df = prepare_data(train_df)
# val_df = prepare_data(val_df)
# test_df = prepare_data(test_df)

# plt.hist(train_df['document_word_count'])
# plt.show()

print("len of train dataframe", len(train_df))

# # Mean Girls example, after clean-up:
# print("Cleaned sample document: ", train_df.iloc[33614].document)
# print("Cleaned sample summary: ", train_df.iloc[33614].summary)
# print("Summary word count: ", train_df.iloc[33614].summary_word_count)
# print("Document word count: ", train_df.iloc[33614].document_word_count)
#
# print("Average Summary word count: ", train_df.summary_word_count.mean())
# print("Average Document word count: ", train_df.document_word_count.mean())
#
# print("Max Summary word count: ", train_df.summary_word_count.max())
# print("Max Document word count: ", train_df.document_word_count.max())

# Following this guide closely: http://seekinginference.com/applied_nlp/T5.html


#plt.hist(train_df[train_df.document_word_count <1000].summary_word_count)

#plt.hist(train_df.document_word_count)
#plt.show()


# print("entries with document length > 1000", train_df[train_df.document_word_count>=1000].shape[0])
#




# STEP 2: INSTANTIATE T5 TOKENZIER and TOKENIZE THE DATA:


tokenizer = T5Tokenizer.from_pretrained('t5-small')

print("eos token id: ", tokenizer.eos_token_id)
print("unk token id: ", tokenizer.unk_token_id)
print("pad token id: ", tokenizer.pad_token_id)

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
                        pad_to_max_length=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                        )
        input_ids.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])
    # print("len of input_ids", len(input_ids))
    # print("len of input_ids[0]", input_ids[0].size())
    # print("len of input_ids[0]", input_ids[1].size())
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


# # Get max length for tokenized documents:
#
# token_len = []
# for i in tqdm(range(train_df.shape[0])):
#      token_len.append(len(tokenizer.tokenize(train_df.iloc[i]['document'])))
#
# doc_df = pd.DataFrame({"len_tokens": token_len})
# doc_max_len = doc_df['len_tokens'].max()
#
# print("max length of tokenized documents: ", doc_max_len)
# # max is --1868
#
# # Get max length for tokenized summaries:
# token_len = []
# for i in tqdm(range(train_df.shape[0])):
#     token_len.append(len(tokenizer.tokenize(train_df.iloc[i]['summary'])))
#
# summary_df = pd.DataFrame({"len_tokens": token_len})
# summary_max_len = summary_df['len_tokens'].max()
#
# print("max length of tokenized summaries: ", summary_max_len)
# # max is 955
#
# doc_input_ids, doc_attention_masks = tokenize(train_df['document'].values, tokenizer, 1859)
# summary_input_ids, summary_attention_masks = tokenize(train_df['summary'].values, tokenizer, 907)


#
# PREPARE TENSOR DATASETS: TRAIN, TEST, VAL:

def prepare_dataset(df, tokenizer, threshold):

    #threshold = 1000
    df = prepare_data(df, threshold)


    doc_input_ids, doc_attention_masks = tokenize(df['document'].values, tokenizer, 1859)
    summary_input_ids, summary_attention_masks = tokenize(df['summary'].values, tokenizer, 907)

    tensor_df = TensorDataset(doc_input_ids, doc_attention_masks, summary_input_ids, summary_attention_masks)
    return tensor_df


train_dataset = prepare_dataset(train_df, tokenizer)
val_dataset = prepare_dataset(val_df, tokenizer)
test_dataset = prepare_dataset(test_df, tokenizer)
