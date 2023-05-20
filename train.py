import preprocessing
from preprocessing import prepare_data, tokenize, prepare_dataset
import torch
import pandas as pd
import tqdm
import csv
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
#from pytorch_transformers import AdamW, WarmupLinearSchedule, T5Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm, trange, tqdm_notebook
from sklearn.metrics import matthews_corrcoef, f1_score


# STEP 0: GET THE DATA:

train_df = pd.read_json("./train.jsonl", lines=True)
test_df = pd.read_json("./test.jsonl", lines=True)
val_df = pd.read_json("./validation.jsonl", lines=True)

train_df = preprocessing.prepare_data(train_df)
val_df = preprocessing.prepare_data(val_df)
test_df = preprocessing.prepare_data(test_df)


tokenizer = T5Tokenizer.from_pretrained('t5-small')

#Tensor datasests:
train_dataset = prepare_dataset(train_df, tokenizer)
val_dataset = prepare_dataset(val_df, tokenizer)
test_dataset = prepare_dataset(test_df, tokenizer)


# STEP 1: INSTANTIATE MODEL:
model = T5ForConditionalGeneration.from_pretrained('t5-small')
optimizer = AdamW(model.parameters(),
                    lr = 1e-5
                  )

epochs = 5





