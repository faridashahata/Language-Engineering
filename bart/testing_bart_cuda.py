import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer
from t5.preprocessing import prepare_dataset
from torchmetrics.text.rouge import ROUGEScore
from config import *
from pprint import pprint


train_df = pd.read_json(TRAIN_DATA_PATH, lines=True)
test_df = pd.read_json(TEST_DATA_PATH, lines=True)
val_df = pd.read_json(VAL_DATA_PATH, lines=True)

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

# Tensor datasets:
train_dataset = prepare_dataset(train_df, tokenizer, THRESHOLD)
val_dataset = prepare_dataset(val_df, tokenizer, THRESHOLD)
test_dataset = prepare_dataset(test_df, tokenizer, THRESHOLD)

print("Train data size: ", len(train_dataset))
print("Val data size: ", len(val_dataset))
print("Test data size: ", len(test_dataset))

# Set up the testing dataloader:
dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=2)

test_stats = []


# Testing loop:
def test(model, dataloader):
    # model in eval mode:
    model.eval()

    total_test_loss = 0
    predictions = []
    actual_summaries = []
    docs = []

    for step, batch in enumerate(dataloader):
        # Progress update:
        if step % 10 == 0:
            print(f"Batch {step} of a total {len(dataloader)}")

        # Unpack training batch from dataloader:
        doc_input_ids, doc_attention_masks = batch[0], batch[1]
        summary_input_ids, summary_attention_masks = batch[2], batch[3]

        # Forward pass:
        with torch.no_grad():
            outputs = model(input_ids=doc_input_ids, attention_mask=doc_attention_masks,
                            labels=summary_input_ids, decoder_attention_mask=summary_attention_masks)
            loss, pred_scores = outputs.loss, outputs.logits

        # Sum loss over all batches:
        total_test_loss += loss.item()

        generated_ids = model.generate(input_ids=doc_input_ids, attention_mask=doc_attention_masks,
                                       do_sample=True, temperature=0.8, top_p=0.9, max_length=200, min_length=50,
                                       repetition_penalty=2.5)

        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in summary_input_ids]
        doc = [tokenizer.decode(d, skip_special_tokens=True, clean_up_tokenization_spaces=True) for d in doc_input_ids]

        predictions.extend(preds)
        actual_summaries.extend(target)
        docs.extend(doc)

    avg_test_loss = total_test_loss / len(dataloader)
    # Record all statistics from this epoch.
    test_stats.append({'Test Loss': avg_test_loss})

    global test_df
    test_df = pd.DataFrame({'document': docs, 'predicted': predictions, 'actual': actual_summaries})

    return test_stats


# LOAD MODEL:
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
#model.to('cuda')  # if using GPU

test_stats = test(model, dataloader)
print("test stats: ", test_stats)
print("test_df size", test_df.size)
print("test loop ran")

# Save DataFrame to CSV with timestamp in the file name
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
test_df.to_csv(f'test_df_bart-base_{timestamp}.csv')

# ROUGE METRICS:
print("rouge metrics loaded")
preds = test_df.predicted.to_list()
target = test_df.actual.to_list()
rouge = ROUGEScore()
pprint(rouge(preds, target))
