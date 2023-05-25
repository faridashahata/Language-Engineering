import pandas as pd
import torch
import tqdm
from torch.utils.data import (DataLoader)
# from pytorch_transformers import AdamW, WarmupLinearSchedule, T5Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from transformers import get_linear_schedule_with_warmup

import preprocessing
import nlp
from preprocessing import prepare_dataset

# STEP 0: GET THE DATA:

train_df = pd.read_json("./data/train.jsonl", lines=True)
test_df = pd.read_json("./data/test.jsonl", lines=True)
val_df = pd.read_json("./data/validation.jsonl", lines=True)

# train_df = preprocessing.prepare_data(train_df)
# val_df = preprocessing.prepare_data(val_df)
# test_df = preprocessing.prepare_data(test_df)


device = torch.device("mps")

tokenizer = T5Tokenizer.from_pretrained('t5-large')

#Tensor datasets:
train_dataset = prepare_dataset(train_df, tokenizer, 200)
val_dataset = prepare_dataset(val_df, tokenizer, 200)
test_dataset = prepare_dataset(test_df, tokenizer, 200)

print("Train data size: ", len(train_dataset))
print("Val data size: ", len(val_dataset))
print("Test data size: ", len(test_dataset))


# Set up the testing dataloader:
dataloader = DataLoader(dataset=test_dataset,
                                shuffle=False,
                                batch_size=64)


tokenizer = T5Tokenizer.from_pretrained('t5-large')

test_stats = []
# Testing loop:

def test(model, dataloader):
    # model in eval mode:
    model.eval()

    total_test_loss = 0
    predictions = []
    actual_summaries = []

    for step, batch in enumerate(dataloader):



        # Progress update:
        if step % 10 == 0:
            print(f"Batch {step} of a total {len(dataloader)}")

        # Unpack training batch from dataloader:
        doc_input_ids, doc_attention_masks = batch[0], batch[1]
        summary_input_ids, summary_attention_masks = batch[2], batch[3]

        # Forward pass:
        # with autocast():

        outputs = model(input_ids=doc_input_ids,
                        attention_mask=doc_attention_masks,
                        labels=summary_input_ids,
                        decoder_attention_mask=summary_attention_masks)

        loss, pred_scores = outputs[:2]

        # Sum loss over all batches:
        total_test_loss += loss.item()

        generated_ids = model.generate(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_masks,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            max_length=200,
            min_length=50,
            repetition_penalty=2.5

        )


        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in summary_input_ids]
        print("preds:    ", preds)
        print("\n")
        print("target:   ", target)
        print("\n")
        print("\n")
        predictions.extend(preds)
        actual_summaries.extend(target)

    avg_test_loss = total_test_loss/len(dataloader)
    # Record all statistics from this epoch.
    test_stats.append(
        {
            'Test Loss': avg_test_loss,
        }
    )
    global test_df
    #temp_data
    test_df = pd.DataFrame({'predicted': predictions, 'actual': actual_summaries})
    #test_df = test_df.append(temp_data)

    return test_stats


# LOAD MODEL:
model = T5ForConditionalGeneration.from_pretrained('t5-large')
model.load_state_dict(torch.load('t5_model_v2.pt'))

test_stats = test(model, dataloader)
print("test stats: ", test_stats)
print("test_df", test_df)
print("test_df size", test_df.size)
print("test loop ran")
# ROUGE METRICS:

# ROUGE
nlp_rouge = nlp.load_metric('rouge')

print("rouge metrics loaded")
import torchmetrics
from torchmetrics.text.rouge import ROUGEScore
preds = test_df.predicted.to_list()
target = test_df.actual.to_list()
rouge = ROUGEScore()
from pprint import pprint
pprint(rouge(preds, target))
#
# scores = nlp_rouge.compute(
#     test_df.predicted.to_list(), test_df.actual.to_list(),
#     rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
#     use_agregator=True, use_stemmer=False
# )
print("scores computed")

#
# metrics_df = pd.DataFrame({
#     'rouge1': [scores['rouge1'].mid.precision, scores['rouge1'].mid.recall, scores['rouge1'].mid.fmeasure],
#     'rouge2': [scores['rouge2'].mid.precision, scores['rouge2'].mid.recall, scores['rouge2'].mid.fmeasure],
#     'rougeL': [scores['rougeL'].mid.precision, scores['rougeL'].mid.recall, scores['rougeL'].mid.fmeasure]}, index=[ 'P', 'R', 'F'])
#
# print("metrics_df built")
#
# metrics_df.style.format({'rouge1': "{:.4f}", 'rouge2': "{:.4f}", 'rougeL': "{:.4f}"})
#
# print(metrics_df)



