import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from t5.preprocessing import prepare_dataset
from torchmetrics.text.rouge import ROUGEScore
from pprint import pprint
from config import *
import datetime

train_df = pd.read_json(TRAIN_DATA_PATH, lines=True)
test_df = pd.read_json(TEST_DATA_PATH, lines=True)
val_df = pd.read_json(VAL_DATA_PATH, lines=True)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Tensor datasets:
train_dataset = prepare_dataset(train_df, tokenizer, THRESHOLD)
val_dataset = prepare_dataset(val_df, tokenizer, THRESHOLD)
test_dataset = prepare_dataset(test_df, tokenizer, THRESHOLD)

print("Train data size:", len(train_dataset))
print("Val data size:", len(val_dataset))
print("Test data size:", len(test_dataset))

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
        doc = [tokenizer.decode(d, skip_special_tokens=True, clean_up_tokenization_spaces=True) for d in doc_input_ids]

        print("doc:", doc)
        print("\n")
        print("preds:", preds)
        print("\n")
        print("target:", target)
        print("\n")

        predictions.extend(preds)
        actual_summaries.extend(target)
        docs.extend(doc)

    avg_test_loss = total_test_loss / len(dataloader)
    # Record all statistics from this epoch.
    test_stats.append({
        'Test Loss': avg_test_loss,
    })

    # temp_data
    test_df = pd.DataFrame({'document': docs, 'predicted': predictions, 'actual': actual_summaries})
    # test_df = test_df.append(temp_data)

    return test_stats, test_df


# LOAD MODEL:
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.load_state_dict(torch.load(MODEL_TO_TEST))

test_stats, test_df = test(model, dataloader)
print("test stats:", test_stats)
print("test_df size:", test_df.size)
print("test loop ran")

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
test_df.to_csv(f'test_df_t5-base_{timestamp}.csv')

# ROUGE METRICS:
print("rouge metrics loaded")
rouge = ROUGEScore()
pprint(rouge(test_df['predicted'].to_list(), test_df['actual'].to_list()))
