import pandas as pd
import torch
from torch.utils.data import (DataLoader)
# from pytorch_transformers import AdamW, WarmupLinearSchedule, T5Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from transformers import get_linear_schedule_with_warmup

import preprocessing
from preprocessing import prepare_dataset

# STEP 0: GET THE DATA:

train_df = pd.read_json("./data/train.jsonl", lines=True)
test_df = pd.read_json("./data/test.jsonl", lines=True)
val_df = pd.read_json("./data/validation.jsonl", lines=True)

# train_df = preprocessing.prepare_data(train_df)
# val_df = preprocessing.prepare_data(val_df)
# test_df = preprocessing.prepare_data(test_df)


tokenizer = T5Tokenizer.from_pretrained('t5-small')

#Tensor datasets:
train_dataset = prepare_dataset(train_df, tokenizer)
val_dataset = prepare_dataset(val_df, tokenizer)
test_dataset = prepare_dataset(test_df, tokenizer)


# STEP 1: INSTANTIATE MODEL:
model = T5ForConditionalGeneration.from_pretrained('t5-small')
optimizer = AdamW(model.parameters(),
                    lr = 1e-5
                  )

epochs = 5
total_steps = len(train_df) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps
                                            )


def train(model, batch_size, optimizer, epochs, scheduler):

    train_stats = []
    val_stats = []
    for epoch in range(epochs):

        # Set total loss to zero for each epoch:
        total_loss = 0

        # Put model in train mode:
        model.train()

        # Create the training dataloader:
        dataloader = DataLoader(dataset=train_dataset,
                                shuffle=True,
                                batch_size=batch_size)

        for step, batch in enumerate(dataloader):

            # Progress update:
            if step % 10 == 0:
                print(f"Batch {step} of a total {len(dataloader)}")

            # Unpack training batch from dataloader:
            doc_input_ids, doc_attention_masks = batch[0], batch[1]
            summary_input_ids, summary_attention_masks = batch[2], batch[3]


            # Clear previously calculated gradients:
            optimizer.zero_grad()

            # Forward pass:
            #with autocast():

            outputs = model(input_ids=doc_input_ids,
                            attention_mask=doc_attention_masks,
                            labels=summary_input_ids,
                            decoder_attention_mask=summary_attention_masks)

            loss, pred_scores = outputs[:2]

            # Sum loss over all batches:
            total_loss += loss.item()

            # Backward pass:
            loss.backward()

            optimizer.step()

            # Update scheduler:
            scheduler.step()

        avg_loss = total_loss/len(dataloader)

        train_stats.append({ 'Training Loss': avg_loss})

        print("Summary Results: ")
        print("Epoch | train Loss")
        print(f"{epoch} | {avg_loss}")

        # Create the validation dataloader:
        dataloader = DataLoader(dataset=val_dataset,
                                shuffle=False,
                                batch_size=batch_size)

        total_val_loss = 0
        for step, batch in enumerate(dataloader):

            # Progress update:
            if step % 10 == 0:
                print(f"Batch {step} of a total {len(dataloader)}")

            # Unpack training batch from dataloader:
            doc_input_ids, doc_attention_masks = batch[0], batch[1]
            summary_input_ids, summary_attention_masks = batch[2], batch[3]

            # Clear previously calculated gradients:
            optimizer.zero_grad()

            # Forward pass:
            # with autocast():
            with torch.no_grad():
                outputs = model(input_ids=doc_input_ids,
                                attention_mask=doc_attention_masks,
                                labels=summary_input_ids,
                                decoder_attention_mask=summary_attention_masks)

                loss, pred_scores = outputs[:2]

                # Sum loss over all batches:
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(dataloader)
        val_stats.append({'Validation Loss': avg_val_loss})

        print("Summary Results: ")
        print("Epoch | validation Loss")
        print(f"{epoch} | {avg_val_loss}")

        best_val_loss = float('inf')

        if val_stats[epoch]['Validation Loss'] < best_val_loss:
            best_val_loss = val_stats[epoch]['Validation Loss']
            torch.save(model.state_dict(), 't5_model.pt')

            model.save_pretrained('./model_save/t5/')
            tokenizer.save_pretrained('./model_save/t5/')

    return train_stats, val_stats


train(model, 16, optimizer, epochs, scheduler)










