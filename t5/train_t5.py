import json
import os
import time
import numpy as np
import pandas as pd
import torch
import sentencepiece
from torch.optim import AdamW
import time
from torch.utils.data import (DataLoader)
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from t5.preprocessing import prepare_dataset
from config import *

# STEP 0: GET THE DATA:
train_df = pd.read_json(TRAIN_DATA_PATH, lines=True)
test_df = pd.read_json(TEST_DATA_PATH, lines=True)
val_df = pd.read_json(VAL_DATA_PATH, lines=True)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME,  model_max_length=512, truncation=True, padding=True)

# Tensor datasets:

print("Preparing train dataset...")
start_time = time.time()
train_dataset = prepare_dataset(train_df, tokenizer, THRESHOLD)
end_time = time.time()
print("Train dataset preparation time:", end_time - start_time, "seconds")

print("Preparing val dataset...")
start_time = time.time()
val_dataset = prepare_dataset(val_df, tokenizer, THRESHOLD)
end_time = time.time()
print("Val dataset preparation time:", end_time - start_time, "seconds")

print("Preparing test dataset...")
start_time = time.time()
test_dataset = prepare_dataset(test_df, tokenizer, THRESHOLD)
end_time = time.time()
print("Test dataset preparation time:", end_time - start_time, "seconds")

print("Train data size: ", len(train_dataset))
print("Val data size: ", len(val_dataset))
print("Test data size: ", len(test_dataset))

# STEP 1: INSTANTIATE MODEL:
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
optimizer = AdamW(model.parameters(), lr=1e-5) # lr = 5e-4, 3e-4

dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE)

total_steps = len(dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# IMPLEMENT EARLY STOPPING:
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

early_stopper = EarlyStopper(patience=3, min_delta=10)

def train(model, batch_size, optimizer, epochs, scheduler, checkpoint_interval, resume_from_checkpoint=None, model_name_pt=None):
    train_stats = []
    val_stats = []
    if resume_from_checkpoint:
        load_checkpoint(model, optimizer, tokenizer, resume_from_checkpoint, model_name_pt)
        start_epoch = int(resume_from_checkpoint.split('_')[-1])
        print("start epoch", start_epoch)
    else:
        start_epoch = 0
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        # Set total loss to zero for each epoch:
        total_loss = 0

        # Put model in train mode:
        model.train()

        # Create the training dataloader:
        train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

        for step, batch in enumerate(train_dataloader):
            # Progress update:
            if step % 10 == 0:
                print(f"Batch {step} of a total {len(train_dataloader)}")
            # Unpack training batch from dataloader:
            doc_input_ids, doc_attention_masks = batch[0], batch[1]
            summary_input_ids, summary_attention_masks = batch[2], batch[3]

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

        avg_loss = total_loss / len(train_dataloader)

        train_stats.append({'Training Loss': avg_loss})

        print("Summary Results: ")
        print("Epoch | train Loss")
        print(f"{epoch} | {avg_loss}")

        # Create the validation dataloader:
        val_dataloader = DataLoader(dataset=val_dataset,
                                    shuffle=False,
                                    batch_size=batch_size)

        total_val_loss = 0
        for step, batch in enumerate(val_dataloader):

            # Progress update:
            if step % 10 == 0:
                print(f"Batch {step} of a total {len(val_dataloader)}")

            # Unpack training batch from dataloader:
            doc_input_ids, doc_attention_masks = batch[0], batch[1]
            summary_input_ids, summary_attention_masks = batch[2], batch[3]

            # Clear previously calculated gradients:
            optimizer.zero_grad()

            # Forward pass:
            with torch.no_grad():
                outputs = model(input_ids=doc_input_ids,
                                attention_mask=doc_attention_masks,
                                labels=summary_input_ids,
                                decoder_attention_mask=summary_attention_masks)

                loss, pred_scores = outputs[:2]

                # Sum loss over all batches:
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_stats.append({'Validation Loss': avg_val_loss})

        # Add Early Stopping:

        print("Summary Results: ")
        print("Epoch | validation Loss")
        print(f"{epoch} | {avg_val_loss}")

        best_val_loss = float('inf')

        # Exact timestamp:
        t0 = time.ctime().split()[3]

        # Save a checkpoint at regular intervals
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, tokenizer, epoch, avg_val_loss)

        # Save the model if the validation loss is the best we've seen so far
        if epoch % 5 == 0:
            if val_stats[-1]['Validation Loss'] < best_val_loss:
                best_val_loss = val_stats[-1]['Validation Loss']
                model_dir = f'./model_save_{MODEL_NAME}_{THRESHOLD}_{BATCH_SIZE}'
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(model.state_dict(), os.path.join(model_dir, f't5_model_{t0}.pt'))

                model.save_pretrained(f'./model_save_{MODEL_NAME}_{THRESHOLD}_{BATCH_SIZE}/t5_{t0}/')
                tokenizer.save_pretrained(f'./model_save_{MODEL_NAME}_{THRESHOLD}_{BATCH_SIZE}/t5_{t0}/')

        if early_stopper.early_stop(avg_val_loss):
            break
    return train_stats, val_stats

def save_checkpoint(model, optimizer, tokenizer, epoch, val_loss):
    # Create a directory to save the checkpoint
    checkpoint_dir = f'./checkpoints_{MODEL_NAME}_{THRESHOLD}_{BATCH_SIZE}/epoch_{epoch}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save model, optimizer, and tokenizer
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer.pt'))

    # Save additional information
    with open(os.path.join(checkpoint_dir, 'info.txt'), 'w') as f:
        f.write(f'Epoch: {epoch}\n')
        f.write(f'Validation Loss: {val_loss}\n')

def load_checkpoint(model, optimizer, tokenizer, checkpoint_dir, model_name_pt):
    # Load model state dict
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, model_name_pt)))
    #    model.load_state_dict(torch.load(os.path.join('./model_save_t5-base_250_4', model_name_pt)))

    # Load optimizer state dict
    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'optimizer.pt')))
    # Load tokenizer
    tokenizer.from_pretrained(checkpoint_dir)

# Set batch size in the global var in preprocessing:
train_stats, val_stats = train(model, BATCH_SIZE, optimizer, EPOCHS, scheduler,  checkpoint_interval=1)

# Set the path to the checkpoint directory EXAMPLE
# checkpoint_dir = './checkpoints/epoch_3'
# Call the train function with the checkpoint
#train_stats, val_stats = train(model, BATCH_SIZE, optimizer, EPOCHS, scheduler, checkpoint_interval=1, resume_from_checkpoint=checkpoint_dir)
# Save the train and validation stats
with open(f'checkpoints_{MODEL_NAME}_{THRESHOLD}_{BATCH_SIZE}/train_stats.json', 'w') as f:
    json.dump(train_stats, f)
with open(f'checkpoints_{MODEL_NAME}_{THRESHOLD}_{BATCH_SIZE}/val_stats.json', 'w') as f:
    json.dump(val_stats, f)
