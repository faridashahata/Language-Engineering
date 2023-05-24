import os
import time
import numpy as np
import pandas as pd
import torch
import shutil


from torch.utils.data import (DataLoader)
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from transformers import get_linear_schedule_with_warmup
from preprocessing import prepare_dataset, MODEL_NAME, BATCH_SIZE

# STEP 0: GET THE DATA:
train_df = pd.read_json("./data/train.jsonl", lines=True)
test_df = pd.read_json("./data/test.jsonl", lines=True)
val_df = pd.read_json("./data/validation.jsonl", lines=True)

# Set global variables:
EPOCHS = 5
THRESHOLD: int = 200

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Tensor datasets:
train_dataset = prepare_dataset(train_df, tokenizer, THRESHOLD)
val_dataset = prepare_dataset(val_df, tokenizer, THRESHOLD)
test_dataset = prepare_dataset(test_df, tokenizer, THRESHOLD)

print("Train data size: ", len(train_dataset))

# STEP 1: INSTANTIATE MODEL:
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
optimizer = AdamW(model.parameters(), lr=3e-5) # lr = 5e-4

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

def train(model, batch_size, optimizer, epochs, scheduler, checkpoint_interval, resume_from_checkpoint=None):
    train_stats = []
    val_stats = []
    if resume_from_checkpoint:
        load_checkpoint(model, optimizer, tokenizer, resume_from_checkpoint)
        start_epoch = int(resume_from_checkpoint.split('_')[-1])
    else:
        start_epoch = 0
    for epoch in range(start_epoch, epochs):

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

            # Clear previously calculated gradients:
            optimizer.zero_grad()

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
        if early_stopper.early_stop(avg_val_loss):
            break

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
        if val_stats[epoch]['Validation Loss'] < best_val_loss:
            best_val_loss = val_stats[epoch]['Validation Loss']
            torch.save(model.state_dict(), f't5_model_{t0}.pt')

            model.save_pretrained(f'./model_save/t5_{t0}/')
            tokenizer.save_pretrained(f'./model_save/t5_{t0}/')

    return train_stats, val_stats

def save_checkpoint(model, optimizer, tokenizer, epoch, val_loss):
    # Create a directory to save the checkpoint
    checkpoint_dir = f'./checkpoints/epoch_{epoch}'
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

def load_checkpoint(model, optimizer, tokenizer, checkpoint_dir):
    # Load model state dict
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'model.pt')))
    # model.to('mds')
    # Load optimizer state dict
    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'optimizer.pt')))
    # Load tokenizer
    tokenizer.from_pretrained(checkpoint_dir)

# Set batch size in the global var in preprocessing:
train(model, BATCH_SIZE, optimizer, EPOCHS, scheduler,  checkpoint_interval=1)

# Set the path to the checkpoint directory EXAMPLE
# checkpoint_dir = './checkpoints/epoch_3'
# Call the train function with the checkpoint
#train_stats, val_stats = train(model, BATCH_SIZE, optimizer, EPOCHS, scheduler, checkpoint_interval=1, resume_from_checkpoint=checkpoint_dir)

