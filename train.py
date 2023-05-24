import pandas as pd
import torch
import tqdm
import time
import numpy as np
from torch.utils.data import (DataLoader)
# from pytorch_transformers import AdamW, WarmupLinearSchedule, T5Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from transformers import get_linear_schedule_with_warmup

import preprocessing
from preprocessing import prepare_dataset, MODEL_NAME, BATCH_SIZE

# STEP 0: GET THE DATA:

train_df = pd.read_json("./data/train.jsonl", lines=True)
test_df = pd.read_json("./data/test.jsonl", lines=True)
val_df = pd.read_json("./data/validation.jsonl", lines=True)

# train_df = preprocessing.prepare_data(train_df)
# val_df = preprocessing.prepare_data(val_df)
# test_df = preprocessing.prepare_data(test_df)


device = torch.device("mps")
EPOCHS = 5
global THRESHOLD
THRESHOLD = 200

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

#Tensor datasets:

train_dataset = prepare_dataset(train_df, tokenizer, THRESHOLD)
val_dataset = prepare_dataset(val_df, tokenizer, THRESHOLD)
test_dataset = prepare_dataset(test_df, tokenizer, THRESHOLD)

print("Train data size: ", len(train_dataset))


# STEP 1: INSTANTIATE MODEL:
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
optimizer = AdamW(model.parameters(),
                   lr = 3e-5
                  #lr = 5e-4
                  )



dataloader = DataLoader(dataset=train_dataset,
                                shuffle=True,
                                batch_size=BATCH_SIZE)


total_steps = len(dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps
                                            )

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

def train(model, batch_size, optimizer, epochs, scheduler):

    train_stats = []
    val_stats = []
    for epoch in range(epochs):

        # Set total loss to zero for each epoch:
        total_loss = 0

        # Put model in train mode:
        model.train()

        # Create the training dataloader:
        train_dataloader = DataLoader(dataset=train_dataset,
                                shuffle=True,
                                batch_size=batch_size)

        for step, batch in enumerate(train_dataloader):

            # Progress update:
            if step % 10 == 0:
                print(f"Batch {step} of a total {len(train_dataloader)}")

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

        avg_loss = total_loss/len(train_dataloader)

        train_stats.append({ 'Training Loss': avg_loss})

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
            # with autocast():
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

        if val_stats[epoch]['Validation Loss'] < best_val_loss:
            best_val_loss = val_stats[epoch]['Validation Loss']
            torch.save(model.state_dict(), f't5_model_{t0}.pt')

            model.save_pretrained(f'./model_save/t5_{t0}/')
            tokenizer.save_pretrained(f'./model_save/t5_{t0}/')

    return train_stats, val_stats

# Set batch size in the global var in preprocessing:
train(model, BATCH_SIZE, optimizer, EPOCHS, scheduler)


#EXP 0: lr = 1e-5, 5 epochs, t5 small: 16 batch_size:

# Epoch | train Loss
# 0 | 2.928162097334862

# Epoch | validation Loss
# 0 | 1.9831322878599167

# Epoch | train Loss
# 1 | 1.9147634766499202

# Epoch | validation Loss
# 1 | 1.8701450414955616

# Epoch | train Loss
# 2 | 1.8536313583453496
#
# Epoch | validation Loss
# 2 | 1.8421144299209118


#FIRST EXPERIMENT: OVERFIT: lr = 5e-4, 10 epochs, t5 small: 16 batch_size:

# 10 epochs:
# #
# Epoch | train Loss
# 0 | 1.7621653946240743

# Epoch | validation Loss
# 0 | 1.6970857009291649


# # Epoch | train Loss
# # 1 | 1.626004677216212



# Epoch | validation Loss
# 1 | 1.6728606931865215

#
# Epoch | train Loss
# 2 | 1.5664739483594894


# Epoch | validation Loss
# 2 | 1.665628295391798

# Epoch | train Loss
# 3 | 1.5200032677253088

# Epoch | validation Loss
# 3 | 1.6614952087402344

# Epoch | train Loss
# 4 | 1.4804342768589656

# Epoch | validation Loss
# 4 | 1.665056113153696

#
# Epoch | train Loss
# 5 | 1.4472915301720302

# Epoch | validation Loss
# 5 | 1.6688082218170166

# Epoch | train Loss
# 6 | 1.4198416829109193

#
# Epoch | validation Loss
# 6 | 1.6688379794359207

# Epoch | train Loss
# 7 | 1.397624674042066
#
# Epoch | validation Loss
# 7 | 1.6684650368988514
#

# Epoch | train Loss
# 8 | 1.3807224975029628
#
# Epoch | validation Loss
# 8 | 1.6724535338580608

# Epoch | train Loss
# 9 | 1.3687105401357016
#
# Epoch | validation Loss
# 9 | 1.6760194823145866

# ROUGE SCORES: first run:
# {'rouge1_fmeasure': tensor(0.2521),
#  'rouge1_precision': tensor(0.2698),
#  'rouge1_recall': tensor(0.2438),
#  'rouge2_fmeasure': tensor(0.0282),
#  'rouge2_precision': tensor(0.0304),
#  'rouge2_recall': tensor(0.0271),
#  'rougeL_fmeasure': tensor(0.1347),
#  'rougeL_precision': tensor(0.1446),
#  'rougeL_recall': tensor(0.1299),
#  'rougeLsum_fmeasure': tensor(0.1948),
#  'rougeLsum_precision': tensor(0.2091),
#  'rougeLsum_recall': tensor(0.1879)}


# SECOND EXPERIMENT: 5 epochs,  lr = 3e-5, t5 small, 64 batch_size: (issue: dataloader out of train loop had batck size 16 instead of 64)
#
# Epoch | train Loss
# 0 | 2.940731824239095
#
#
# Epoch | validation Loss
# 0 | 1.8101017475128174

# Epoch | train Loss
# 1 | 1.7195128846168517
#
# Epoch | validation Loss
# 1 | 1.673126995563507


# Epoch | train Loss
# 2 | 1.6483326522509256
#
# Epoch | validation Loss
# 2 | 1.6299879252910614

# Epoch | train Loss
# 3 | 1.6174597962697348
#
# Epoch | validation Loss
# 3 | 1.6070178896188736

# Epoch | train Loss
# 4 | 1.5962289261817932
#
#
# Epoch | validation Loss
# 4 | 1.5885822623968124

#[{'Test Loss': 1.3741268600736345}]
# {'rouge1_fmeasure': tensor(0.2785),
#  'rouge1_precision': tensor(0.3328),
#  'rouge1_recall': tensor(0.2473),
#  'rouge2_fmeasure': tensor(0.0424),
#  'rouge2_precision': tensor(0.0508),
#  'rouge2_recall': tensor(0.0376),
#  'rougeL_fmeasure': tensor(0.1629),
#  'rougeL_precision': tensor(0.1953),
#  'rougeL_recall': tensor(0.1444),
#  'rougeLsum_fmeasure': tensor(0.2319),
#  'rougeLsum_precision': tensor(0.2777),
#  'rougeLsum_recall': tensor(0.2055)}
