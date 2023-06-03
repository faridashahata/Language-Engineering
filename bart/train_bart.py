from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, RandomSampler
import torch
import os
import jsonlines
from torch.optim import lr_scheduler, AdamW
from config import *


# Hyperparameters
batch_size = BATCH_SIZE
learning_rate = LEARNING_RATE
num_epochs = EPOCHS

# Define dataset and dataloader
class BartDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = []

        with open(data_path, 'r') as file:
            reader = jsonlines.Reader(file)
            prosessed_items = 0
            for item in reader:
                if len(item["document"]) == 0 or len(item["summary"]) == 0:
                    continue
                if len(item["document"].split()) > (THRESHOLD * 1.5) or len(item["summary"].split()) > THRESHOLD:
                    continue

                # Preprocess your data here
                prosessed_items += 1
                processed_item = {
                    "input_text": item["document"],
                    "output_text": item["summary"]
                }
                self.data.append(processed_item)
        print("prosessed_items: ", prosessed_items)

    def __getitem__(self, index):
        item = self.data[index]
        inputs = tokenizer.batch_encode_plus([item["input_text"]], return_tensors='pt', padding='max_length', truncation=True)
        labels = tokenizer.batch_encode_plus([item["output_text"]], return_tensors='pt', padding='max_length', truncation=True)
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }

    def __len__(self):
        return len(self.data)

# Load tokenizer
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

train_data_path = TRAIN_DATA_PATH
test_data_path = TEST_DATA_PATH
val_data_path = VAL_DATA_PATH

print("Preprocessing data...", train_data_path)
train_dataset = BartDataset(train_data_path)
print("Preprocessing data...", val_data_path)
val_dataset = BartDataset(val_data_path)
print("Preprocessing data...", test_data_path)
test_dataset = BartDataset(test_data_path)

train_sampler = RandomSampler(train_dataset)
val_sampler = RandomSampler(val_dataset)

train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

# Load pre-trained BART model
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.pad_token_id = tokenizer.pad_token_id

model

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} of {num_epochs}")
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)

    # Validation loop
    model.eval()
    total_val_loss = 0

    for batch in tqdm(val_dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss = outputs.loss
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)

    print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # Adjust the learning rate
    scheduler.step()

# Save the fine-tuned model
output_dir = ".."
model_dir = MODEL_DIR
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
torch.save(model.state_dict(), os.path.join(model_dir, MODEL_FILE_NAME))
torch.save(optimizer.state_dict(), os.path.join(model_dir, OPTIMIZER_FILE_NAME))
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print("Fine-tuned model saved.")
