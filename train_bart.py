from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, RandomSampler
import torch
import torchvision
import jsonlines

# Define dataset and dataloader
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = []

        with open(data_path, 'r') as file:
            reader = jsonlines.Reader(file)
            prosessed_items = 0
            for item in reader:
                if len(item["document"]) == 0 or len(item["summary"]) == 0:
                    continue
                if len(item["document"]) > 1000 or len(item["summary"]) > 500:
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


# Hyperparameters
batch_size = 16
learning_rate = 1e-3
num_epochs = 3

# Load tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

train_data_path = "./data/train.jsonl"
val_data_path = "./data/validation.jsonl"

train_dataset = MyDataset(train_data_path)
val_dataset = MyDataset(val_data_path)

train_sampler = RandomSampler(train_dataset)
val_sampler = RandomSampler(val_dataset)

train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

# Load pre-trained BART model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
model.config.pad_token_id = tokenizer.pad_token_id

# Well, this does not work on my mac
device = torch.device('opencl' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

model.to(device)

# Define optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)




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
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss = outputs.loss
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)

    print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # Adjust the learning rate
    scheduler.step()

# Save the fine-tuned model
output_dir = "."
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Fine-tuned model saved.")