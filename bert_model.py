import pandas as pd
import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler, BertModel

class BERTSentimentClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=2):
        super(BERTSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Pooled output from BERT
        logits = self.classifier(pooled_output)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTSentimentClassifier(num_labels=2).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class ReviewDataset(Dataset):
    def __init__(self, reviews, sentiments, tokenizer, max_length=128):
        self.reviews = reviews
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        sentiment = self.sentiments[idx]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }


ruta = os.path.dirname(os.path.abspath(__file__))
ruta = os.path.join(ruta, "data/IMDB Dataset.csv")

df = pd.read_csv(ruta)

df = df.sample(n=10000, random_state=42).reset_index(drop=True)

label_map = {"negative": 0, "positive": 1}
df["sentiment"] = df["sentiment"].map(label_map)

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index).sample(frac=0.5, random_state=42)
test_df = df.drop(train_df.index).drop(val_df.index)

train_dataset = ReviewDataset(
    reviews=train_df.review.to_numpy(),
    sentiments=train_df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_length=128
)

val_dataset = ReviewDataset(
    reviews=val_df.review.to_numpy(),
    sentiments=val_df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_length=128
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()


def train_model(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_acc = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        acc = (predicted == labels).sum().item() / labels.size(0)
        total_acc += acc

        pbar.set_description(f"Training | Loss: {loss.item():.4f} | Acc: {acc:.2f}")
    
    return total_loss / len(loader), total_acc / len(loader)

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    
    pbar = tqdm(loader, desc="Evaluating")
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, dim=1)
            acc = (predicted == labels).sum().item() / labels.size(0)
            total_acc += acc

            pbar.set_description(f"Evaluating | Loss: {loss.item():.4f} | Acc: {acc:.2f}")
    
    return total_loss / len(loader), total_acc / len(loader)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

N_EPOCHS = 5

for epoch in tqdm(range(N_EPOCHS), desc="Epochs"):
    print(f'\nEpoch {epoch + 1}/{N_EPOCHS}')
    
    # Train model
    train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
    
    # Evaluate model
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
    
    # Print metrics
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc * 100:.2f}%')