# BERT-Sentiment-Analysis-Fine-Tuning-for-Movie-Reviews

A Deep Learning project that implements Transfer Learning using the BERT (Bidirectional Encoder Representations from Transformers) architecture. This model is fine-tuned on the IMDB Dataset to classify movie reviews as positive or negative with high accuracy.

## Key Features

**Custom Architecture**: Implements a custom nn.Module wrapper around bert-base-uncased, adding a linear classification layer on top of the pooled output.

**Fine-Tuning**: Freezes early layers (implicit in BERT base usage) and trains the specific classification head tailored to the IMDB domain.

**Data Processing**: Efficient tokenization and batching using PyTorch DataLoader and Hugging Face BertTokenizer.

**Performance Tracking**: Real-time training monitoring with tqdm for loss and accuracy visualization.

## Tech Stack

**Core**: Python, PyTorch.

**NLP**: Hugging Face Transformers (bert-base-uncased).

**Data Handling**: Pandas, NumPy.

**Dataset**: IMDB Large Movie Review Dataset.

## Project Structure
Bash

    ├── data/
    │   └── IMDB Dataset.csv  # The dataset file
    ├── bert_model.py         # Main training and evaluation script
    ├── requirements.txt      # Dependencies
    └── README.md             # Documentation

## Model Architecture

The code does not simply use a pre-built classifier but defines a custom neural network structure:

### Python

    class BERTSentimentClassifier(nn.Module):
        def __init__(self, bert_model_name='bert-base-uncased', num_labels=2):
            super(BERTSentimentClassifier, self).__init__()
            self.bert = BertModel.from_pretrained(bert_model_name)
            # Custom Classification Head
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs[1] 
            logits = self.classifier(pooled_output)
            return logits

## How to Run

Clone the repository:

    git clone https://github.com/tu-usuario/BERT-Sentiment-Analysis.git
    cd BERT-Sentiment-Analysis

Install dependencies:

    pip install torch transformers pandas tqdm

Run the training script:

    python bert_model.py



Author

Joan Guerrero - Software Developer & AI Specialist [LinkedIn](https://www.linkedin.com/in/jg-chez/)