import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClauseDataset(Dataset):
    def __init__(self, clauses, tokenizer, max_length=512):
        self.clauses = clauses
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.clauses)
    
    def __getitem__(self, idx):
        clause = self.clauses[idx]
        encoding = self.tokenizer(
            clause["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": clause["label"]
        }

class ClauseModelTrainer:
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
    def prepare_dataset(self, standard_clauses_file: str = "data/raw/standard_clauses.json"):
        """Prepare dataset from standard clauses and their variants."""
        with open(standard_clauses_file, 'r') as f:
            standard_clauses = json.load(f)
        
        # Create positive pairs (standard clause with its variants)
        positive_pairs = []
        for clause in standard_clauses:
            positive_pairs.append({
                "text": clause["text"],
                "label": 1
            })
            for variant in clause.get("variants", []):
                positive_pairs.append({
                    "text": variant,
                    "label": 1
                })
        
        # Create negative pairs (different standard clauses)
        negative_pairs = []
        for i, clause1 in enumerate(standard_clauses):
            for clause2 in standard_clauses[i+1:]:
                negative_pairs.append({
                    "text": clause1["text"],
                    "label": 0
                })
                negative_pairs.append({
                    "text": clause2["text"],
                    "label": 0
                })
        
        # Combine and split dataset
        all_pairs = positive_pairs + negative_pairs
        train_data, val_data = train_test_split(all_pairs, test_size=0.2, random_state=42)
        
        # Create datasets
        train_dataset = ClauseDataset(train_data, self.tokenizer)
        val_dataset = ClauseDataset(val_data, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            # Calculate loss (you might want to add a classification head)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs.last_hidden_state.mean(dim=1),
                labels.float()
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                # Calculate loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs.last_hidden_state.mean(dim=1),
                    labels.float()
                )
                
                # Calculate accuracy
                predictions = (outputs.last_hidden_state.mean(dim=1) > 0).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                total_loss += loss.item()
        
        return {
            "loss": total_loss / len(val_loader),
            "accuracy": correct / total
        }
    
    def train(self, train_dataset, val_dataset, num_epochs=3, batch_size=16):
        """Train the model for multiple epochs."""
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Training loss: {train_loss:.4f}")
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            logger.info(f"Validation loss: {val_metrics['loss']:.4f}")
            logger.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model("best_model")
    
    def save_model(self, output_dir: str):
        """Save the model and tokenizer."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")

def main():
    # Initialize trainer
    trainer = ClauseModelTrainer()
    
    # Prepare dataset
    train_dataset, val_dataset = trainer.prepare_dataset("data/raw/standard_clauses.json")
    
    # Train model
    trainer.train(train_dataset, val_dataset)
    
    # Save final model
    trainer.save_model("models/trained")

if __name__ == "__main__":
    main() 