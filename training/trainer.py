import torch
import torch.nn as nn
from config import config

class Trainer:
    def __init__(self, model):
        self.model = model.to(config.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )

    def train(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            self.optimizer.zero_grad()

            outputs = self.model(
                batch["w2v"].to(config.device),
                batch["mfcc"].to(config.device),
                batch["mel"].to(config.device),
                batch["pros"].to(config.device)
            )

            loss = self.criterion(
                outputs,
                batch["label"].to(config.device)
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        print(f"Training Loss: {total_loss:.4f}")
        return total_loss
