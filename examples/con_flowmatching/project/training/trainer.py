import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNFTrainer:
    def __init__(self, model, train_loader, test_loader, device, 
                 lr=2e-4, weight_decay=1e-4, ema_decay=0.999, alpha=0.1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.alpha = alpha
        self.ema_decay = ema_decay

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        self.scaler = GradScaler()

        # EMA model
        self.ema_model = copy.deepcopy(model)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, x_1 in enumerate(pbar):
            x_1 = x_1.to(self.device)
            batch_size = x_1.shape[0]

            # Sample noise
            x_0 = torch.randn_like(x_1)

            # Sample time uniformly
            t = torch.rand(batch_size, device=self.device)

            with autocast():
                # Forward pass
                f_t, f_next, v_t, v_next = self.model(x_0, x_1, t)

                # Compute losses
                f_loss = torch.mean((f_t - f_next).pow(2))
                v_loss = torch.mean((v_t - v_next).pow(2))
                loss = f_loss + self.alpha * v_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update EMA model
            with torch.no_grad():
                for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                    ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        logger.info(f'Epoch {epoch} Average Loss: {avg_loss:.4f}')
        return avg_loss

    def evaluate(self):
        self.ema_model.eval()
        total_loss = 0

        with torch.no_grad():
            for x_1 in self.test_loader:
                x_1 = x_1.to(self.device)
                batch_size = x_1.shape[0]

                x_0 = torch.randn_like(x_1)
                t = torch.rand(batch_size, device=self.device)

                f_t, f_next, v_t, v_next = self.ema_model(x_0, x_1, t)
                f_loss = torch.mean((f_t - f_next).pow(2))
                v_loss = torch.mean((v_t - v_next).pow(2))
                loss = f_loss + self.alpha * v_loss

                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_loader)
        logger.info(f'Test Loss: {avg_loss:.4f}')
        return avg_loss