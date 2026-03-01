import os
import torch
import logging
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model, dataloaders, criterion, optimizer, device, scheduler=None, patience=7):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        
        # 저장 폴더 경로 설정 및 생성
        self.save_dir = "models/weights"
        os.makedirs(self.save_dir, exist_ok=True)

    def fit(self, num_epochs):
        best_acc = 0.0

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            train_loss, train_acc = self._run_epoch('train')
            val_loss, val_acc = self._run_epoch('val')

            logger.info(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })

            if self.scheduler:
                self.scheduler.step(val_loss)

            # Early Stopping 및 모델 저장 로직
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0
                
                # 경로 지정
                save_path = os.path.join(self.save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                
                logger.info(f"Best model saved to {save_path} (Val Loss improved).")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

            if val_acc > best_acc:
                best_acc = val_acc

        logger.info(f"Training completed. Best Val Acc: {best_acc:.4f}")

    def _run_epoch(self, phase):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss, corrects = 0.0, 0
        
        with torch.set_grad_enabled(phase == 'train'):
            for inputs, labels in tqdm(self.dataloaders[phase], desc=phase.capitalize()):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if phase == 'train':
                    self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
        epoch_acc = corrects.double() / len(self.dataloaders[phase].dataset)
        return epoch_loss, epoch_acc