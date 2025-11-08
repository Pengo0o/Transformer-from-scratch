import torch
import torch.nn as nn
from tqdm import tqdm
import os


class Trainer:
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler, device, num_epochs, save_dir='results', writer=None):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.writer = writer
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.global_step = 0

    def train_epoch(self):

        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            self.optimizer.zero_grad()
            output = self.model(src, tgt_input)
  
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            
            loss = self.criterion(output, tgt_output)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/GradientNorm', grad_norm.item(), self.global_step)
                if self.scheduler is not None:
                    self.writer.add_scalar('Train/LearningRate', 
                                         self.optimizer.param_groups[0]['lr'], 
                                         self.global_step)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.global_step += 1
            total_loss += loss.item()
            
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for src, tgt in tqdm(self.val_loader, desc="Validation"):
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                output = self.model(src, tgt_input)
                
                output = output.contiguous().view(-1, output.size(-1))
                tgt_output = tgt_output.contiguous().view(-1)
                
                loss = self.criterion(output, tgt_output)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)

    def train(self):

        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.writer.add_scalars('Loss/TrainVal', {
                    'train': train_loss,
                    'validation': val_loss
                }, epoch)
                
                if self.scheduler is not None:
                    self.writer.add_scalar('LearningRate', 
                                         self.optimizer.param_groups[0]['lr'], 
                                         epoch)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            if self.scheduler is not None:
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt', epoch, val_loss)
            

    def save_checkpoint(self, filename, epoch, loss):

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'global_step': self.global_step
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename):

        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.global_step = checkpoint.get('global_step', 0)
        
        return checkpoint['epoch'], checkpoint['loss']
