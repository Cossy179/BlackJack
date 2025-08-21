"""
Phase 2 Imitation Learning

Supervised classifier that maps observations to actions while respecting legal action masks.
Uses cross-entropy loss applied only to legal actions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path

from .data_generation import GameplayEpisode, GameplayStep
from .state_representation import get_feature_dimensions
from .device_utils import device_manager, to_device, get_device, is_cuda_available


class BlackjackDataset(Dataset):
    """PyTorch dataset for blackjack imitation learning"""
    
    def __init__(self, 
                 observations: np.ndarray,  # (N, 12)
                 actions: np.ndarray,       # (N,)
                 legal_masks: np.ndarray,   # (N, 5)
                 device: torch.device = None):  # Optional device override
        
        # Use device manager for consistent device handling
        target_device = device if device is not None else get_device()
        
        self.observations = torch.FloatTensor(observations).to(target_device)
        self.actions = torch.LongTensor(actions).to(target_device)
        self.legal_masks = torch.FloatTensor(legal_masks).to(target_device)
        
        assert len(self.observations) == len(self.actions) == len(self.legal_masks)
        print(f"📊 Dataset created on {target_device} with {len(self.observations)} samples")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return {
            'observation': self.observations[idx],
            'action': self.actions[idx],
            'legal_mask': self.legal_masks[idx]
        }


class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss that only applies to legal actions"""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, logits, targets, legal_masks):
        """
        Args:
            logits: (batch_size, num_actions) raw network outputs
            targets: (batch_size,) target action indices
            legal_masks: (batch_size, num_actions) boolean mask of legal actions
        """
        
        # Apply mask by setting illegal action logits to very negative values
        masked_logits = logits.clone()
        masked_logits[legal_masks == 0] = -1e9
        
        # Compute cross-entropy loss
        loss = nn.functional.cross_entropy(
            masked_logits, targets, reduction=self.reduction
        )
        
        return loss


class BlackjackPolicyNet(nn.Module):
    """
    Neural network for blackjack policy imitation.
    
    Architecture:
    - Input: 12-dimensional state features
    - Hidden: 3 layers with 256 units each, ReLU activation
    - Output: 5 action logits (STAND, HIT, DOUBLE, SPLIT, SURRENDER)
    """
    
    def __init__(self, 
                 input_size: int = 12,
                 hidden_size: int = 256,
                 num_actions: int = 5,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Action head
        self.action_head = nn.Linear(hidden_size, num_actions)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, observations):
        """
        Forward pass.
        
        Args:
            observations: (batch_size, input_size) state features
            
        Returns:
            logits: (batch_size, num_actions) action logits
        """
        features = self.feature_net(observations)
        logits = self.action_head(features)
        return logits
    
    def get_action_probabilities(self, observations, legal_masks):
        """
        Get action probabilities for given observations.
        
        Args:
            observations: (batch_size, input_size) state features
            legal_masks: (batch_size, num_actions) legal action masks
            
        Returns:
            probs: (batch_size, num_actions) action probabilities
        """
        with torch.no_grad():
            logits = self.forward(observations)
            
            # Apply legal action mask
            masked_logits = logits.clone()
            masked_logits[legal_masks == 0] = -1e9
            
            # Convert to probabilities
            probs = torch.softmax(masked_logits, dim=-1)
            
        return probs
    
    def predict_action(self, observation, legal_mask):
        """
        Predict single action for given observation.
        
        Args:
            observation: (input_size,) state features
            legal_mask: (num_actions,) legal action mask
            
        Returns:
            action: Predicted action index
        """
        with torch.no_grad():
            # Convert to tensors on correct device
            if not isinstance(observation, torch.Tensor):
                observation = device_manager.from_numpy(observation) if isinstance(observation, np.ndarray) else device_manager.tensor(observation)
            else:
                observation = to_device(observation)
                
            if not isinstance(legal_mask, torch.Tensor):
                legal_mask = device_manager.from_numpy(legal_mask) if isinstance(legal_mask, np.ndarray) else device_manager.tensor(legal_mask)
            else:
                legal_mask = to_device(legal_mask)
            
            # Add batch dimension
            obs_batch = observation.unsqueeze(0)
            mask_batch = legal_mask.unsqueeze(0)
            
            # Get probabilities
            probs = self.get_action_probabilities(obs_batch, mask_batch)
            
            # Select action with highest probability
            action = torch.argmax(probs, dim=-1).item()
            
        return action


class ImitationTrainer:
    """Trainer for blackjack policy imitation learning"""
    
    def __init__(self,
                 model: BlackjackPolicyNet,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 device: str = None):
        
        # Use centralized device manager
        if device is not None:
            device_manager.set_device(device)
        
        self.device = get_device()
        self.model = to_device(model)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = MaskedCrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        observations = batch['observation'].to(self.device)
        actions = batch['action'].to(self.device)
        legal_masks = batch['legal_mask'].to(self.device)
        
        # Forward pass
        logits = self.model(observations)
        loss = self.criterion(logits, actions, legal_masks)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy (only considering legal actions)
        with torch.no_grad():
            # Apply mask to logits
            masked_logits = logits.clone()
            masked_logits[legal_masks == 0] = -1e9
            
            # Get predictions
            predictions = torch.argmax(masked_logits, dim=-1)
            correct = (predictions == actions).float()
            accuracy = correct.mean()
        
        return loss.item(), accuracy.item()
    
    def validate_step(self, batch):
        """Single validation step"""
        self.model.eval()
        
        with torch.no_grad():
            observations = batch['observation'].to(self.device)
            actions = batch['action'].to(self.device)
            legal_masks = batch['legal_mask'].to(self.device)
            
            # Forward pass
            logits = self.model(observations)
            loss = self.criterion(logits, actions, legal_masks)
            
            # Calculate accuracy
            masked_logits = logits.clone()
            masked_logits[legal_masks == 0] = -1e9
            predictions = torch.argmax(masked_logits, dim=-1)
            correct = (predictions == actions).float()
            accuracy = correct.mean()
        
        return loss.item(), accuracy.item()
    
    def train_epoch(self, train_loader, val_loader=None):
        """Train for one epoch"""
        
        # Training phase
        train_losses = []
        train_accuracies = []
        
        for batch in train_loader:
            loss, accuracy = self.train_step(batch)
            train_losses.append(loss)
            train_accuracies.append(accuracy)
        
        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = np.mean(train_accuracies)
        
        self.train_losses.append(epoch_train_loss)
        self.train_accuracies.append(epoch_train_acc)
        
        # Validation phase
        epoch_val_loss = None
        epoch_val_acc = None
        
        if val_loader is not None:
            val_losses = []
            val_accuracies = []
            
            for batch in val_loader:
                loss, accuracy = self.validate_step(batch)
                val_losses.append(loss)
                val_accuracies.append(accuracy)
            
            epoch_val_loss = np.mean(val_losses)
            epoch_val_acc = np.mean(val_accuracies)
            
            self.val_losses.append(epoch_val_loss)
            self.val_accuracies.append(epoch_val_acc)
        
        return {
            'train_loss': epoch_train_loss,
            'train_accuracy': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_accuracy': epoch_val_acc
        }
    
    def train(self, 
              train_loader,
              num_epochs: int = 50,
              val_loader=None,
              print_every: int = 10,
              save_path: Optional[str] = None):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of training epochs
            val_loader: Optional validation data loader
            print_every: Print progress every N epochs
            save_path: Optional path to save best model
        """
        
        print(f"Training imitation model for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Train epoch
            metrics = self.train_epoch(train_loader, val_loader)
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}:")
                print(f"  Train: Loss={metrics['train_loss']:.4f}, Acc={metrics['train_accuracy']:.4f}")
                if metrics['val_loss'] is not None:
                    print(f"  Val:   Loss={metrics['val_loss']:.4f}, Acc={metrics['val_accuracy']:.4f}")
            
            # Save best model
            if save_path and metrics['val_accuracy'] is not None:
                if metrics['val_accuracy'] > best_val_acc:
                    best_val_acc = metrics['val_accuracy']
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  New best model saved! Val Acc: {best_val_acc:.4f}")
        
        print("Training complete!")
        
        if save_path and val_loader is None:
            # Save final model if no validation
            torch.save(self.model.state_dict(), save_path)
            print(f"Final model saved to {save_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        if self.val_losses:
            ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Acc', color='blue')
        if self.val_accuracies:
            ax2.plot(self.val_accuracies, label='Val Acc', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()


def train_imitation_model(episodes: List[GameplayEpisode],
                         train_split: float = 0.8,
                         batch_size: int = 256,
                         num_epochs: int = 50,
                         learning_rate: float = 1e-3,
                         save_path: Optional[str] = None) -> Tuple[BlackjackPolicyNet, ImitationTrainer]:
    """
    Train imitation learning model on demonstration data.
    
    Args:
        episodes: List of gameplay episodes
        train_split: Fraction of data for training (rest for validation)
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        save_path: Optional path to save trained model
        
    Returns:
        model: Trained policy network
        trainer: Trainer with training history
    """
    
    # Convert episodes to training data
    from .data_generation import DatasetGenerator
    generator = DatasetGenerator()
    observations, actions, legal_masks = generator.episodes_to_training_data(episodes)
    
    print(f"Training data: {len(observations)} samples")
    print(f"Action distribution: {np.bincount(actions) / len(actions)}")
    
    # Split data
    num_train = int(len(observations) * train_split)
    indices = np.random.permutation(len(observations))
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    # Create datasets
    train_dataset = BlackjackDataset(
        observations[train_indices],
        actions[train_indices], 
        legal_masks[train_indices]
    )
    
    val_dataset = BlackjackDataset(
        observations[val_indices],
        actions[val_indices],
        legal_masks[val_indices]
    ) if len(val_indices) > 0 else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    ) if val_dataset else None
    
    # Create model and trainer
    model = BlackjackPolicyNet(
        input_size=get_feature_dimensions(),
        hidden_size=256,
        num_actions=5
    )
    
    trainer = ImitationTrainer(
        model=model,
        learning_rate=learning_rate
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_path=save_path
    )
    
    return model, trainer
