"""
Bet Sizing Policy for Blackjack (Phase 5)

Implements a separate bet-sizing policy that:
1. Conditions on true count and shoe depth
2. Chooses from discrete bet sizes
3. Optimizes bankroll growth using Kelly criterion approximation
4. Trains with policy gradient methods while holding play policy fixed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import math

from .device_utils import device_manager, to_device, get_device


@dataclass
class BetSizingConfig:
    """Configuration for bet sizing policy"""
    
    # Bet size options (in units)
    bet_sizes: List[float] = None
    min_bet: float = 1.0
    max_bet: float = 10.0
    
    # Network architecture
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Kelly criterion parameters
    use_kelly_criterion: bool = True
    bankroll_fraction_limit: float = 0.25  # Max fraction of bankroll to bet
    kelly_scaling: float = 0.5  # Conservative Kelly scaling
    
    # Environment parameters
    initial_bankroll: float = 1000.0
    
    def __post_init__(self):
        if self.bet_sizes is None:
            # Default bet sizes: 1, 2, 3, 5, 8 units
            self.bet_sizes = [1.0, 2.0, 3.0, 5.0, 8.0]


class BetSizingPolicy(nn.Module):
    """
    Neural network policy for bet sizing decisions.
    
    Inputs:
    - True count
    - Shoe depth (cards remaining / total cards)
    - Current bankroll
    - Recent performance metrics
    
    Output:
    - Probability distribution over discrete bet sizes
    """
    
    def __init__(self, config: BetSizingConfig):
        super().__init__()
        self.config = config
        
        # Input features: true_count, shoe_depth, bankroll_ratio, recent_performance
        input_dim = 4
        
        # Policy network (actor)
        layers = []
        prev_dim = input_dim
        
        for _ in range(config.num_layers):
            layers.extend([
                nn.Linear(prev_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = config.hidden_dim
            
        layers.append(nn.Linear(prev_dim, len(config.bet_sizes)))
        self.policy_net = nn.Sequential(*layers)
        
        # Value network (critic)
        value_layers = []
        prev_dim = input_dim
        
        for _ in range(config.num_layers):
            value_layers.extend([
                nn.Linear(prev_dim, config.hidden_dim),
                nn.ReLU(), 
                nn.Dropout(config.dropout)
            ])
            prev_dim = config.hidden_dim
            
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_net = nn.Sequential(*value_layers)
        
        # Move to device
        self.to(get_device())
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both policy and value networks.
        
        Args:
            state: Batch of state features [batch_size, 4]
            
        Returns:
            policy_logits: Raw logits for bet size distribution [batch_size, num_bet_sizes]
            value: State value estimates [batch_size, 1]
        """
        policy_logits = self.policy_net(state)
        value = self.value_net(state)
        return policy_logits, value
    
    def get_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities from policy network"""
        policy_logits, _ = self.forward(state)
        return F.softmax(policy_logits, dim=-1)
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select bet size action.
        
        Args:
            state: State features [4]
            deterministic: If True, select argmax action
            
        Returns:
            action: Bet size index
            log_prob: Log probability of selected action
            value: State value estimate
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        policy_logits, value = self.forward(state)
        action_probs = F.softmax(policy_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1)))
        else:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
        return action.item(), log_prob.squeeze(), value.squeeze()


class KellyCriterionCalculator:
    """
    Calculates optimal bet sizes using Kelly Criterion approximation.
    
    Kelly formula: f = (bp - q) / b
    Where:
    - f = fraction of bankroll to bet
    - b = odds received (1 for even money)
    - p = probability of winning
    - q = probability of losing (1 - p)
    """
    
    @staticmethod
    def calculate_kelly_fraction(win_probability: float, 
                               odds: float = 1.0,
                               scaling: float = 0.5) -> float:
        """
        Calculate Kelly fraction for bet sizing.
        
        Args:
            win_probability: Estimated probability of winning
            odds: Odds received on winning bet (1.0 for even money)
            scaling: Conservative scaling factor (0.5 = half Kelly)
            
        Returns:
            Fraction of bankroll to bet (0.0 to 1.0)
        """
        if win_probability <= 0.5:
            return 0.0  # No edge, don't bet
            
        # Kelly formula: f = (bp - q) / b
        p = win_probability
        q = 1.0 - p
        b = odds
        
        kelly_fraction = (b * p - q) / b
        
        # Apply conservative scaling
        kelly_fraction *= scaling
        
        # Ensure non-negative and reasonable bounds
        return max(0.0, min(kelly_fraction, 0.25))  # Cap at 25% of bankroll
    
    @staticmethod
    def estimate_win_probability(true_count: float, base_win_rate: float = 0.42) -> float:
        """
        Estimate win probability based on true count.
        
        Args:
            true_count: Current true count
            base_win_rate: Base win rate at neutral count
            
        Returns:
            Estimated win probability
        """
        # Rough approximation: each true count point adds ~0.5% to win rate
        count_bonus = true_count * 0.005
        win_prob = base_win_rate + count_bonus
        
        # Clamp to reasonable bounds
        return max(0.1, min(win_prob, 0.6))


@dataclass
class BetSizingState:
    """State representation for bet sizing decisions"""
    
    true_count: float
    shoe_depth: float      # Fraction of cards remaining (0.0 to 1.0)
    bankroll_ratio: float  # Current bankroll / initial bankroll
    recent_performance: float  # Recent expected value per hand
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for neural network input"""
        features = [
            self.true_count,
            self.shoe_depth, 
            self.bankroll_ratio,
            self.recent_performance
        ]
        return device_manager.tensor(features, dtype=torch.float32)
    
    @classmethod
    def from_game_state(cls, 
                       true_count: float,
                       cards_remaining: int,
                       total_cards: int,
                       current_bankroll: float,
                       initial_bankroll: float,
                       recent_performance: float) -> 'BetSizingState':
        """Create from game state information"""
        return cls(
            true_count=true_count,
            shoe_depth=cards_remaining / total_cards,
            bankroll_ratio=current_bankroll / initial_bankroll,
            recent_performance=recent_performance
        )


class BetSizingAgent:
    """
    Main agent for bet sizing decisions.
    
    Combines neural network policy with Kelly criterion for optimal bet sizing.
    """
    
    def __init__(self, config: BetSizingConfig = None):
        self.config = config or BetSizingConfig()
        self.policy = BetSizingPolicy(self.config)
        self.kelly_calculator = KellyCriterionCalculator()
        
        # Optimizer for policy gradient training
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate
        )
        
        # Training history
        self.training_history = []
        
        print(f"💰 Bet Sizing Agent initialized")
        print(f"   Bet sizes: {self.config.bet_sizes}")
        print(f"   Policy network: {self.config.hidden_dim}x{self.config.num_layers}")
        print(f"   Kelly criterion: {'enabled' if self.config.use_kelly_criterion else 'disabled'}")
        print(f"   Device: {get_device()}")
    
    def select_bet_size(self, 
                       state: BetSizingState,
                       deterministic: bool = False) -> Tuple[float, Dict[str, Any]]:
        """
        Select bet size for current state.
        
        Args:
            state: Current betting state
            deterministic: If True, use deterministic policy
            
        Returns:
            bet_size: Selected bet size in units
            info: Additional information about the decision
        """
        state_tensor = state.to_tensor()
        
        # Get policy decision
        action_idx, log_prob, value = self.policy.select_action(state_tensor, deterministic)
        policy_bet_size = self.config.bet_sizes[action_idx]
        
        # Calculate Kelly suggestion if enabled
        kelly_bet_size = None
        if self.config.use_kelly_criterion:
            win_prob = self.kelly_calculator.estimate_win_probability(state.true_count)
            kelly_fraction = self.kelly_calculator.calculate_kelly_fraction(
                win_prob, scaling=self.config.kelly_scaling
            )
            
            # Convert Kelly fraction to actual bet size
            current_bankroll = state.bankroll_ratio * self.config.initial_bankroll
            kelly_bet_size = kelly_fraction * current_bankroll
            
            # Snap to nearest available bet size
            kelly_bet_size = min(self.config.bet_sizes, 
                               key=lambda x: abs(x - kelly_bet_size))
        
        # Final bet size (could blend policy and Kelly, but for now use policy)
        final_bet_size = policy_bet_size
        
        # Prepare info
        info = {
            'action_idx': action_idx,
            'log_prob': log_prob.item() if hasattr(log_prob, 'item') else log_prob,
            'value': value.item() if hasattr(value, 'item') else value,
            'policy_bet_size': policy_bet_size,
            'kelly_bet_size': kelly_bet_size,
            'true_count': state.true_count,
            'shoe_depth': state.shoe_depth,
            'bankroll_ratio': state.bankroll_ratio
        }
        
        return final_bet_size, info
    
    def update_policy(self, trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Update policy using collected trajectories (PPO-style).
        
        Args:
            trajectories: List of trajectory data
            
        Returns:
            Training metrics
        """
        if not trajectories:
            return {}
        
        # Convert trajectories to tensors
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        
        for traj in trajectories:
            states.append(traj['state'].to_tensor())
            actions.append(traj['action_idx'])
            log_probs.append(traj['log_prob'])
            rewards.append(traj['reward'])
            values.append(traj['value'])
        
        states = torch.stack(states)
        actions = device_manager.tensor(actions, dtype=torch.long)
        old_log_probs = device_manager.tensor(log_probs, dtype=torch.float32)
        rewards = device_manager.tensor(rewards, dtype=torch.float32)
        old_values = device_manager.tensor(values, dtype=torch.float32)
        
        # Calculate advantages using GAE
        advantages = self._calculate_advantages(rewards, old_values)
        returns = advantages + old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy update
        total_loss = 0.0
        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        
        # Multiple epochs for each update
        for _ in range(3):  # PPO update epochs
            # Forward pass
            policy_logits, new_values = self.policy(states)
            action_probs = F.softmax(policy_logits, dim=-1)
            
            # Calculate new log probabilities
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()
            
            # Calculate ratio for PPO clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Policy loss with clipping
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values.squeeze(), returns)
            
            # Total loss
            loss = (policy_loss + 
                   self.config.value_loss_coef * value_loss - 
                   self.config.entropy_coef * entropy)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()
            entropy_total += entropy.item()
        
        # Training metrics
        metrics = {
            'total_loss': total_loss / 3,
            'policy_loss': policy_loss_total / 3,
            'value_loss': value_loss_total / 3,
            'entropy': entropy_total / 3,
            'avg_advantage': advantages.mean().item(),
            'avg_return': returns.mean().item()
        }
        
        self.training_history.append(metrics)
        return metrics
    
    def _calculate_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Calculate advantages using Generalized Advantage Estimation (GAE)"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages[t] = gae
            
        return advantages
    
    def save_model(self, filepath: str):
        """Save model weights and configuration"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, filepath)
        
    def load_model(self, filepath: str):
        """Load model weights and configuration"""
        checkpoint = torch.load(filepath, map_location=get_device())
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])


def create_bet_sizing_config(
    conservative: bool = True,
    bet_spread: str = "moderate"
) -> BetSizingConfig:
    """
    Create bet sizing configuration for different strategies.
    
    Args:
        conservative: If True, use conservative Kelly scaling and smaller bets
        bet_spread: "small", "moderate", or "aggressive" bet spread
        
    Returns:
        BetSizingConfig instance
    """
    
    # Define bet spreads
    bet_spreads = {
        "small": [1.0, 2.0, 3.0],           # 1-3 unit spread
        "moderate": [1.0, 2.0, 3.0, 5.0],  # 1-5 unit spread  
        "aggressive": [1.0, 2.0, 4.0, 8.0, 12.0]  # 1-12 unit spread
    }
    
    config = BetSizingConfig(
        bet_sizes=bet_spreads.get(bet_spread, bet_spreads["moderate"]),
        kelly_scaling=0.25 if conservative else 0.5,
        bankroll_fraction_limit=0.15 if conservative else 0.25,
        learning_rate=2e-4 if conservative else 3e-4,
        entropy_coef=0.02 if conservative else 0.01
    )
    
    return config
