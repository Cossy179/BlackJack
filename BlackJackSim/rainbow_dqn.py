"""
Rainbow DQN Implementation for Blackjack RL (Phase 3/4)

Implements key Rainbow DQN components:
- Dueling network architecture (state value + advantage streams)
- Distributional value learning (C51 categorical DQN)
- Double Q-learning for reduced overestimation
- Prioritized experience replay
- Noisy networks for exploration
- Multi-step returns
- Target network with soft updates

Designed to work with curriculum learning and action masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import random
from collections import namedtuple, deque
import math

from .device_utils import device_manager, to_device, get_device
from .state_representation import get_feature_dimensions
from .BlackJack import PlayOptions


# Experience tuple for replay buffer
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done', 
    'legal_actions', 'next_legal_actions', 'n_step_return', 'n_step_discount'
])


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for exploration (NoisyNets).
    
    Replaces epsilon-greedy exploration with parameter noise.
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise buffers (not learnable)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Sample new noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Outer product for weight noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise using factorized Gaussian noise"""
        x = torch.randn(size, device=get_device())
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    """
    Dueling DQN with distributional value learning (C51).
    
    Separates state value and advantage streams, then combines them.
    Uses categorical distribution for value learning.
    """
    
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 512,
        num_actions: int = 5,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        use_noisy: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Create support for categorical distribution
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream (state value function)
        if use_noisy:
            self.value_net = nn.Sequential(
                NoisyLinear(hidden_dim, hidden_dim),
                nn.ReLU(),
                NoisyLinear(hidden_dim, num_atoms)
            )
        else:
            self.value_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_atoms)
            )
        
        # Advantage stream (action advantages)
        if use_noisy:
            self.advantage_net = nn.Sequential(
                NoisyLinear(hidden_dim, hidden_dim),
                nn.ReLU(),
                NoisyLinear(hidden_dim, num_actions * num_atoms)
            )
        else:
            self.advantage_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions * num_atoms)
            )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, legal_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass returning action-value distributions.
        
        Args:
            x: State features [batch_size, input_dim]
            legal_actions: Legal action mask [batch_size, num_actions]
            
        Returns:
            Action-value distributions [batch_size, num_actions, num_atoms]
        """
        batch_size = x.size(0)
        
        # Extract features
        features = self.feature_net(x)
        
        # Value stream: V(s) distribution
        value_dist = self.value_net(features)  # [batch_size, num_atoms]
        value_dist = value_dist.unsqueeze(1).expand(batch_size, self.num_actions, self.num_atoms)
        
        # Advantage stream: A(s,a) distributions
        advantage_dist = self.advantage_net(features)  # [batch_size, num_actions * num_atoms]
        advantage_dist = advantage_dist.view(batch_size, self.num_actions, self.num_atoms)
        
        # Dueling combination: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        advantage_mean = advantage_dist.mean(dim=1, keepdim=True)
        q_dist = value_dist + advantage_dist - advantage_mean
        
        # Apply legal action masking if provided
        if legal_actions is not None:
            # Set illegal actions to very negative values
            illegal_mask = (legal_actions == 0).unsqueeze(-1)  # [batch_size, num_actions, 1]
            q_dist = q_dist.masked_fill(illegal_mask, float('-inf'))
        
        return q_dist
    
    def get_q_values(self, x: torch.Tensor, legal_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get expected Q-values from distributions.
        
        Returns:
            Q-values [batch_size, num_actions]
        """
        q_dist = self.forward(x, legal_actions)
        q_probs = F.softmax(q_dist, dim=-1)
        q_values = torch.sum(q_probs * self.support.view(1, 1, -1), dim=-1)
        return q_values
    
    def get_action(self, x: torch.Tensor, legal_actions: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Select action using the current policy.
        
        Args:
            x: Single state [input_dim]
            legal_actions: Legal action mask [num_actions]
            epsilon: Epsilon for epsilon-greedy (usually 0 for noisy nets)
            
        Returns:
            Selected action index
        """
        if random.random() < epsilon:
            # Random legal action
            legal_indices = torch.nonzero(legal_actions, as_tuple=True)[0]
            return random.choice(legal_indices.cpu().numpy())
        
        with torch.no_grad():
            x_batch = x.unsqueeze(0)
            legal_batch = legal_actions.unsqueeze(0)
            q_values = self.get_q_values(x_batch, legal_batch)
            
            # Mask illegal actions
            masked_q = q_values.clone()
            masked_q[legal_batch == 0] = float('-inf')
            
            action = torch.argmax(masked_q, dim=1).item()
        
        return action
    
    def reset_noise(self):
        """Reset noise in noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Samples experiences based on their TD-error, allowing the agent
    to learn more from surprising transitions.
    """
    
    def __init__(
        self,
        capacity: int = 1000000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame_count = 0
    
    def beta(self) -> float:
        """Get current beta value (annealed from beta_start to 1.0)"""
        progress = min(self.frame_count / self.beta_frames, 1.0)
        return self.beta_start + progress * (1.0 - self.beta_start)
    
    def push(self, experience: Experience) -> None:
        """Add experience to buffer"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        # Ensure max_priority is at least epsilon to avoid zero priorities
        max_priority = max(max_priority, self.epsilon)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritized sampling.
        
        Returns:
            experiences: List of sampled experiences
            indices: Indices of sampled experiences
            weights: Importance sampling weights
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        probabilities = priorities ** self.alpha
        # Ensure probabilities are valid (no NaN or zero sum)
        if probabilities.sum() == 0 or np.isnan(probabilities.sum()):
            # Use uniform distribution as fallback
            probabilities = np.ones_like(probabilities)
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta())
        weights /= weights.max()
        
        self.frame_count += 1
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def __len__(self) -> int:
        return len(self.buffer)


class RainbowDQNAgent:
    """
    Rainbow DQN Agent with all components.
    
    Combines dueling network, distributional learning, prioritized replay,
    noisy networks, multi-step returns, and double Q-learning.
    """
    
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 512,
        num_actions: int = 5,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        lr: float = 1e-4,
        gamma: float = 0.99,
        n_step: int = 3,
        buffer_capacity: int = 1000000,
        batch_size: int = 512,
        target_update_freq: int = 1000,
        tau: float = 0.005,
        use_noisy: bool = True,
        priority_alpha: float = 0.6,
        priority_beta_start: float = 0.4,
        priority_beta_frames: int = 100000
    ):
        
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.use_noisy = use_noisy
        
        # Create support for categorical distribution
        self.support = torch.linspace(v_min, v_max, num_atoms).to(get_device())
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Networks
        self.online_net = to_device(DuelingDQN(
            input_dim, hidden_dim, num_actions, num_atoms, v_min, v_max, use_noisy
        ))
        self.target_net = to_device(DuelingDQN(
            input_dim, hidden_dim, num_actions, num_atoms, v_min, v_max, use_noisy
        ))
        
        # Copy parameters to target network
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        
        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=priority_alpha,
            beta_start=priority_beta_start,
            beta_frames=priority_beta_frames
        )
        
        # Multi-step buffer for n-step returns
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Training tracking
        self.total_steps = 0
        self.training_steps = 0
        
        print(f"🌈 Rainbow DQN Agent initialized")
        print(f"   Device: {get_device()}")
        print(f"   Network: {hidden_dim}x3 hidden layers")
        print(f"   Atoms: {num_atoms} ({v_min:.1f} to {v_max:.1f})")
        print(f"   N-step: {n_step}, Batch: {batch_size}")
        print(f"   Noisy nets: {use_noisy}")
    
    def get_action(self, state: torch.Tensor, legal_actions: torch.Tensor, epsilon: float = 0.0) -> int:
        """Select action using current policy"""
        return self.online_net.get_action(state, legal_actions, epsilon)
    
    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        legal_actions: torch.Tensor,
        next_legal_actions: torch.Tensor
    ) -> None:
        """Store experience in replay buffer with n-step processing"""
        
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done, legal_actions, next_legal_actions))
        
        # Only store in replay buffer if we have enough for n-step
        if len(self.n_step_buffer) == self.n_step:
            # Calculate n-step return
            n_step_return = 0.0
            n_step_discount = 1.0
            
            for i in range(self.n_step):
                step_reward = self.n_step_buffer[i][2]
                n_step_return += n_step_discount * step_reward
                n_step_discount *= self.gamma
                
                # Stop if episode ended
                if self.n_step_buffer[i][4]:  # done
                    break
            
            # Get first and last states
            first_step = self.n_step_buffer[0]
            last_step = self.n_step_buffer[-1]
            
            experience = Experience(
                state=first_step[0],
                action=first_step[1],
                reward=n_step_return,
                next_state=last_step[3],
                done=last_step[4],
                legal_actions=first_step[5],
                next_legal_actions=last_step[6],
                n_step_return=n_step_return,
                n_step_discount=n_step_discount
            )
            
            self.replay_buffer.push(experience)
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        batch = self._prepare_batch(experiences, weights)
        
        # Compute loss
        loss, td_errors = self._compute_loss(batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Update target network
        if self.training_steps % self.target_update_freq == 0:
            self._soft_update_target()
        
        # Reset noise
        if self.use_noisy:
            self.online_net.reset_noise()
            self.target_net.reset_noise()
        
        self.training_steps += 1
        
        return {
            'loss': loss.item(),
            'td_error_mean': td_errors.mean().item(),
            'buffer_size': len(self.replay_buffer),
            'beta': self.replay_buffer.beta()
        }
    
    def _prepare_batch(self, experiences: List[Experience], weights: np.ndarray) -> Dict[str, torch.Tensor]:
        """Prepare batch tensors"""
        states = torch.stack([exp.state for exp in experiences])
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long, device=get_device())
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32, device=get_device())
        next_states = torch.stack([exp.next_state for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool, device=get_device())
        legal_actions = torch.stack([exp.legal_actions for exp in experiences])
        next_legal_actions = torch.stack([exp.next_legal_actions for exp in experiences])
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=get_device())
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'legal_actions': legal_actions,
            'next_legal_actions': next_legal_actions,
            'weights': weights_tensor
        }
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distributional loss with double Q-learning"""
        
        # Current Q-distributions
        current_dist = self.online_net(batch['states'], batch['legal_actions'])
        current_dist = current_dist[range(self.batch_size), batch['actions']]
        
        # Target Q-distributions using double Q-learning
        with torch.no_grad():
            # Use online network to select actions
            next_q_values = self.online_net.get_q_values(batch['next_states'], batch['next_legal_actions'])
            next_q_values[batch['next_legal_actions'] == 0] = float('-inf')
            next_actions = next_q_values.argmax(dim=1)
            
            # Use target network to evaluate selected actions
            next_dist = self.target_net(batch['next_states'], batch['next_legal_actions'])
            next_dist = next_dist[range(self.batch_size), next_actions]
            
            # Compute target distribution
            target_dist = self._compute_target_distribution(
                batch['rewards'], next_dist, batch['dones']
            )
        
        # Cross-entropy loss
        log_current_dist = F.log_softmax(current_dist, dim=1)
        loss = -torch.sum(target_dist * log_current_dist, dim=1)
        
        # Apply importance sampling weights
        weighted_loss = batch['weights'] * loss
        
        # TD errors for priority update
        td_errors = loss.detach()
        
        return weighted_loss.mean(), td_errors
    
    def _compute_target_distribution(
        self,
        rewards: torch.Tensor,
        next_dist: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute target distribution for categorical DQN"""
        
        batch_size = rewards.size(0)
        
        # Compute projected target support
        target_support = rewards.unsqueeze(1) + (self.gamma ** self.n_step) * self.support.unsqueeze(0) * (~dones).unsqueeze(1)
        target_support = target_support.clamp(self.v_min, self.v_max)
        
        # Compute projection indices and weights
        b = (target_support - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Fix disappearing probability mass when l = b = u (b is integer)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.num_atoms - 1)) * (l == u)] += 1
        
        # Distribute probability of next_dist
        target_dist = torch.zeros_like(next_dist)
        next_prob = F.softmax(next_dist, dim=1)
        
        for i in range(batch_size):
            target_dist[i].index_add_(0, l[i], next_prob[i] * (u[i].float() - b[i]))
            target_dist[i].index_add_(0, u[i], next_prob[i] * (b[i] - l[i].float()))
        
        return target_dist
    
    def _soft_update_target(self):
        """Soft update of target network parameters"""
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)
    
    def save_model(self, filepath: str):
        """Save model weights"""
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_steps': self.training_steps
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=get_device())
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_steps = checkpoint['training_steps']
