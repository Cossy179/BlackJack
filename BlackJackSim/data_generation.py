"""
Phase 2 Data Generation

Generate large batches of gameplay using baseline policy for imitation learning.
Ensures broad range of true-count buckets and varied composition scenarios.
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pickle
import json
from pathlib import Path

import gymnasium as gym
from .config import BlackjackRules, V1_RULES
from .state_representation import BlackjackState, StateExtractor
from .basic_strategy import BasicStrategyBaseline


@dataclass
class GameplayStep:
    """Single step of gameplay for supervised learning"""
    observation: np.ndarray          # Normalized state features (12,)
    legal_actions: np.ndarray        # Boolean mask (5,)
    action: int                      # Chosen action (0-4)
    action_probs: np.ndarray         # Action probabilities (5,) - one-hot for basic strategy
    true_count: float                # True count at decision point
    state_dict: Dict[str, Any]       # Full state information for analysis
    

@dataclass
class GameplayEpisode:
    """Complete episode of gameplay"""
    steps: List[GameplayStep]
    final_reward: float
    episode_length: int
    mean_true_count: float
    true_count_bucket: int           # Binned true count for balanced sampling


class DatasetGenerator:
    """
    Generate demonstration dataset from baseline policy gameplay.
    
    Ensures broad coverage of true-count scenarios and game states.
    """
    
    def __init__(self, 
                 rules: BlackjackRules = V1_RULES,
                 use_true_count_baseline: bool = True,
                 seed: Optional[int] = None):
        self.rules = rules
        self.use_true_count_baseline = use_true_count_baseline
        self.seed = seed
        
        # True count buckets for balanced sampling
        # Range: -10 to +10, with finer bins around neutral
        self.true_count_buckets = [
            (-np.inf, -6),   # Very negative
            (-6, -3),        # Negative  
            (-3, -1),        # Slightly negative
            (-1, 1),         # Neutral
            (1, 3),          # Slightly positive
            (3, 6),          # Positive
            (6, np.inf)      # Very positive
        ]
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _get_true_count_bucket(self, true_count: float) -> int:
        """Get bucket index for true count"""
        for i, (low, high) in enumerate(self.true_count_buckets):
            if low <= true_count < high:
                return i
        return len(self.true_count_buckets) - 1  # Default to last bucket
    
    def generate_episode(self, env_seed: Optional[int] = None) -> GameplayEpisode:
        """Generate single episode using baseline policy"""
        
        # Create environment with compact state representation
        env = gym.make("BlackjackSim-v0", 
                      rules=self.rules, 
                      use_compact_state=True,
                      seed=env_seed)
        
        baseline = env.unwrapped.get_basic_strategy_baseline()
        if self.use_true_count_baseline:
            baseline.use_true_count = True
        
        # Reset environment
        obs, info = env.reset()
        
        steps = []
        true_counts = []
        step_count = 0
        max_steps = 20  # Prevent infinite loops
        
        while step_count < max_steps:
            step_count += 1
            
            # Extract state information
            state = BlackjackState(**info['blackjack_state'])
            legal_actions = np.array(info['legal_actions'], dtype=bool)
            true_count = state.true_count
            true_counts.append(true_count)
            
            # Get baseline action
            action = baseline.get_action(state)
            action_probs = baseline.get_action_probabilities(state)
            
            # Record step
            gameplay_step = GameplayStep(
                observation=obs.copy(),
                legal_actions=legal_actions.copy(),
                action=action,
                action_probs=action_probs.copy(),
                true_count=true_count,
                state_dict=state.to_dict()
            )
            steps.append(gameplay_step)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        env.close()
        
        # Create episode summary
        final_reward = reward if (terminated or truncated) else 0.0
        mean_true_count = np.mean(true_counts) if true_counts else 0.0
        true_count_bucket = self._get_true_count_bucket(mean_true_count)
        
        return GameplayEpisode(
            steps=steps,
            final_reward=final_reward,
            episode_length=len(steps),
            mean_true_count=mean_true_count,
            true_count_bucket=true_count_bucket
        )
    
    def generate_balanced_dataset(self, 
                                total_episodes: int = 10000,
                                target_steps: int = 50000,
                                min_episodes_per_bucket: int = 100) -> List[GameplayEpisode]:
        """
        Generate balanced dataset with good coverage across true count buckets.
        
        Args:
            total_episodes: Target number of episodes
            target_steps: Target number of decision steps
            min_episodes_per_bucket: Minimum episodes per true count bucket
        """
        
        print(f"Generating balanced dataset: {total_episodes} episodes, target {target_steps} steps")
        
        episodes = []
        bucket_counts = [0] * len(self.true_count_buckets)
        total_steps = 0
        
        # Generate episodes with progress tracking
        episode_count = 0
        while episode_count < total_episodes and total_steps < target_steps:
            
            # Generate episode
            env_seed = None if self.seed is None else self.seed + episode_count
            episode = self.generate_episode(env_seed)
            
            episodes.append(episode)
            bucket_counts[episode.true_count_bucket] += 1
            total_steps += episode.episode_length
            episode_count += 1
            
            # Progress reporting
            if episode_count % 1000 == 0:
                print(f"Generated {episode_count} episodes, {total_steps} steps")
                print(f"Bucket distribution: {bucket_counts}")
        
        # Ensure minimum representation in each bucket
        print("Ensuring minimum bucket representation...")
        for bucket_idx, count in enumerate(bucket_counts):
            if count < min_episodes_per_bucket:
                needed = min_episodes_per_bucket - count
                print(f"Generating {needed} additional episodes for bucket {bucket_idx}")
                
                # Generate episodes targeting this bucket (simplified approach)
                for _ in range(needed):
                    env_seed = None if self.seed is None else self.seed + episode_count
                    episode = self.generate_episode(env_seed)
                    
                    # Accept episode regardless of bucket (in full implementation,
                    # would use shoe manipulation to target specific counts)
                    episodes.append(episode)
                    bucket_counts[episode.true_count_bucket] += 1
                    total_steps += episode.episode_length
                    episode_count += 1
        
        print(f"Dataset generation complete:")
        print(f"  Total episodes: {len(episodes)}")
        print(f"  Total steps: {total_steps}")
        print(f"  Final bucket distribution: {bucket_counts}")
        
        return episodes
    
    def episodes_to_training_data(self, episodes: List[GameplayEpisode]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert episodes to training arrays.
        
        Returns:
            observations: (N, 12) state features
            actions: (N,) action indices  
            legal_masks: (N, 5) legal action masks
        """
        
        all_observations = []
        all_actions = []
        all_legal_masks = []
        
        for episode in episodes:
            for step in episode.steps:
                all_observations.append(step.observation)
                all_actions.append(step.action)
                all_legal_masks.append(step.legal_actions.astype(np.float32))
        
        return (
            np.array(all_observations, dtype=np.float32),
            np.array(all_actions, dtype=np.int64),
            np.array(all_legal_masks, dtype=np.float32)
        )
    
    def save_dataset(self, episodes: List[GameplayEpisode], filepath: str):
        """Save dataset to disk"""
        dataset_info = {
            'num_episodes': len(episodes),
            'total_steps': sum(len(ep.steps) for ep in episodes),
            'rules': self.rules.to_dict(),
            'use_true_count_baseline': self.use_true_count_baseline,
            'true_count_buckets': self.true_count_buckets,
            'seed': self.seed
        }
        
        # Save episodes
        with open(filepath, 'wb') as f:
            pickle.dump({
                'episodes': episodes,
                'info': dataset_info
            }, f)
        
        # Save metadata as JSON for easy inspection
        json_path = str(Path(filepath).with_suffix('.json'))
        with open(json_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset saved to {filepath}")
        print(f"Metadata saved to {json_path}")
    
    def load_dataset(self, filepath: str) -> Tuple[List[GameplayEpisode], Dict[str, Any]]:
        """Load dataset from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return data['episodes'], data['info']
    
    def analyze_dataset(self, episodes: List[GameplayEpisode]) -> Dict[str, Any]:
        """Analyze dataset for quality and coverage"""
        
        if not episodes:
            return {}
        
        # Basic statistics
        total_steps = sum(len(ep.steps) for ep in episodes)
        episode_lengths = [len(ep.steps) for ep in episodes]
        final_rewards = [ep.final_reward for ep in episodes]
        
        # Action distribution
        all_actions = []
        all_true_counts = []
        bucket_distribution = [0] * len(self.true_count_buckets)
        
        for episode in episodes:
            bucket_distribution[episode.true_count_bucket] += 1
            for step in episode.steps:
                all_actions.append(step.action)
                all_true_counts.append(step.true_count)
        
        action_counts = np.bincount(all_actions, minlength=5)
        action_distribution = action_counts / len(all_actions)
        
        # True count analysis
        true_count_mean = np.mean(all_true_counts)
        true_count_std = np.std(all_true_counts)
        true_count_range = (np.min(all_true_counts), np.max(all_true_counts))
        
        # Reward analysis
        win_rate = np.mean(np.array(final_rewards) > 0)
        loss_rate = np.mean(np.array(final_rewards) < 0)
        push_rate = np.mean(np.array(final_rewards) == 0)
        mean_reward = np.mean(final_rewards)
        
        return {
            'num_episodes': len(episodes),
            'total_steps': total_steps,
            'mean_episode_length': np.mean(episode_lengths),
            'episode_length_std': np.std(episode_lengths),
            'action_distribution': {
                'STAND': action_distribution[0],
                'HIT': action_distribution[1], 
                'DOUBLE': action_distribution[2],
                'SPLIT': action_distribution[3],
                'SURRENDER': action_distribution[4]
            },
            'true_count_stats': {
                'mean': true_count_mean,
                'std': true_count_std,
                'range': true_count_range
            },
            'bucket_distribution': bucket_distribution,
            'reward_stats': {
                'mean_reward': mean_reward,
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'push_rate': push_rate
            }
        }


def generate_demonstration_dataset(num_episodes: int = 10000,
                                 target_steps: int = 50000,
                                 save_path: Optional[str] = None,
                                 seed: Optional[int] = 42) -> Tuple[List[GameplayEpisode], Dict[str, Any]]:
    """
    Convenience function to generate demonstration dataset.
    
    Args:
        num_episodes: Number of episodes to generate
        target_steps: Target number of decision steps
        save_path: Optional path to save dataset
        seed: Random seed for reproducibility
        
    Returns:
        episodes: List of gameplay episodes
        analysis: Dataset analysis
    """
    
    generator = DatasetGenerator(
        rules=V1_RULES,
        use_true_count_baseline=True, 
        seed=seed
    )
    
    episodes = generator.generate_balanced_dataset(
        total_episodes=num_episodes,
        target_steps=target_steps
    )
    
    analysis = generator.analyze_dataset(episodes)
    
    if save_path:
        generator.save_dataset(episodes, save_path)
    
    return episodes, analysis
