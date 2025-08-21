"""
Curriculum Learning System for Blackjack RL (Phase 4)

Implements progressive action unlocking to accelerate convergence:
1. Stage 1: HIT, STAND, DOUBLE only
2. Stage 2: Add SURRENDER 
3. Stage 3: Add SPLIT (starting with Aces and Eights, then general pairs)

The curriculum maintains the same network and optimizer throughout,
simply expanding the set of legal actions as rules unlock.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum, auto
import json
import time

from .config import BlackjackRules, V1_RULES
from .BlackJack import PlayOptions


class CurriculumStage(Enum):
    """Curriculum learning stages"""
    BASIC_ACTIONS = 1      # HIT, STAND, DOUBLE only
    ADD_SURRENDER = 2      # + SURRENDER
    ADD_SPLIT_ACES = 3     # + SPLIT for Aces and Eights
    FULL_ACTIONS = 4       # All actions including general pairs


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    
    # Stage transition criteria
    min_episodes_per_stage: int = 50000      # Minimum episodes before considering transition
    performance_window: int = 10000          # Episodes to average for performance check
    stability_threshold: float = 0.01       # Max standard deviation for "stable" performance
    min_improvement_rate: float = 0.001     # Min improvement per 1000 episodes
    
    # Performance thresholds for stage transitions
    basic_actions_threshold: float = -0.1    # Expected value to unlock surrender
    surrender_threshold: float = -0.05       # Expected value to unlock splits
    split_aces_threshold: float = 0.0        # Expected value for full actions
    
    # Training parameters per stage
    learning_rates: Dict[CurriculumStage, float] = None
    batch_sizes: Dict[CurriculumStage, int] = None
    
    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = {
                CurriculumStage.BASIC_ACTIONS: 1e-4,
                CurriculumStage.ADD_SURRENDER: 8e-5,    # Slightly lower as we add complexity
                CurriculumStage.ADD_SPLIT_ACES: 6e-5,
                CurriculumStage.FULL_ACTIONS: 4e-5
            }
        
        if self.batch_sizes is None:
            self.batch_sizes = {
                CurriculumStage.BASIC_ACTIONS: 512,
                CurriculumStage.ADD_SURRENDER: 768,
                CurriculumStage.ADD_SPLIT_ACES: 1024,
                CurriculumStage.FULL_ACTIONS: 1024
            }


class ActionMaskManager:
    """
    Manages action masking for curriculum learning.
    
    Progressively unlocks actions based on curriculum stage while
    respecting the underlying game rules.
    """
    
    def __init__(self, rules: BlackjackRules = V1_RULES):
        self.rules = rules
        self.stage = CurriculumStage.BASIC_ACTIONS
        
        # Define which actions are available at each stage
        self.stage_actions = {
            CurriculumStage.BASIC_ACTIONS: {PlayOptions.STAY, PlayOptions.HIT, PlayOptions.DOUBLE},
            CurriculumStage.ADD_SURRENDER: {PlayOptions.STAY, PlayOptions.HIT, PlayOptions.DOUBLE, PlayOptions.SURRENDER},
            CurriculumStage.ADD_SPLIT_ACES: {PlayOptions.STAY, PlayOptions.HIT, PlayOptions.DOUBLE, PlayOptions.SURRENDER, PlayOptions.SPLIT},
            CurriculumStage.FULL_ACTIONS: {PlayOptions.STAY, PlayOptions.HIT, PlayOptions.DOUBLE, PlayOptions.SURRENDER, PlayOptions.SPLIT}
        }
    
    def set_stage(self, stage: CurriculumStage):
        """Update the current curriculum stage"""
        self.stage = stage
        print(f"📚 Curriculum stage updated to: {stage.name}")
        print(f"   Available actions: {[action.name for action in self.stage_actions[stage]]}")
    
    def get_curriculum_mask(self, game_legal_mask: List[bool], hand_cards: List[int], is_first_decision: bool) -> List[bool]:
        """
        Apply curriculum masking on top of game legal actions.
        
        Args:
            game_legal_mask: Legal actions from game rules [STAY, HIT, DOUBLE, SPLIT, SURRENDER]
            hand_cards: Cards in the current hand
            is_first_decision: Whether this is the first decision for the hand
            
        Returns:
            Curriculum-filtered legal action mask
        """
        curriculum_mask = game_legal_mask.copy()
        available_actions = self.stage_actions[self.stage]
        
        # Disable actions not available in current stage
        # Map from PlayOptions enum to array indices (0-4)
        action_to_index = {
            PlayOptions.STAY: 0,
            PlayOptions.HIT: 1, 
            PlayOptions.DOUBLE: 2,
            PlayOptions.SPLIT: 3,
            PlayOptions.SURRENDER: 4
        }
        
        for action in PlayOptions:
            if action not in available_actions:
                idx = action_to_index[action]
                curriculum_mask[idx] = False
        
        # Special logic for split restrictions in intermediate stages
        if self.stage == CurriculumStage.ADD_SPLIT_ACES:
            # Only allow splitting Aces and Eights
            split_idx = action_to_index[PlayOptions.SPLIT]
            if game_legal_mask[split_idx] and is_first_decision:
                if len(hand_cards) == 2 and hand_cards[0] == hand_cards[1]:
                    # Check if it's Aces (1) or Eights (8)
                    card_value = hand_cards[0]
                    if card_value not in [1, 8]:  # Not Aces or Eights
                        curriculum_mask[split_idx] = False
        
        return curriculum_mask
    
    def get_stage_description(self) -> str:
        """Get human-readable description of current stage"""
        descriptions = {
            CurriculumStage.BASIC_ACTIONS: "Basic Actions: HIT, STAND, DOUBLE",
            CurriculumStage.ADD_SURRENDER: "Basic + SURRENDER",
            CurriculumStage.ADD_SPLIT_ACES: "Basic + SURRENDER + SPLIT (Aces & Eights only)",
            CurriculumStage.FULL_ACTIONS: "All Actions: Full Blackjack Rules"
        }
        return descriptions[self.stage]


@dataclass
class PerformanceMetrics:
    """Performance tracking for curriculum decisions"""
    
    episodes: int = 0
    total_reward: float = 0.0
    recent_rewards: List[float] = None
    win_rate: float = 0.0
    push_rate: float = 0.0
    loss_rate: float = 0.0
    avg_reward_per_episode: float = 0.0
    
    def __post_init__(self):
        if self.recent_rewards is None:
            self.recent_rewards = []
    
    def update(self, episode_reward: float, episode_result: str):
        """Update metrics with new episode"""
        self.episodes += 1
        self.total_reward += episode_reward
        self.recent_rewards.append(episode_reward)
        
        # Keep only recent rewards for performance calculation
        max_history = 10000
        if len(self.recent_rewards) > max_history:
            self.recent_rewards = self.recent_rewards[-max_history:]
        
        # Update rates based on result
        if episode_result in ["WIN", "BLACKJACK", "DOUBLEWIN"]:
            self.win_rate = self._update_rate(self.win_rate, 1.0)
        elif episode_result == "PUSH":
            self.push_rate = self._update_rate(self.push_rate, 1.0)
        else:  # LOST, DOUBLELOSS, SURRENDER
            self.loss_rate = self._update_rate(self.loss_rate, 1.0)
        
        self.avg_reward_per_episode = np.mean(self.recent_rewards) if self.recent_rewards else 0.0
    
    def _update_rate(self, current_rate: float, new_value: float, alpha: float = 0.001) -> float:
        """Exponential moving average update"""
        return current_rate * (1 - alpha) + new_value * alpha
    
    def get_performance_stability(self, window: int = 1000) -> float:
        """Calculate performance stability (lower = more stable)"""
        if len(self.recent_rewards) < window:
            return float('inf')  # Not enough data
        
        recent_window = self.recent_rewards[-window:]
        return np.std(recent_window)
    
    def get_recent_performance(self, window: int = 1000) -> float:
        """Get average performance over recent window"""
        if len(self.recent_rewards) < window:
            return self.avg_reward_per_episode
        
        return np.mean(self.recent_rewards[-window:])


class CurriculumManager:
    """
    Manages the curriculum learning process.
    
    Decides when to transition between stages based on performance
    metrics and stability criteria.
    """
    
    def __init__(self, config: CurriculumConfig = None, rules: BlackjackRules = V1_RULES):
        self.config = config or CurriculumConfig()
        self.rules = rules
        self.mask_manager = ActionMaskManager(rules)
        self.metrics = PerformanceMetrics()
        
        self.stage_start_episode = 0
        self.stage_history: List[Dict[str, Any]] = []
        self.transition_log: List[str] = []
        
        print(f"🎓 Curriculum Manager initialized")
        print(f"   Starting stage: {self.mask_manager.stage.name}")
        print(f"   {self.mask_manager.get_stage_description()}")
    
    def should_advance_stage(self) -> bool:
        """
        Determine if we should advance to the next curriculum stage.
        
        Criteria:
        1. Minimum episodes completed
        2. Performance has stabilized
        3. Performance threshold met
        """
        episodes_in_stage = self.metrics.episodes - self.stage_start_episode
        
        # Check minimum episodes
        if episodes_in_stage < self.config.min_episodes_per_stage:
            return False
        
        # Check performance stability
        stability = self.metrics.get_performance_stability(self.config.performance_window)
        if stability > self.config.stability_threshold:
            return False
        
        # Check performance threshold for current stage
        recent_performance = self.metrics.get_recent_performance(self.config.performance_window)
        
        threshold_map = {
            CurriculumStage.BASIC_ACTIONS: self.config.basic_actions_threshold,
            CurriculumStage.ADD_SURRENDER: self.config.surrender_threshold,
            CurriculumStage.ADD_SPLIT_ACES: self.config.split_aces_threshold,
        }
        
        current_stage = self.mask_manager.stage
        if current_stage in threshold_map:
            required_threshold = threshold_map[current_stage]
            if recent_performance < required_threshold:
                return False
        
        return True
    
    def advance_stage(self) -> bool:
        """
        Advance to the next curriculum stage if possible.
        
        Returns:
            True if stage was advanced, False if already at final stage
        """
        current_stage = self.mask_manager.stage
        
        # Record current stage performance
        episodes_in_stage = self.metrics.episodes - self.stage_start_episode
        recent_performance = self.metrics.get_recent_performance(self.config.performance_window)
        stability = self.metrics.get_performance_stability(self.config.performance_window)
        
        stage_record = {
            'stage': current_stage.name,
            'episodes': episodes_in_stage,
            'performance': recent_performance,
            'stability': stability,
            'completed_at_episode': self.metrics.episodes
        }
        self.stage_history.append(stage_record)
        
        # Determine next stage
        next_stage_map = {
            CurriculumStage.BASIC_ACTIONS: CurriculumStage.ADD_SURRENDER,
            CurriculumStage.ADD_SURRENDER: CurriculumStage.ADD_SPLIT_ACES,
            CurriculumStage.ADD_SPLIT_ACES: CurriculumStage.FULL_ACTIONS,
            CurriculumStage.FULL_ACTIONS: None  # Already at final stage
        }
        
        next_stage = next_stage_map.get(current_stage)
        if next_stage is None:
            return False  # Already at final stage
        
        # Advance stage
        self.mask_manager.set_stage(next_stage)
        self.stage_start_episode = self.metrics.episodes
        
        # Log transition
        transition_msg = (
            f"📈 Stage Transition: {current_stage.name} → {next_stage.name} "
            f"(Episode {self.metrics.episodes}, Performance: {recent_performance:.4f})"
        )
        self.transition_log.append(transition_msg)
        print(transition_msg)
        
        return True
    
    def update_performance(self, episode_reward: float, episode_result: str) -> bool:
        """
        Update performance metrics and check for stage advancement.
        
        Returns:
            True if stage was advanced, False otherwise
        """
        self.metrics.update(episode_reward, episode_result)
        
        # Check for stage advancement
        if self.should_advance_stage():
            return self.advance_stage()
        
        return False
    
    def get_curriculum_mask(self, game_legal_mask: List[bool], hand_cards: List[int], is_first_decision: bool) -> List[bool]:
        """Get curriculum-filtered action mask"""
        return self.mask_manager.get_curriculum_mask(game_legal_mask, hand_cards, is_first_decision)
    
    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage"""
        return self.mask_manager.stage
    
    def get_stage_progress(self) -> Dict[str, Any]:
        """Get progress information for current stage"""
        episodes_in_stage = self.metrics.episodes - self.stage_start_episode
        recent_performance = self.metrics.get_recent_performance(min(1000, episodes_in_stage))
        stability = self.metrics.get_performance_stability(min(1000, episodes_in_stage))
        
        return {
            'stage': self.mask_manager.stage.name,
            'stage_description': self.mask_manager.get_stage_description(),
            'episodes_in_stage': episodes_in_stage,
            'min_episodes_required': self.config.min_episodes_per_stage,
            'recent_performance': recent_performance,
            'stability': stability,
            'stability_threshold': self.config.stability_threshold,
            'ready_to_advance': self.should_advance_stage(),
            'is_final_stage': self.mask_manager.stage == CurriculumStage.FULL_ACTIONS
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive curriculum summary"""
        return {
            'current_stage': self.get_stage_progress(),
            'total_episodes': self.metrics.episodes,
            'overall_performance': self.metrics.avg_reward_per_episode,
            'stage_history': self.stage_history,
            'transition_log': self.transition_log,
            'config': {
                'min_episodes_per_stage': self.config.min_episodes_per_stage,
                'stability_threshold': self.config.stability_threshold,
                'performance_thresholds': {
                    'basic_actions': self.config.basic_actions_threshold,
                    'surrender': self.config.surrender_threshold,
                    'split_aces': self.config.split_aces_threshold
                }
            }
        }
    
    def save_progress(self, filepath: str):
        """Save curriculum progress to file"""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def is_complete(self) -> bool:
        """Check if curriculum is complete (reached final stage)"""
        return self.mask_manager.stage == CurriculumStage.FULL_ACTIONS


def create_curriculum_config(
    quick_mode: bool = False,
    performance_oriented: bool = True
) -> CurriculumConfig:
    """
    Create curriculum configuration for different training scenarios.
    
    Args:
        quick_mode: If True, use smaller episode requirements for faster testing
        performance_oriented: If True, use stricter thresholds for better final performance
    """
    
    if quick_mode:
        # For testing and development
        return CurriculumConfig(
            min_episodes_per_stage=1000,
            performance_window=500,
            stability_threshold=0.05,
            basic_actions_threshold=-0.2,
            surrender_threshold=-0.1,
            split_aces_threshold=-0.05
        )
    
    elif performance_oriented:
        # For high-quality training
        return CurriculumConfig(
            min_episodes_per_stage=100000,
            performance_window=20000,
            stability_threshold=0.005,
            basic_actions_threshold=-0.05,
            surrender_threshold=-0.02,
            split_aces_threshold=0.01
        )
    
    else:
        # Default balanced configuration
        return CurriculumConfig()
