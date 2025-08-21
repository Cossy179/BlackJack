"""
Curriculum Training Loop for Blackjack Rainbow DQN (Phase 4)

Integrates curriculum learning with Rainbow DQN training.
Progressively unlocks actions while maintaining the same network.
"""

import gymnasium as gym
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path
import json

from .curriculum import CurriculumManager, CurriculumConfig, CurriculumStage
from .rainbow_dqn import RainbowDQNAgent
from .device_utils import device_manager, to_device, get_device, print_device_status
from .config import V1_RULES
from .state_representation import BlackjackState, StateExtractor


class CurriculumTrainer:
    """
    Main trainer that combines Rainbow DQN with curriculum learning.
    
    Manages the training loop, environment interaction, and curriculum progression.
    """
    
    def __init__(
        self,
        env_id: str = "BlackjackSim-v0",
        curriculum_config: CurriculumConfig = None,
        agent_config: Dict[str, Any] = None,
        rules = V1_RULES,
        save_dir: str = "curriculum_training_results",
        device: str = None
    ):
        
        # Set device
        if device is not None:
            device_manager.set_device(device)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Create environment
        self.env = gym.make(env_id, rules=rules, use_compact_state=True)
        self.state_extractor = StateExtractor(rules)
        
        # Initialize curriculum manager
        self.curriculum = CurriculumManager(curriculum_config or CurriculumConfig(), rules)
        
        # Initialize Rainbow DQN agent
        self.agent = self._create_agent(agent_config or {})
        
        # Training metrics
        self.episode_rewards = deque(maxlen=10000)
        self.episode_lengths = deque(maxlen=10000)
        self.training_metrics = []
        self.evaluation_results = []
        
        # Training state
        self.total_episodes = 0
        self.total_steps = 0
        self.start_time = time.time()
        
        print(f"🎓 Curriculum Trainer initialized")
        print(f"   Environment: {env_id}")
        print(f"   Device: {get_device()}")
        print(f"   Save directory: {self.save_dir}")
        print_device_status()
    
    def _create_agent(self, config: Dict[str, Any]) -> RainbowDQNAgent:
        """Create Rainbow DQN agent with configuration"""
        
        default_config = {
            'input_dim': 12,
            'hidden_dim': 512,
            'num_actions': 5,
            'lr': 1e-4,
            'gamma': 0.99,
            'n_step': 3,
            'buffer_capacity': 1000000,
            'batch_size': 512,
            'target_update_freq': 1000,
            'use_noisy': True
        }
        
        # Override with curriculum-specific learning rates
        current_stage = self.curriculum.get_current_stage()
        if hasattr(self.curriculum.config, 'learning_rates'):
            default_config['lr'] = self.curriculum.config.learning_rates[current_stage]
        if hasattr(self.curriculum.config, 'batch_sizes'):
            default_config['batch_size'] = self.curriculum.config.batch_sizes[current_stage]
        
        # Apply user config
        agent_config = {**default_config, **config}
        
        return RainbowDQNAgent(**agent_config)
    
    def _apply_curriculum_mask(self, legal_actions: List[bool], blackjack_state) -> torch.Tensor:
        """Apply curriculum masking to legal actions"""
        
        # Handle both dict and BlackjackState object
        if isinstance(blackjack_state, dict):
            player_total = blackjack_state.get('player_total', 10)
            num_cards = blackjack_state.get('num_cards', 2)
        else:
            player_total = blackjack_state.player_total
            num_cards = blackjack_state.num_cards
        
        # Get curriculum-filtered mask
        curriculum_mask = self.curriculum.get_curriculum_mask(
            legal_actions,
            [player_total],  # Simplified for curriculum
            num_cards == 2   # is_first_decision
        )
        
        return device_manager.tensor(curriculum_mask, dtype=torch.float32)
    
    def train_episode(self) -> Dict[str, float]:
        """Train for one episode"""
        
        # Reset environment
        obs, info = self.env.reset()
        state_tensor = to_device(torch.tensor(obs, dtype=torch.float32))
        
        episode_reward = 0.0
        episode_steps = 0
        done = False
        
        while not done:
            # Get legal actions and apply curriculum mask
            legal_actions = info['legal_actions']
            blackjack_state = info['blackjack_state']
            
            curriculum_legal_actions = self._apply_curriculum_mask(legal_actions, blackjack_state)
            
            # Select action
            action = self.agent.get_action(state_tensor, curriculum_legal_actions)
            
            # Take step
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            done = terminated or truncated
            
            next_state_tensor = to_device(torch.tensor(next_obs, dtype=torch.float32))
            
            # Get next legal actions with curriculum mask
            if not done:
                next_legal_actions = self._apply_curriculum_mask(
                    next_info['legal_actions'],
                    next_info['blackjack_state']
                )
            else:
                next_legal_actions = device_manager.zeros(5)
            
            # Store experience
            self.agent.store_experience(
                state_tensor,
                action,
                reward,
                next_state_tensor,
                done,
                curriculum_legal_actions,
                next_legal_actions
            )
            
            # Update state
            state_tensor = next_state_tensor
            info = next_info
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # Training step
            if len(self.agent.replay_buffer) > self.agent.batch_size:
                train_metrics = self.agent.train_step()
                if train_metrics:
                    self.training_metrics.append({
                        'step': self.total_steps,
                        'episode': self.total_episodes,
                        **train_metrics
                    })
        
        # Update curriculum
        final_result = info.get('result', 'UNKNOWN')
        stage_advanced = self.curriculum.update_performance(episode_reward, final_result)
        
        # Update agent learning rate if stage advanced
        if stage_advanced:
            self._update_agent_for_new_stage()
        
        # Record episode metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_steps)
        self.total_episodes += 1
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'stage_advanced': stage_advanced,
            'current_stage': self.curriculum.get_current_stage().name
        }
    
    def _update_agent_for_new_stage(self):
        """Update agent configuration for new curriculum stage"""
        current_stage = self.curriculum.get_current_stage()
        
        # Update learning rate
        if hasattr(self.curriculum.config, 'learning_rates'):
            new_lr = self.curriculum.config.learning_rates[current_stage]
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"📚 Updated learning rate to {new_lr:.6f} for stage {current_stage.name}")
        
        # Update batch size (requires creating new agent - for now just log)
        if hasattr(self.curriculum.config, 'batch_sizes'):
            new_batch_size = self.curriculum.config.batch_sizes[current_stage]
            print(f"📚 Batch size for stage {current_stage.name}: {new_batch_size}")
    
    def evaluate(self, num_episodes: int = 1000, deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate current policy.
        
        Args:
            num_episodes: Number of episodes to evaluate
            deterministic: If True, use deterministic policy (no noise/exploration)
            
        Returns:
            Evaluation metrics
        """
        print(f"🎯 Evaluating policy for {num_episodes} episodes...")
        
        # Temporarily disable training mode for deterministic evaluation
        if deterministic:
            self.agent.online_net.eval()
        
        eval_rewards = []
        eval_results = {'WIN': 0, 'LOSS': 0, 'PUSH': 0, 'BLACKJACK': 0, 'SURRENDER': 0}
        action_counts = {i: 0 for i in range(5)}
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            state_tensor = to_device(torch.tensor(obs, dtype=torch.float32))
            
            episode_reward = 0.0
            done = False
            
            while not done:
                # Get legal actions with curriculum mask
                legal_actions = info['legal_actions']
                blackjack_state = info['blackjack_state']
                curriculum_legal_actions = self._apply_curriculum_mask(legal_actions, blackjack_state)
                
                # Select action (deterministic)
                with torch.no_grad():
                    action = self.agent.get_action(state_tensor, curriculum_legal_actions, epsilon=0.0)
                
                action_counts[action] += 1
                
                # Take step
                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                done = terminated or truncated
                
                state_tensor = to_device(torch.tensor(next_obs, dtype=torch.float32))
                info = next_info
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            result = info.get('result', 'UNKNOWN')
            if result in eval_results:
                eval_results[result] += 1
            elif result in ['LOST', 'DOUBLELOSS']:
                eval_results['LOSS'] += 1
            elif result in ['WIN', 'DOUBLEWIN']:
                eval_results['WIN'] += 1
        
        # Re-enable training mode
        if deterministic:
            self.agent.online_net.train()
        
        # Calculate metrics
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        win_rate = eval_results['WIN'] / num_episodes
        push_rate = eval_results['PUSH'] / num_episodes
        loss_rate = eval_results['LOSS'] / num_episodes
        
        action_names = ['STAND', 'HIT', 'DOUBLE', 'SPLIT', 'SURRENDER']
        total_actions = sum(action_counts.values())
        action_distribution = {action_names[i]: count/total_actions for i, count in action_counts.items()}
        
        results = {
            'num_episodes': num_episodes,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'win_rate': win_rate,
            'push_rate': push_rate,
            'loss_rate': loss_rate,
            'action_distribution': action_distribution,
            'curriculum_stage': self.curriculum.get_current_stage().name,
            'total_episodes_trained': self.total_episodes
        }
        
        print(f"📊 Evaluation Results:")
        print(f"   Mean Reward: {mean_reward:.4f} ± {std_reward:.4f}")
        print(f"   Win Rate: {win_rate:.3f}, Push Rate: {push_rate:.3f}, Loss Rate: {loss_rate:.3f}")
        print(f"   Stage: {results['curriculum_stage']}")
        
        self.evaluation_results.append(results)
        return results
    
    def train(
        self,
        max_episodes: int = 1000000,
        eval_frequency: int = 10000,
        save_frequency: int = 50000,
        target_performance: float = 0.1,
        patience: int = 100000
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            max_episodes: Maximum episodes to train
            eval_frequency: How often to evaluate
            save_frequency: How often to save checkpoints
            target_performance: Target performance to stop training
            patience: Episodes without improvement before stopping
            
        Returns:
            Training summary
        """
        
        print(f"🚀 Starting curriculum training...")
        print(f"   Max episodes: {max_episodes:,}")
        print(f"   Eval frequency: {eval_frequency:,}")
        print(f"   Target performance: {target_performance:.4f}")
        
        best_performance = float('-inf')
        episodes_without_improvement = 0
        
        # Initial evaluation
        self.evaluate()
        
        training_start_time = time.time()
        
        try:
            for episode in range(max_episodes):
                # Train episode
                episode_metrics = self.train_episode()
                
                # Progress reporting
                if episode % 1000 == 0:
                    recent_reward = np.mean(list(self.episode_rewards)[-100:]) if self.episode_rewards else 0.0
                    progress = self.curriculum.get_stage_progress()
                    
                    print(f"Episode {episode:,}: "
                          f"Reward={recent_reward:.3f}, "
                          f"Stage={progress['stage']}, "
                          f"Buffer={len(self.agent.replay_buffer):,}")
                
                # Evaluation
                if episode % eval_frequency == 0 and episode > 0:
                    eval_results = self.evaluate()
                    current_performance = eval_results['mean_reward']
                    
                    # Check for improvement
                    if current_performance > best_performance:
                        best_performance = current_performance
                        episodes_without_improvement = 0
                        
                        # Save best model
                        self.save_checkpoint(f"best_model_stage_{self.curriculum.get_current_stage().name}.pth")
                    else:
                        episodes_without_improvement += eval_frequency
                    
                    # Check stopping criteria
                    if current_performance >= target_performance:
                        print(f"🎯 Target performance {target_performance:.4f} reached!")
                        break
                    
                    if episodes_without_improvement >= patience:
                        print(f"⏸️  No improvement for {patience:,} episodes. Stopping early.")
                        break
                
                # Save checkpoint
                if episode % save_frequency == 0 and episode > 0:
                    self.save_checkpoint(f"checkpoint_episode_{episode}.pth")
                
                # Check if curriculum is complete
                if self.curriculum.is_complete():
                    final_eval = self.evaluate(num_episodes=5000)
                    if final_eval['mean_reward'] >= target_performance:
                        print(f"🎓 Curriculum complete with target performance achieved!")
                        break
        
        except KeyboardInterrupt:
            print("\n⏸️  Training interrupted by user")
        
        training_time = time.time() - training_start_time
        
        # Final evaluation
        final_results = self.evaluate(num_episodes=5000)
        
        # Training summary
        summary = {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'training_time_hours': training_time / 3600,
            'final_performance': final_results['mean_reward'],
            'best_performance': best_performance,
            'curriculum_completed': self.curriculum.is_complete(),
            'final_stage': self.curriculum.get_current_stage().name,
            'curriculum_summary': self.curriculum.get_summary()
        }
        
        # Save final results
        self.save_results(summary)
        
        print(f"\n🎉 Training Complete!")
        print(f"   Episodes: {self.total_episodes:,}")
        print(f"   Time: {training_time/3600:.2f} hours")
        print(f"   Final Performance: {final_results['mean_reward']:.4f}")
        print(f"   Curriculum: {summary['curriculum_completed']}")
        
        return summary
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint_path = self.save_dir / filename
        
        # Save agent
        self.agent.save_model(str(checkpoint_path))
        
        # Save training state
        state_path = checkpoint_path.with_suffix('.json')
        training_state = {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'curriculum_summary': self.curriculum.get_summary(),
            'recent_rewards': list(self.episode_rewards)[-1000:],  # Last 1000
            'training_metrics': self.training_metrics[-1000:],    # Last 1000
            'evaluation_results': self.evaluation_results
        }
        
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2, default=str)
        
        print(f"💾 Checkpoint saved: {filename}")
    
    def save_results(self, summary: Dict[str, Any]):
        """Save final training results"""
        results_path = self.save_dir / "training_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save curriculum progress
        self.curriculum.save_progress(str(self.save_dir / "curriculum_progress.json"))
        
        # Create training plots
        self._create_training_plots()
        
        print(f"📊 Results saved to {self.save_dir}")
    
    def _create_training_plots(self):
        """Create training visualization plots"""
        if not self.evaluation_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance over time
        episodes = [r['total_episodes_trained'] for r in self.evaluation_results]
        rewards = [r['mean_reward'] for r in self.evaluation_results]
        
        axes[0, 0].plot(episodes, rewards, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Episodes')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].set_title('Performance Over Time')
        axes[0, 0].grid(True)
        
        # Win/Push/Loss rates
        win_rates = [r['win_rate'] for r in self.evaluation_results]
        push_rates = [r['push_rate'] for r in self.evaluation_results]
        loss_rates = [r['loss_rate'] for r in self.evaluation_results]
        
        axes[0, 1].plot(episodes, win_rates, 'g-', label='Win Rate')
        axes[0, 1].plot(episodes, push_rates, 'y-', label='Push Rate')
        axes[0, 1].plot(episodes, loss_rates, 'r-', label='Loss Rate')
        axes[0, 1].set_xlabel('Episodes')
        axes[0, 1].set_ylabel('Rate')
        axes[0, 1].set_title('Game Outcome Rates')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Training loss (if available)
        if self.training_metrics:
            train_steps = [m['step'] for m in self.training_metrics[-1000:]]
            train_losses = [m['loss'] for m in self.training_metrics[-1000:]]
            
            axes[1, 0].plot(train_steps, train_losses, 'r-', alpha=0.7)
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Training Loss (Recent)')
            axes[1, 0].grid(True)
        
        # Curriculum progress
        curriculum_summary = self.curriculum.get_summary()
        stage_history = curriculum_summary.get('stage_history', [])
        
        if stage_history:
            stage_names = [s['stage'] for s in stage_history]
            stage_episodes = [s['episodes'] for s in stage_history]
            stage_performance = [s['performance'] for s in stage_history]
            
            x_pos = np.arange(len(stage_names))
            bars = axes[1, 1].bar(x_pos, stage_performance, color='skyblue', alpha=0.7)
            
            # Add episode counts on bars
            for i, (bar, eps) in enumerate(zip(bars, stage_episodes)):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{eps:,}', ha='center', va='bottom', fontsize=8)
            
            axes[1, 1].set_xlabel('Curriculum Stage')
            axes[1, 1].set_ylabel('Final Performance')
            axes[1, 1].set_title('Performance by Curriculum Stage')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(stage_names, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "training_progress.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📈 Training plots saved")


def train_curriculum_agent(
    max_episodes: int = 500000,
    quick_mode: bool = False,
    config_overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Convenience function to train a curriculum agent.
    
    Args:
        max_episodes: Maximum episodes to train
        quick_mode: If True, use faster settings for testing
        config_overrides: Override default configurations
        
    Returns:
        Training summary
    """
    
    # Create configurations
    if quick_mode:
        curriculum_config = CurriculumConfig(
            min_episodes_per_stage=2000,
            performance_window=1000,
            stability_threshold=0.05,
            basic_actions_threshold=-0.3,
            surrender_threshold=-0.2,
            split_aces_threshold=-0.1
        )
        agent_config = {
            'hidden_dim': 256,
            'batch_size': 256,
            'buffer_capacity': 100000,
            'lr': 2e-4
        }
    else:
        curriculum_config = CurriculumConfig()
        agent_config = {}
    
    if config_overrides:
        agent_config.update(config_overrides)
    
    # Create trainer
    trainer = CurriculumTrainer(
        curriculum_config=curriculum_config,
        agent_config=agent_config
    )
    
    # Train
    results = trainer.train(
        max_episodes=max_episodes,
        eval_frequency=5000 if quick_mode else 10000,
        save_frequency=10000 if quick_mode else 50000
    )
    
    return results
