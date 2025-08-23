"""
Integrated Training for Blackjack with Bet Sizing (Phase 5)

Combines the trained playing policy from Phase 4 with the new bet sizing policy.
Trains bet sizing while keeping the playing policy fixed.
"""

import gymnasium as gym
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import time
import json
from pathlib import Path

from .bet_sizing import BetSizingAgent, BetSizingState, BetSizingConfig, create_bet_sizing_config
from .rainbow_dqn import RainbowDQNAgent
from .device_utils import device_manager, get_device, print_device_status
from .config import V1_RULES
from .state_representation import StateExtractor


class IntegratedBlackjackTrainer:
    """
    Trainer that combines fixed playing policy with trainable bet sizing policy.
    
    The playing policy (from Phase 4) is frozen and used to make playing decisions.
    The bet sizing policy is trained to optimize bankroll growth.
    """
    
    def __init__(self,
                 play_policy_path: str,
                 bet_sizing_config: BetSizingConfig = None,
                 rules = V1_RULES,
                 initial_bankroll: float = 1000.0,
                 save_dir: str = "phase5_results"):
        
        self.rules = rules
        self.initial_bankroll = initial_bankroll
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Load frozen playing policy
        print("🎯 Loading trained playing policy...")
        self.play_agent = RainbowDQNAgent(
            input_dim=12,
            hidden_dim=256,
            num_actions=5
        )
        
        try:
            checkpoint = torch.load(play_policy_path, map_location=get_device())
            if 'policy_state_dict' in checkpoint:
                self.play_agent.online_net.load_state_dict(checkpoint['policy_state_dict'])
            else:
                self.play_agent.online_net.load_state_dict(checkpoint)
            self.play_agent.online_net.eval()  # Freeze in evaluation mode
            print(f"   ✅ Playing policy loaded from {play_policy_path}")
        except Exception as e:
            print(f"   ❌ Error loading playing policy: {e}")
            print(f"   ℹ️  Will train from random playing policy")
        
        # Create bet sizing agent
        self.bet_config = bet_sizing_config or create_bet_sizing_config()
        self.bet_config.initial_bankroll = initial_bankroll
        self.bet_agent = BetSizingAgent(self.bet_config)
        
        # Create environment
        self.env = gym.make("BlackjackSim-v0", rules=self.rules, use_compact_state=True)
        self.state_extractor = StateExtractor(self.rules)
        
        # Tracking variables
        self.current_bankroll = initial_bankroll
        self.episode_count = 0
        self.bet_trajectories = []
        self.performance_history = []
        
        print(f"🎰 Integrated Blackjack Trainer initialized")
        print(f"   Initial bankroll: ${initial_bankroll:,.0f}")
        print(f"   Bet sizes: {self.bet_config.bet_sizes}")
        print(f"   Save directory: {save_dir}")
        print_device_status()
    
    def play_episode(self, bet_size: float) -> Dict[str, Any]:
        """
        Play a single episode with fixed bet size.
        
        Args:
            bet_size: Bet size for this episode
            
        Returns:
            Episode result with reward and metadata
        """
        obs, info = self.env.reset()
        episode_reward = 0.0
        step_count = 0
        
        # Extract initial state
        blackjack_state = self.env.unwrapped._get_blackjack_state()
        
        while True:
            # Get state representation for playing policy
            state_array = blackjack_state.to_array()
            state_tensor = device_manager.tensor(state_array, dtype=torch.float32).unsqueeze(0)
            
            # Get legal actions mask
            legal_actions = info.get('legal_actions', [True] * 5)
            mask = device_manager.tensor(legal_actions, dtype=torch.float32)
            
            # Select action using frozen playing policy
            with torch.no_grad():
                action = self.play_agent.get_action(state_tensor, mask, epsilon=0.0)
            
            # Take action
            obs, reward, terminated, truncated, info = self.env.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
                
            # Update state for next iteration
            blackjack_state = self.env.unwrapped._get_blackjack_state()
        
        # Calculate bankroll change (reward * bet_size)
        bankroll_change = episode_reward * bet_size
        
        return {
            'episode_reward': episode_reward,
            'bankroll_change': bankroll_change,
            'bet_size': bet_size,
            'step_count': step_count,
            'final_result': info.get('result', 'UNKNOWN')
        }
    
    def collect_bet_trajectory(self, num_episodes: int = 100) -> List[Dict[str, Any]]:
        """
        Collect a trajectory of bet sizing decisions and outcomes.
        
        Args:
            num_episodes: Number of episodes to collect
            
        Returns:
            List of trajectory data for bet sizing training
        """
        trajectory = []
        episode_rewards = deque(maxlen=50)  # For recent performance tracking
        
        for episode in range(num_episodes):
            # Get current game state for bet sizing
            obs, info = self.env.reset()
            blackjack_state = self.env.unwrapped._get_blackjack_state()
            
            # Extract bet sizing state
            true_count = getattr(blackjack_state, 'true_count', 0.0)
            cards_remaining = getattr(blackjack_state, 'cards_remaining', 208)  # 4 decks default
            total_cards = 208  # 4 deck shoe
            recent_performance = np.mean(episode_rewards) if episode_rewards else 0.0
            
            bet_state = BetSizingState.from_game_state(
                true_count=true_count,
                cards_remaining=cards_remaining,
                total_cards=total_cards,
                current_bankroll=self.current_bankroll,
                initial_bankroll=self.initial_bankroll,
                recent_performance=recent_performance
            )
            
            # Select bet size
            bet_size, bet_info = self.bet_agent.select_bet_size(bet_state)
            
            # Play episode with selected bet size
            episode_result = self.play_episode(bet_size)
            
            # Update bankroll
            self.current_bankroll += episode_result['bankroll_change']
            episode_rewards.append(episode_result['episode_reward'])
            
            # Calculate reward for bet sizing (focus on bankroll growth)
            if self.current_bankroll > 0:
                # Reward based on bankroll growth rate
                bankroll_ratio = self.current_bankroll / self.initial_bankroll
                bet_reward = np.log(bankroll_ratio) * 100  # Log bankroll growth
            else:
                bet_reward = -100  # Penalty for going broke
            
            # Store trajectory data
            trajectory_data = {
                'state': bet_state,
                'action_idx': bet_info['action_idx'],
                'log_prob': bet_info['log_prob'],
                'value': bet_info['value'],
                'reward': bet_reward,
                'bankroll_change': episode_result['bankroll_change'],
                'episode_reward': episode_result['episode_reward'],
                'current_bankroll': self.current_bankroll
            }
            
            trajectory.append(trajectory_data)
            
            self.episode_count += 1
            
            # Progress update
            if episode % 25 == 0 and episode > 0:
                avg_bet = np.mean([t['state'].bankroll_ratio for t in trajectory[-25:]])
                print(f"  Episode {episode}: Bankroll=${self.current_bankroll:.0f}, "
                      f"Avg bet ratio={avg_bet:.3f}")
        
        return trajectory
    
    def train_bet_sizing(self, 
                        num_iterations: int = 1000,
                        episodes_per_iteration: int = 100,
                        eval_frequency: int = 100) -> Dict[str, Any]:
        """
        Train bet sizing policy while keeping playing policy fixed.
        
        Args:
            num_iterations: Number of training iterations
            episodes_per_iteration: Episodes to collect per iteration
            eval_frequency: How often to run evaluation
            
        Returns:
            Training summary
        """
        print(f"\n💰 Starting Phase 5 Bet Sizing Training")
        print(f"   Iterations: {num_iterations}")
        print(f"   Episodes per iteration: {episodes_per_iteration}")
        print(f"   Total episodes: {num_iterations * episodes_per_iteration:,}")
        print()
        
        start_time = time.time()
        best_bankroll = self.current_bankroll
        
        for iteration in range(num_iterations):
            iteration_start = time.time()
            
            # Collect trajectory
            trajectory = self.collect_bet_trajectory(episodes_per_iteration)
            
            # Update bet sizing policy
            training_metrics = self.bet_agent.update_policy(trajectory)
            
            # Track performance
            iteration_bankroll = self.current_bankroll
            iteration_profit = iteration_bankroll - self.initial_bankroll
            roi = (iteration_profit / self.initial_bankroll) * 100
            
            iteration_time = time.time() - iteration_start
            
            print(f"Iteration {iteration + 1}/{num_iterations}:")
            print(f"  Bankroll: ${iteration_bankroll:,.0f} (ROI: {roi:+.1f}%)")
            print(f"  Policy Loss: {training_metrics.get('policy_loss', 0):.4f}")
            print(f"  Value Loss: {training_metrics.get('value_loss', 0):.4f}")
            print(f"  Entropy: {training_metrics.get('entropy', 0):.4f}")
            print(f"  Time: {iteration_time:.1f}s")
            
            # Save best model
            if iteration_bankroll > best_bankroll:
                best_bankroll = iteration_bankroll
                self.bet_agent.save_model(self.save_dir / "best_bet_sizing_model.pth")
                print(f"  💾 Best model saved (${best_bankroll:,.0f})")
            
            # Evaluation
            if (iteration + 1) % eval_frequency == 0:
                eval_results = self.evaluate_bet_sizing(num_episodes=1000)
                print(f"  📊 Evaluation: Bankroll=${eval_results['final_bankroll']:,.0f}, "
                      f"ROI={eval_results['roi']:.1f}%, Sharpe={eval_results['sharpe_ratio']:.2f}")
            
            print()
            
            # Save progress
            self.save_training_progress(iteration, training_metrics, iteration_bankroll)
            
            # Early stopping if bankroll gets too low
            if self.current_bankroll < self.initial_bankroll * 0.1:
                print(f"⚠️  Stopping training due to low bankroll (${self.current_bankroll:.0f})")
                break
        
        total_time = time.time() - start_time
        final_profit = self.current_bankroll - self.initial_bankroll
        final_roi = (final_profit / self.initial_bankroll) * 100
        
        print(f"\n🎉 Bet Sizing Training Complete!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Final bankroll: ${self.current_bankroll:,.0f}")
        print(f"   Total profit: ${final_profit:+,.0f}")
        print(f"   ROI: {final_roi:+.1f}%")
        print(f"   Episodes trained: {self.episode_count:,}")
        
        # Final evaluation
        final_eval = self.evaluate_bet_sizing(num_episodes=5000)
        
        return {
            'training_time_minutes': total_time / 60,
            'final_bankroll': self.current_bankroll,
            'total_profit': final_profit,
            'roi_percent': final_roi,
            'episodes_trained': self.episode_count,
            'final_evaluation': final_eval,
            'best_bankroll': best_bankroll
        }
    
    def evaluate_bet_sizing(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Evaluate bet sizing policy performance.
        
        Args:
            num_episodes: Number of episodes for evaluation
            
        Returns:
            Evaluation metrics
        """
        print(f"📊 Evaluating bet sizing policy ({num_episodes:,} episodes)...")
        
        # Save current state
        original_bankroll = self.current_bankroll
        
        # Reset for evaluation
        eval_bankroll = self.initial_bankroll
        episode_profits = []
        bet_sizes_used = []
        
        for episode in range(num_episodes):
            # Get bet sizing state
            obs, info = self.env.reset()
            blackjack_state = self.env.unwrapped._get_blackjack_state()
            
            true_count = getattr(blackjack_state, 'true_count', 0.0)
            recent_perf = np.mean(episode_profits[-50:]) if episode_profits else 0.0
            
            bet_state = BetSizingState.from_game_state(
                true_count=true_count,
                cards_remaining=208,
                total_cards=208,
                current_bankroll=eval_bankroll,
                initial_bankroll=self.initial_bankroll,
                recent_performance=recent_perf
            )
            
            # Select bet size (deterministic for evaluation)
            bet_size, _ = self.bet_agent.select_bet_size(bet_state, deterministic=True)
            bet_sizes_used.append(bet_size)
            
            # Play episode
            episode_result = self.play_episode(bet_size)
            profit = episode_result['bankroll_change']
            episode_profits.append(profit)
            eval_bankroll += profit
        
        # Calculate metrics
        total_profit = eval_bankroll - self.initial_bankroll
        roi = (total_profit / self.initial_bankroll) * 100
        avg_profit_per_episode = np.mean(episode_profits)
        profit_std = np.std(episode_profits)
        
        # Sharpe ratio (risk-adjusted return)
        sharpe_ratio = avg_profit_per_episode / profit_std if profit_std > 0 else 0
        
        # Bet size statistics
        avg_bet_size = np.mean(bet_sizes_used)
        bet_size_std = np.std(bet_sizes_used)
        
        # Restore original state
        self.current_bankroll = original_bankroll
        
        eval_results = {
            'final_bankroll': eval_bankroll,
            'total_profit': total_profit,
            'roi': roi,
            'avg_profit_per_episode': avg_profit_per_episode,
            'profit_std': profit_std,
            'sharpe_ratio': sharpe_ratio,
            'avg_bet_size': avg_bet_size,
            'bet_size_std': bet_size_std,
            'episodes_evaluated': num_episodes
        }
        
        print(f"   Final bankroll: ${eval_bankroll:,.0f}")
        print(f"   ROI: {roi:+.1f}%")
        print(f"   Sharpe ratio: {sharpe_ratio:.2f}")
        print(f"   Avg bet size: {avg_bet_size:.1f} units")
        
        return eval_results
    
    def save_training_progress(self, iteration: int, metrics: Dict, bankroll: float):
        """Save training progress to file"""
        progress_data = {
            'iteration': iteration,
            'bankroll': bankroll,
            'roi_percent': ((bankroll - self.initial_bankroll) / self.initial_bankroll) * 100,
            'episodes_completed': self.episode_count,
            'training_metrics': metrics,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        progress_file = self.save_dir / "training_progress.json"
        
        # Load existing progress
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                all_progress = json.load(f)
        else:
            all_progress = []
        
        all_progress.append(progress_data)
        
        # Save updated progress
        with open(progress_file, 'w') as f:
            json.dump(all_progress, f, indent=2)


def train_phase5_bet_sizing(
    play_policy_path: str = "curriculum_training_results/best_model_stage_ADD_SPLIT_ACES.pth",
    config_type: str = "moderate",
    initial_bankroll: float = 1000.0,
    num_iterations: int = 500
) -> Dict[str, Any]:
    """
    Convenience function to train Phase 5 bet sizing.
    
    Args:
        play_policy_path: Path to trained playing policy from Phase 4
        config_type: "conservative", "moderate", or "aggressive"
        initial_bankroll: Starting bankroll
        num_iterations: Number of training iterations
        
    Returns:
        Training results
    """
    
    # Create bet sizing configuration
    if config_type == "conservative":
        bet_config = create_bet_sizing_config(conservative=True, bet_spread="small")
    elif config_type == "aggressive":
        bet_config = create_bet_sizing_config(conservative=False, bet_spread="aggressive")
    else:  # moderate
        bet_config = create_bet_sizing_config(conservative=True, bet_spread="moderate")
    
    # Create trainer
    trainer = IntegratedBlackjackTrainer(
        play_policy_path=play_policy_path,
        bet_sizing_config=bet_config,
        initial_bankroll=initial_bankroll
    )
    
    # Train
    results = trainer.train_bet_sizing(
        num_iterations=num_iterations,
        episodes_per_iteration=100,
        eval_frequency=50
    )
    
    return results
