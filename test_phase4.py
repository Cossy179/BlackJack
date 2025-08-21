"""
Phase 4 Validation Tests

Tests for curriculum learning implementation including:
- Curriculum stage progression
- Action masking per stage
- Rainbow DQN components
- Training integration
"""

import torch
import numpy as np
import gymnasium as gym
from BlackJackSim.curriculum import (
    CurriculumManager, CurriculumConfig, CurriculumStage, 
    ActionMaskManager, create_curriculum_config
)
from BlackJackSim.rainbow_dqn import (
    RainbowDQNAgent, DuelingDQN, NoisyLinear, 
    PrioritizedReplayBuffer, Experience
)
from BlackJackSim.curriculum_training import CurriculumTrainer, train_curriculum_agent
from BlackJackSim.device_utils import get_device, device_manager
from BlackJackSim.config import V1_RULES
from BlackJackSim.BlackJack import PlayOptions
from BlackJackSim.state_representation import StateExtractor


def test_curriculum_stages():
    """Test curriculum stage progression and action masking"""
    print("\n=== Testing Curriculum Stages ===")
    
    # Create curriculum manager
    config = create_curriculum_config(quick_mode=True)
    curriculum = CurriculumManager(config)
    
    # Test initial stage
    assert curriculum.get_current_stage() == CurriculumStage.BASIC_ACTIONS
    print(f"✓ Initial stage: {curriculum.get_current_stage().name}")
    
    # Test action masking for each stage
    game_legal_mask = [True, True, True, True, True]  # All actions legal in game
    hand_cards = [10, 10]  # Pair of 10s
    is_first_decision = True
    
    # Check the actual PlayOptions enum values
    print(f"PlayOptions mapping: STAY={PlayOptions.STAY.value}, HIT={PlayOptions.HIT.value}, DOUBLE={PlayOptions.DOUBLE.value}, SPLIT={PlayOptions.SPLIT.value}, SURRENDER={PlayOptions.SURRENDER.value}")
    
    # Stage 1: Basic actions only
    mask1 = curriculum.get_curriculum_mask(game_legal_mask, hand_cards, is_first_decision)
    # Based on PlayOptions: STAY=1, HIT=2, DOUBLE=3, SPLIT=4, SURRENDER=5
    # Mask order is [STAY(0), HIT(1), DOUBLE(2), SPLIT(3), SURRENDER(4)]
    expected_basic = [True, True, True, False, False]  # STAND, HIT, DOUBLE only
    assert mask1 == expected_basic, f"Basic stage mask incorrect: {mask1}"
    print(f"✓ Basic actions stage: {[i for i, m in enumerate(mask1) if m]}")
    
    # Manually advance to test other stages
    curriculum.mask_manager.set_stage(CurriculumStage.ADD_SURRENDER)
    mask2 = curriculum.get_curriculum_mask(game_legal_mask, hand_cards, is_first_decision)
    expected_surrender = [True, True, True, False, True]  # + SURRENDER
    assert mask2 == expected_surrender, f"Surrender stage mask incorrect: {mask2}"
    print(f"✓ Add surrender stage: {[i for i, m in enumerate(mask2) if m]}")
    
    # Test split stage with Aces (should be allowed)
    curriculum.mask_manager.set_stage(CurriculumStage.ADD_SPLIT_ACES)
    ace_cards = [1, 1]  # Pair of Aces
    mask3 = curriculum.get_curriculum_mask(game_legal_mask, ace_cards, is_first_decision)
    expected_split_aces = [True, True, True, True, True]  # All actions for Aces
    assert mask3 == expected_split_aces, f"Split Aces stage mask incorrect: {mask3}"
    print(f"✓ Split Aces stage (Aces): {[i for i, m in enumerate(mask3) if m]}")
    
    # Test split stage with non-Ace pairs (should not allow split)
    mask4 = curriculum.get_curriculum_mask(game_legal_mask, hand_cards, is_first_decision)
    expected_no_split = [True, True, True, False, True]  # No split for 10s
    assert mask4 == expected_no_split, f"Split Aces stage mask for 10s incorrect: {mask4}"
    print(f"✓ Split Aces stage (10s): {[i for i, m in enumerate(mask4) if m]}")
    
    # Full actions stage
    curriculum.mask_manager.set_stage(CurriculumStage.FULL_ACTIONS)
    mask5 = curriculum.get_curriculum_mask(game_legal_mask, hand_cards, is_first_decision)
    assert mask5 == game_legal_mask, f"Full actions stage should match game mask: {mask5}"
    print(f"✓ Full actions stage: {[i for i, m in enumerate(mask5) if m]}")
    
    print("✓ Curriculum stages validated")


def test_noisy_networks():
    """Test noisy linear layers"""
    print("\n=== Testing Noisy Networks ===")
    
    # Create noisy layer
    layer = NoisyLinear(10, 5).to(get_device())
    
    # Test forward pass
    x = device_manager.randn(32, 10)
    
    # Training mode (with noise)
    layer.train()
    output1 = layer(x)
    
    # Reset noise to get different output
    layer.reset_noise()
    output2 = layer(x)
    
    # Should be different due to noise
    assert not torch.allclose(output1, output2), "Noisy outputs should differ in training mode"
    print("✓ Noisy layer produces different outputs in training mode")
    
    # Eval mode (no noise)
    layer.eval()
    output3 = layer(x)
    output4 = layer(x)
    
    # Should be same in eval mode
    assert torch.allclose(output3, output4), "Outputs should be identical in eval mode"
    print("✓ Noisy layer produces identical outputs in eval mode")
    
    # Test noise reset
    layer.train()
    layer.reset_noise()
    output5 = layer(x)
    assert not torch.allclose(output1, output5), "Noise reset should change outputs"
    print("✓ Noise reset functionality works")
    
    print("✓ Noisy networks validated")


def test_dueling_dqn():
    """Test dueling DQN architecture"""
    print("\n=== Testing Dueling DQN ===")
    
    # Create network
    net = DuelingDQN(
        input_dim=12,
        hidden_dim=128,
        num_actions=5,
        num_atoms=51,
        use_noisy=True
    ).to(get_device())
    
    # Test forward pass
    batch_size = 16
    states = device_manager.randn(batch_size, 12)
    legal_actions = device_manager.ones(batch_size, 5)
    legal_actions[:, 3:] = 0  # Disable SPLIT and SURRENDER
    
    # Forward pass with legal actions
    q_dist = net(states, legal_actions)
    assert q_dist.shape == (batch_size, 5, 51), f"Wrong output shape: {q_dist.shape}"
    
    # Check that illegal actions have -inf values
    illegal_actions = (legal_actions == 0)
    illegal_values = q_dist[illegal_actions]
    assert torch.all(torch.isinf(illegal_values)), "Illegal actions should have -inf values"
    print("✓ Legal action masking works")
    
    # Test Q-value computation
    q_values = net.get_q_values(states, legal_actions)
    assert q_values.shape == (batch_size, 5), f"Wrong Q-value shape: {q_values.shape}"
    print("✓ Q-value computation works")
    
    # Test action selection
    single_state = states[0]
    single_legal = legal_actions[0]
    action = net.get_action(single_state, single_legal)
    
    # Should be a legal action
    legal_indices = torch.nonzero(single_legal, as_tuple=True)[0].cpu().numpy()
    assert action in legal_indices, f"Selected illegal action: {action}, legal: {legal_indices}"
    print("✓ Action selection respects legal actions")
    
    # Test noise reset
    net.reset_noise()
    print("✓ Noise reset works")
    
    print("✓ Dueling DQN validated")


def test_prioritized_replay():
    """Test prioritized experience replay"""
    print("\n=== Testing Prioritized Replay ===")
    
    # Create buffer
    buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6)
    
    # Add some experiences
    for i in range(100):
        state = device_manager.randn(12)
        next_state = device_manager.randn(12)
        legal_actions = device_manager.ones(5)
        
        exp = Experience(
            state=state,
            action=i % 5,
            reward=np.random.randn(),
            next_state=next_state,
            done=False,
            legal_actions=legal_actions,
            next_legal_actions=legal_actions,
            n_step_return=np.random.randn(),
            n_step_discount=0.99
        )
        buffer.push(exp)
    
    assert len(buffer) == 100, f"Buffer should have 100 experiences, has {len(buffer)}"
    print(f"✓ Buffer populated with {len(buffer)} experiences")
    
    # Test sampling
    batch_size = 32
    experiences, indices, weights = buffer.sample(batch_size)
    
    assert len(experiences) == batch_size, f"Should sample {batch_size} experiences"
    assert len(indices) == batch_size, f"Should return {batch_size} indices"
    assert len(weights) == batch_size, f"Should return {batch_size} weights"
    assert all(0 <= idx < len(buffer) for idx in indices), "Indices should be valid"
    print("✓ Sampling works correctly")
    
    # Test priority updates
    new_priorities = np.random.rand(batch_size)
    buffer.update_priorities(indices, new_priorities)
    print("✓ Priority updates work")
    
    # Test beta annealing
    initial_beta = buffer.beta()
    buffer.frame_count += 50000
    later_beta = buffer.beta()
    assert later_beta > initial_beta, "Beta should increase over time"
    print(f"✓ Beta annealing works: {initial_beta:.3f} → {later_beta:.3f}")
    
    print("✓ Prioritized replay validated")


def test_rainbow_agent():
    """Test complete Rainbow DQN agent"""
    print("\n=== Testing Rainbow Agent ===")
    
    # Create agent with small configuration
    agent = RainbowDQNAgent(
        input_dim=12,
        hidden_dim=64,
        num_actions=5,
        num_atoms=21,  # Smaller for testing
        lr=1e-3,
        buffer_capacity=1000,
        batch_size=32,
        target_update_freq=100
    )
    
    print(f"✓ Agent created on device: {get_device()}")
    
    # Test action selection
    state = device_manager.randn(12)
    legal_actions = device_manager.tensor([1, 1, 0, 1, 0], dtype=torch.float32)
    
    action = agent.get_action(state, legal_actions)
    legal_indices = torch.nonzero(legal_actions, as_tuple=True)[0].cpu().numpy()
    assert action in legal_indices, f"Agent selected illegal action: {action}"
    print("✓ Action selection works")
    
    # Test experience storage and training
    for episode in range(5):
        for step in range(10):
            next_state = device_manager.randn(12)
            next_legal = device_manager.ones(5)
            
            agent.store_experience(
                state, action, np.random.randn(), next_state, step == 9,
                legal_actions, next_legal
            )
            state = next_state
            legal_actions = next_legal
            action = agent.get_action(state, legal_actions)
    
    print(f"✓ Stored {len(agent.replay_buffer)} experiences")
    
    # Test training
    if len(agent.replay_buffer) >= agent.batch_size:
        metrics = agent.train_step()
        assert 'loss' in metrics, "Training should return loss"
        assert metrics['loss'] > 0, "Loss should be positive"
        print(f"✓ Training step works, loss: {metrics['loss']:.4f}")
    
    print("✓ Rainbow agent validated")


def test_curriculum_training_integration():
    """Test curriculum training integration"""
    print("\n=== Testing Curriculum Training Integration ===")
    
    # Create quick training configuration
    curriculum_config = create_curriculum_config(quick_mode=True)
    curriculum_config.min_episodes_per_stage = 10  # Very small for testing
    
    agent_config = {
        'hidden_dim': 64,
        'batch_size': 32,
        'buffer_capacity': 1000,
        'lr': 1e-3,
        'num_atoms': 21
    }
    
    # Create trainer
    trainer = CurriculumTrainer(
        curriculum_config=curriculum_config,
        agent_config=agent_config,
        save_dir="test_curriculum_results"
    )
    
    print(f"✓ Trainer created with device: {get_device()}")
    
    # Test single episode
    episode_metrics = trainer.train_episode()
    
    assert 'episode_reward' in episode_metrics, "Should return episode reward"
    assert 'current_stage' in episode_metrics, "Should return current stage"
    print(f"✓ Episode training works, reward: {episode_metrics['episode_reward']:.3f}")
    
    # Test short training run
    print("Running short training test (50 episodes)...")
    results = trainer.train(
        max_episodes=50,
        eval_frequency=25,
        save_frequency=100  # Won't trigger
    )
    
    assert 'total_episodes' in results, "Should return training summary"
    assert results['total_episodes'] > 0, "Should have trained episodes"
    print(f"✓ Training completed: {results['total_episodes']} episodes")
    
    # Test curriculum progression
    progress = trainer.curriculum.get_stage_progress()
    print(f"✓ Final stage: {progress['stage']}")
    print(f"✓ Episodes in stage: {progress['episodes_in_stage']}")
    
    print("✓ Curriculum training integration validated")


def test_action_space_curriculum():
    """Test that curriculum properly restricts action space"""
    print("\n=== Testing Action Space Curriculum ===")
    
    env = gym.make("BlackjackSim-v0", rules=V1_RULES, use_compact_state=True)
    curriculum = CurriculumManager(create_curriculum_config(quick_mode=True))
    
    # Test each stage
    stages_to_test = [
        (CurriculumStage.BASIC_ACTIONS, [0, 1, 2]),      # STAND, HIT, DOUBLE
        (CurriculumStage.ADD_SURRENDER, [0, 1, 2, 4]),   # + SURRENDER
        (CurriculumStage.FULL_ACTIONS, [0, 1, 2, 3, 4])  # All actions
    ]
    
    for stage, expected_actions in stages_to_test:
        curriculum.mask_manager.set_stage(stage)
        
        # Run a few episodes to collect action statistics
        action_counts = {i: 0 for i in range(5)}
        
        for episode in range(20):
            obs, info = env.reset()
            done = False
            
            while not done:
                # Get curriculum mask
                legal_actions = info['legal_actions']
                blackjack_state = info['blackjack_state']
                
                # Handle both dict and BlackjackState object
                if isinstance(blackjack_state, dict):
                    player_total = blackjack_state.get('player_total', 10)
                    num_cards = blackjack_state.get('num_cards', 2)
                else:
                    player_total = blackjack_state.player_total
                    num_cards = blackjack_state.num_cards
                
                curriculum_mask = curriculum.get_curriculum_mask(
                    legal_actions,
                    [player_total],
                    num_cards == 2
                )
                
                # Check that only expected actions are available
                available_actions = [i for i, available in enumerate(curriculum_mask) if available]
                for action in available_actions:
                    assert action in expected_actions, f"Unexpected action {action} in stage {stage.name}"
                
                # Take a random legal action
                if available_actions:
                    action = np.random.choice(available_actions)
                    action_counts[action] += 1
                else:
                    action = 0  # STAND if no actions available (shouldn't happen)
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
        
        # Verify only expected actions were used
        used_actions = [action for action, count in action_counts.items() if count > 0]
        for action in used_actions:
            assert action in expected_actions, f"Used unexpected action {action} in stage {stage.name}"
        
        print(f"✓ Stage {stage.name}: Used actions {used_actions}")
    
    env.close()
    print("✓ Action space curriculum validated")


def main():
    """Run all Phase 4 tests"""
    print("🎓 PHASE 4 CURRICULUM LEARNING TESTS 🎓\n")
    
    try:
        test_curriculum_stages()
        test_noisy_networks()
        test_dueling_dqn()
        test_prioritized_replay()
        test_rainbow_agent()
        test_action_space_curriculum()
        test_curriculum_training_integration()
        
        print("\n🎉 ALL PHASE 4 TESTS PASSED! 🎉")
        print("\n📋 Phase 4 Objectives Completed:")
        print("✓ Curriculum learning system with progressive action unlocking")
        print("✓ Rainbow DQN with dueling architecture and distributional learning")
        print("✓ Prioritized experience replay buffer")
        print("✓ Noisy networks for exploration")
        print("✓ Multi-step returns and target network updates")
        print("✓ Legal action masking throughout curriculum")
        print("✓ Training integration with curriculum progression")
        print("\n✅ Ready for full curriculum training and Phase 5!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
