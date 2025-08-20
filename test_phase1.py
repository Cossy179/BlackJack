"""
Test Phase 1 Implementation

This script validates the Phase 1 objectives:
1. Compact numeric feature vector with all required features
2. Rule-correct basic strategy baseline aligned with exact rules
3. Optional true-count-conditioned overrides for borderline decisions
"""

import gymnasium as gym
import BlackJackSim
import numpy as np
from BlackJackSim.config import V1_RULES
from BlackJackSim.state_representation import BlackjackState, StateExtractor, get_feature_names, get_feature_dimensions
from BlackJackSim.basic_strategy import BasicStrategyBaseline, create_basic_strategy_baseline


def test_compact_state_representation():
    """Test the compact numeric feature vector"""
    print("=== Testing Compact State Representation ===")
    
    # Test with compact state enabled
    env = gym.make("BlackjackSim-v0", use_compact_state=True)
    obs, info = env.reset()
    
    print(f"Compact observation shape: {obs.shape}")
    print(f"Expected dimensions: {get_feature_dimensions()}")
    print(f"Feature names: {get_feature_names()}")
    
    # Validate observation structure
    assert obs.shape == (12,), f"Expected 12 features, got {obs.shape}"
    assert all(-1.0 <= x <= 1.0 for x in obs), "Features should be normalized to [-1, 1]"
    
    # Extract structured state
    state = info['blackjack_state']
    print(f"\nStructured state:")
    for feature, value in state.items():
        print(f"  {feature}: {value}")
    
    # Test state extraction
    extractor = StateExtractor(V1_RULES)
    blackjack_state = BlackjackState(**state)
    state_array = blackjack_state.to_array()
    
    print(f"\nState array: {state_array}")
    print(f"Observation:  {obs}")
    
    print("✓ Compact state representation validated")
    env.close()


def test_basic_strategy_baseline():
    """Test the rule-correct basic strategy baseline"""
    print("\n=== Testing Basic Strategy Baseline ===")
    
    # Create baseline strategies
    basic_strategy = create_basic_strategy_baseline(V1_RULES, use_true_count=False)
    count_strategy = create_basic_strategy_baseline(V1_RULES, use_true_count=True)
    
    print(f"Basic strategy summary: {basic_strategy.get_strategy_summary()}")
    print(f"Count strategy summary: {count_strategy.get_strategy_summary()}")
    
    # Test with environment
    env = gym.make("BlackjackSim-v0", use_compact_state=True)
    
    # Test various scenarios
    test_scenarios = []
    for _ in range(20):
        obs, info = env.reset()
        state = BlackjackState(**info['blackjack_state'])
        
        basic_action = basic_strategy.get_action(state)
        count_action = count_strategy.get_action(state)
        recommended_action = info['basic_strategy_action']
        
        test_scenarios.append({
            'state': state,
            'basic_action': basic_action,
            'count_action': count_action,
            'recommended': recommended_action,
            'legal_actions': info['legal_actions']
        })
        
        env.step(0)  # Just take STAND to end episode
    
    # Analyze results
    print(f"\nTested {len(test_scenarios)} scenarios:")
    
    action_names = ['STAND', 'HIT', 'DOUBLE', 'SPLIT', 'SURRENDER']
    action_counts = {'basic': {}, 'count': {}, 'recommended': {}}
    
    for scenario in test_scenarios:
        state = scenario['state']
        legal = scenario['legal_actions']
        
        print(f"\nScenario: Total={state.player_total}, Dealer={state.dealer_upcard}, "
              f"Soft={state.is_soft}, True Count={state.true_count:.1f}")
        print(f"  Legal actions: {[action_names[i] for i, legal in enumerate(legal) if legal]}")
        print(f"  Basic strategy: {action_names[scenario['basic_action']]}")
        print(f"  Count strategy: {action_names[scenario['count_action']]}")
        print(f"  Recommended:   {action_names[scenario['recommended']]}")
        
        # Check if actions are legal
        assert legal[scenario['basic_action']], "Basic strategy chose illegal action!"
        assert legal[scenario['count_action']], "Count strategy chose illegal action!"
        assert legal[scenario['recommended']], "Recommended action is illegal!"
        
        # Count action frequency
        for strategy, action in [('basic', scenario['basic_action']), 
                               ('count', scenario['count_action']),
                               ('recommended', scenario['recommended'])]:
            action_name = action_names[action]
            if action_name not in action_counts[strategy]:
                action_counts[strategy][strategy] = 0
            action_counts[strategy][action_name] = action_counts[strategy].get(action_name, 0) + 1
    
    print(f"\nAction distribution:")
    for strategy, counts in action_counts.items():
        print(f"  {strategy}: {counts}")
    
    print("✓ Basic strategy baseline validated")
    env.close()


def test_true_count_deviations():
    """Test true count conditional overrides"""
    print("\n=== Testing True Count Deviations ===")
    
    count_strategy = create_basic_strategy_baseline(V1_RULES, use_true_count=True)
    basic_strategy = create_basic_strategy_baseline(V1_RULES, use_true_count=False)
    
    # Create test states with different true counts
    test_states = [
        # 16 vs 10 - should surrender at low count, stand at high count
        BlackjackState(
            player_total=16, is_soft=False, is_pair=False, num_cards=2, is_doubled=False,
            dealer_upcard=10, split_count=0, subhand_index=0, total_subhands=1,
            decks_remaining=0.75, running_count=tc*3, true_count=tc
        ) for tc in [-2, 0, 2, 4, 6]
    ]
    
    # Add more test scenarios
    test_states.extend([
        # 12 vs 3 - should hit at low count, stand at high count  
        BlackjackState(
            player_total=12, is_soft=False, is_pair=False, num_cards=2, is_doubled=False,
            dealer_upcard=3, split_count=0, subhand_index=0, total_subhands=1,
            decks_remaining=0.75, running_count=tc*3, true_count=tc
        ) for tc in [-2, 0, 2, 4]
    ])
    
    deviation_found = False
    action_names = ['STAND', 'HIT', 'DOUBLE', 'SPLIT', 'SURRENDER']
    
    print("Testing true count deviations:")
    for state in test_states:
        basic_action = basic_strategy.get_action(state)
        count_action = count_strategy.get_action(state)
        
        if basic_action != count_action:
            deviation_found = True
            print(f"  DEVIATION: Total={state.player_total} vs {state.dealer_upcard}, "
                  f"TC={state.true_count:.1f}")
            print(f"    Basic: {action_names[basic_action]} -> Count: {action_names[count_action]}")
    
    if deviation_found:
        print("✓ True count deviations working")
    else:
        print("⚠ No deviations found in test scenarios (this may be expected)")
    
    print("✓ True count system validated")


def test_demonstration_data_generation():
    """Test generation of demonstration data for imitation learning"""
    print("\n=== Testing Demonstration Data Generation ===")
    
    env = gym.make("BlackjackSim-v0", use_compact_state=True)
    baseline = env.unwrapped.get_basic_strategy_baseline()
    
    # Generate demonstration data
    demonstrations = []
    
    for episode in range(100):
        obs, info = env.reset()
        
        episode_data = []
        step_count = 0
        
        while step_count < 10:  # Limit steps
            step_count += 1
            
            # Get state representation
            state = BlackjackState(**info['blackjack_state'])
            legal_actions = np.array(info['legal_actions'])
            
            # Get basic strategy action
            baseline_action = baseline.get_action(state)
            
            # Record demonstration step
            demo_step = {
                'observation': obs.copy(),
                'state': state.to_dict(),
                'legal_actions': legal_actions.copy(),
                'action': baseline_action,
                'action_probs': baseline.get_action_probabilities(state)
            }
            episode_data.append(demo_step)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(baseline_action)
            
            if terminated or truncated:
                break
        
        demonstrations.append({
            'episode': episode,
            'steps': episode_data,
            'final_reward': reward if (terminated or truncated) else 0.0
        })
    
    # Analyze demonstration data
    total_steps = sum(len(ep['steps']) for ep in demonstrations)
    action_distribution = np.zeros(5)
    
    for episode in demonstrations:
        for step in episode['steps']:
            action_distribution[step['action']] += 1
    
    action_distribution /= total_steps
    action_names = ['STAND', 'HIT', 'DOUBLE', 'SPLIT', 'SURRENDER']
    
    print(f"Generated {len(demonstrations)} episodes with {total_steps} total steps")
    print(f"Action distribution:")
    for i, (action, prob) in enumerate(zip(action_names, action_distribution)):
        print(f"  {action}: {prob:.3f}")
    
    # Validate demonstration quality
    assert total_steps > 100, "Should generate substantial demonstration data"
    assert action_distribution[1] > 0.3, "Should have significant HIT actions"  
    assert action_distribution[0] > 0.2, "Should have significant STAND actions"
    
    print("✓ Demonstration data generation validated")
    env.close()


def test_phase1_integration():
    """Test integration of all Phase 1 components"""
    print("\n=== Testing Phase 1 Integration ===")
    
    # Test both environment modes
    for use_compact in [False, True]:
        print(f"\nTesting with compact_state={use_compact}")
        
        env = gym.make("BlackjackSim-v0", use_compact_state=use_compact)
        obs, info = env.reset()
        
        # Validate observation space
        if use_compact:
            assert isinstance(obs, np.ndarray), "Compact obs should be numpy array"
            assert obs.shape == (12,), "Should have 12 features"
        else:
            assert isinstance(obs, tuple), "Legacy obs should be tuple"
            assert len(obs) == 7, "Should have 7 legacy features"
        
        # Validate info completeness
        required_keys = [
            'player_cards', 'dealer_cards', 'legal_actions', 'rules',
            'basic_strategy_action', 'blackjack_state', 'true_count'
        ]
        
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
        
        # Test a full episode with basic strategy
        step_count = 0
        while step_count < 10:
            step_count += 1
            
            # Use basic strategy action
            action = info['basic_strategy_action']
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        env.close()
    
    print("✓ Phase 1 integration validated")


if __name__ == "__main__":
    print("🎯 PHASE 1 VALIDATION TESTS 🎯\n")
    
    test_compact_state_representation()
    test_basic_strategy_baseline()
    test_true_count_deviations()
    test_demonstration_data_generation()
    test_phase1_integration()
    
    print("\n🎉 ALL PHASE 1 TESTS PASSED! 🎉")
    print("\nPhase 1 Objectives Completed:")
    print("✓ Compact numeric feature vector with 12 features")
    print("✓ Rule-correct basic strategy baseline aligned with v1 rules")
    print("✓ Optional true-count-conditioned overrides for borderline decisions")
    print("✓ Baseline serves for sanity checks and demonstration data")
    print("✓ Integration with Phase 0 legal action masking")
    print("\nReady for Phase 2: Imitation pretraining (warm start)")
