"""
Test Phase 0 Implementation

This script validates the Phase 0 objectives:
1. Fixed rule set for v1
2. Expected value per round optimization
3. Legal action masks at every decision
"""

import gymnasium as gym
import BlackJackSim
import numpy as np
from BlackJackSim.config import V1_RULES

def test_fixed_rules():
    """Test that the fixed rule set is properly implemented"""
    print("=== Testing Fixed Rule Set ===")
    
    # Test rule configuration
    print(f"Rules: {V1_RULES.to_dict()}")
    
    # Create environment with fixed rules
    env = gym.make("BlackjackSim-v0")
    obs, info = env.reset()
    
    print(f"Rules in environment: {info['rules']}")
    assert info['rules']['num_decks'] == 6, "Should use 6 decks"
    assert info['rules']['blackjack_payout'] == 1.5, "Should use 3:2 payout"
    assert info['rules']['dealer_rule'] == 'stands_soft_17', "Dealer should stand on soft 17"
    
    print("✓ Fixed rules validated")
    env.close()

def test_legal_action_masks():
    """Test that legal action masks are returned at every decision"""
    print("\n=== Testing Legal Action Masks ===")
    
    env = gym.make("BlackjackSim-v0")
    
    for game in range(5):
        print(f"\nGame {game + 1}:")
        obs, info = env.reset()
        
        # Check initial legal actions
        legal_actions = info['legal_actions']
        print(f"Initial state: {obs}")
        print(f"Legal actions: {legal_actions} [STAY, HIT, DOUBLE, SPLIT, SURRENDER]")
        
        # Validate mask structure
        assert len(legal_actions) == 5, "Should have 5 action slots"
        assert all(isinstance(x, (bool, int)) for x in legal_actions), "Should be boolean mask"
        
        # STAY and HIT should almost always be legal (unless busted)
        if obs[0] <= 21:  # not busted
            assert legal_actions[0], "STAY should be legal when not busted"
            assert legal_actions[1], "HIT should be legal when not busted"
        
        # Test action execution with legal mask checking
        step_count = 0
        while step_count < 5:  # Limit steps to avoid infinite loops
            step_count += 1
            
            # Find legal actions
            legal_indices = [i for i, legal in enumerate(legal_actions) if legal]
            if not legal_indices:
                break
                
            # Choose a legal action (prefer HIT if available and total < 17)
            if obs[0] < 17 and legal_actions[1]:  # HIT
                action = 1
            else:  # STAY
                action = 0
                
            print(f"  Step {step_count}: Taking action {action} (legal: {legal_actions[action]})")
            
            obs, reward, terminated, truncated, info = env.step(action)
            legal_actions = info['legal_actions']
            
            print(f"    Result: obs={obs}, reward={reward}, terminated={terminated}")
            print(f"    New legal actions: {legal_actions}")
            
            if terminated or truncated:
                break
    
    print("✓ Legal action masks validated")
    env.close()

def test_expected_value_rewards():
    """Test that rewards properly reflect expected value per round"""
    print("\n=== Testing Expected Value Rewards ===")
    
    env = gym.make("BlackjackSim-v0")
    
    # Track different outcome types and their rewards
    outcomes = {}
    
    for game in range(100):
        obs, info = env.reset()
        
        # Simple strategy: hit if under 17, stay otherwise
        while True:
            if obs[0] < 17:
                action = 1  # HIT
            else:
                action = 0  # STAY
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                result = info['result']
                if result:
                    if result not in outcomes:
                        outcomes[result] = []
                    outcomes[result].append(reward)
                break
    
    # Validate reward mappings
    print("Observed outcomes and rewards:")
    for result, rewards in outcomes.items():
        avg_reward = np.mean(rewards)
        print(f"  {result}: {avg_reward:.3f} (n={len(rewards)})")
        
        # Validate expected rewards
        if result == 'WIN':
            assert all(r == 1.0 for r in rewards), "WIN should give +1.0"
        elif result == 'LOST':
            assert all(r == -1.0 for r in rewards), "LOST should give -1.0"
        elif result == 'BLACKJACK':
            assert all(r == 1.5 for r in rewards), "BLACKJACK should give +1.5 (3:2)"
        elif result == 'PUSH':
            assert all(r == 0.0 for r in rewards), "PUSH should give 0.0"
        elif result == 'SURRENDER':
            assert all(r == -0.5 for r in rewards), "SURRENDER should give -0.5"
        elif result == 'DOUBLEWIN':
            assert all(r == 2.0 for r in rewards), "DOUBLEWIN should give +2.0"
        elif result == 'DOUBLELOSS':
            assert all(r == -2.0 for r in rewards), "DOUBLELOSS should give -2.0"
    
    print("✓ Expected value rewards validated")
    env.close()

def test_enhanced_observation_space():
    """Test that enhanced observation space provides needed info for Phase 1"""
    print("\n=== Testing Enhanced Observation Space ===")
    
    env = gym.make("BlackjackSim-v0")
    obs, info = env.reset()
    
    print(f"Observation: {obs}")
    print(f"Observation space: {env.observation_space}")
    
    # Validate observation structure
    assert len(obs) == 7, "Should have 7 observation components"
    
    total, dealer_face, usable_ace, is_pair, num_cards, is_doubled, split_count = obs
    
    print(f"  Player total: {total}")
    print(f"  Dealer face: {dealer_face}")
    print(f"  Usable ace: {usable_ace}")
    print(f"  Is pair: {is_pair}")
    print(f"  Num cards: {num_cards}")
    print(f"  Is doubled: {is_doubled}")
    print(f"  Split count: {split_count}")
    
    # Validate ranges
    assert 2 <= total <= 31, "Total should be in valid range"
    assert 1 <= dealer_face <= 10, "Dealer face should be 1-10"
    assert usable_ace in [0, 1], "Usable ace should be binary"
    assert is_pair in [0, 1], "Is pair should be binary"
    assert num_cards >= 2, "Should have at least 2 cards"
    assert is_doubled in [0, 1], "Is doubled should be binary"
    assert 0 <= split_count <= 3, "Split count should be 0-3"
    
    print("✓ Enhanced observation space validated")
    env.close()

def test_info_completeness():
    """Test that info dict contains all required Phase 1 information"""
    print("\n=== Testing Info Completeness ===")
    
    env = gym.make("BlackjackSim-v0")
    obs, info = env.reset()
    
    required_keys = [
        'player_cards', 'dealer_cards', 'player_total', 'dealer_total',
        'player_is_soft', 'dealer_is_soft', 'player_blackjack', 'dealer_blackjack',
        'hand_type', 'result', 'legal_actions', 'rules', 'true_count', 'decks_remaining'
    ]
    
    print("Required info keys:")
    for key in required_keys:
        assert key in info, f"Missing required key: {key}"
        print(f"  ✓ {key}: {info[key]}")
    
    print("✓ Info completeness validated")
    env.close()

if __name__ == "__main__":
    print("🎰 PHASE 0 VALIDATION TESTS 🎰\n")
    
    test_fixed_rules()
    test_legal_action_masks()
    test_expected_value_rewards()
    test_enhanced_observation_space()
    test_info_completeness()
    
    print("\n🎉 ALL PHASE 0 TESTS PASSED! 🎉")
    print("\nPhase 0 Objectives Completed:")
    print("✓ Fixed rule set for v1 training and evaluation")
    print("✓ Expected value per round optimization target")
    print("✓ Legal action masks at every decision point")
    print("✓ Enhanced observation space for Phase 1 compatibility")
    print("✓ Complete info dict with all required features")
