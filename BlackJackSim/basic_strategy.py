"""
Phase 1 Basic Strategy Baseline

Rule-correct basic strategy aligned with v1 rules plus optional
true-count-conditioned overrides for borderline decisions.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from enum import Enum

from .BlackJack import PlayOptions, HandType
from .config import BlackjackRules, V1_RULES, DealerRule
from .state_representation import BlackjackState


class BasicStrategyDecision(Enum):
    """Basic strategy decision types"""
    STAND = 0
    HIT = 1
    DOUBLE = 2
    SPLIT = 3
    SURRENDER = 4


class BasicStrategyBaseline:
    """
    Rule-correct basic strategy baseline for v1 rules.
    
    This baseline is aligned with:
    - 6 decks
    - Dealer stands on soft 17
    - 3:2 blackjack payout
    - Double after split allowed
    - Late surrender allowed
    - Max 3 resplits
    """
    
    def __init__(self, rules: BlackjackRules = V1_RULES, use_true_count: bool = False):
        self.rules = rules
        self.use_true_count = use_true_count
        
        # Basic strategy tables for our exact rules
        self._build_strategy_tables()
        
        # True count deviations for borderline decisions
        if use_true_count:
            self._build_true_count_deviations()
    
    def _build_strategy_tables(self):
        """Build basic strategy lookup tables for our exact rules"""
        
        # Hard totals (rows: 8-17, cols: dealer 2-10,A)
        # 0=Stand, 1=Hit, 2=Double, 4=Surrender
        self.hard_strategy = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 8
            [1, 2, 2, 2, 2, 1, 1, 1, 1, 1],  # 9  
            [2, 2, 2, 2, 2, 2, 2, 2, 1, 1],  # 10
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 1],  # 11
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1],  # 12
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # 13
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # 14
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # 15
            [1, 0, 0, 0, 0, 1, 1, 1, 4, 4],  # 16 (surrender vs 9,10,A if allowed)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 17
        ]
        
        # Soft totals (rows: 13-19, cols: dealer 2-10,A)  
        # A,2 through A,8
        self.soft_strategy = [
            [1, 1, 1, 2, 2, 1, 1, 1, 1, 1],  # A,2 (soft 13)
            [1, 1, 1, 2, 2, 1, 1, 1, 1, 1],  # A,3 (soft 14)
            [1, 1, 2, 2, 2, 1, 1, 1, 1, 1],  # A,4 (soft 15)
            [1, 1, 2, 2, 2, 1, 1, 1, 1, 1],  # A,5 (soft 16)
            [1, 2, 2, 2, 2, 1, 1, 1, 1, 1],  # A,6 (soft 17)
            [0, 2, 2, 2, 2, 0, 0, 1, 1, 1],  # A,7 (soft 18)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A,8 (soft 19)
        ]
        
        # Pair splitting (rows: A,A through 10,10, cols: dealer 2-10,A)
        # 1=Hit, 2=Double, 3=Split, 0=Stand
        self.pair_strategy = [
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # A,A
            [1, 1, 3, 3, 3, 3, 1, 1, 1, 1],  # 2,2
            [1, 1, 3, 3, 3, 3, 1, 1, 1, 1],  # 3,3
            [1, 1, 1, 1, 3, 3, 1, 1, 1, 1],  # 4,4
            [2, 2, 2, 2, 2, 2, 2, 2, 1, 1],  # 5,5 (never split, double)
            [1, 3, 3, 3, 3, 1, 1, 1, 1, 1],  # 6,6
            [1, 3, 3, 3, 3, 3, 1, 1, 1, 1],  # 7,7
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # 8,8
            [3, 3, 3, 3, 3, 0, 3, 3, 0, 0],  # 9,9
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10,10 (never split)
        ]
    
    def _build_true_count_deviations(self):
        """Build true count deviation table for borderline decisions"""
        
        # Format: (player_total, dealer_upcard, soft_hand): {true_count_threshold: action}
        self.true_count_deviations = {
            # Insurance bet (not implemented in basic strategy)
            # Some common deviations for hard totals
            (16, 10, False): {0: BasicStrategyDecision.SURRENDER, 4: BasicStrategyDecision.STAND},
            (15, 10, False): {4: BasicStrategyDecision.STAND},
            (12, 3, False): {2: BasicStrategyDecision.STAND},
            (12, 2, False): {3: BasicStrategyDecision.STAND},
            (13, 2, False): {-1: BasicStrategyDecision.HIT},
            (13, 3, False): {-2: BasicStrategyDecision.HIT},
            
            # Some soft hand deviations
            (18, 9, True): {1: BasicStrategyDecision.STAND},  # A,7 vs 9
            (18, 10, True): {1: BasicStrategyDecision.STAND}, # A,7 vs 10
            
            # Pair deviations
            (20, 5, False): {5: BasicStrategyDecision.SPLIT},  # 10,10 vs 5 (very high count)
            (20, 6, False): {4: BasicStrategyDecision.SPLIT},  # 10,10 vs 6 (high count)
        }
    
    def get_action(self, state: BlackjackState) -> int:
        """
        Get basic strategy action for given state.
        
        Args:
            state: Current blackjack state
            
        Returns:
            Action index (0=STAND, 1=HIT, 2=DOUBLE, 3=SPLIT, 4=SURRENDER)
        """
        
        # Check for true count deviations first
        if self.use_true_count:
            deviation_action = self._check_true_count_deviations(state)
            if deviation_action is not None:
                return deviation_action.value
        
        # Handle pairs first (only on initial 2 cards)
        if state.is_pair and state.num_cards == 2:
            return self._get_pair_action(state)
        
        # Handle soft hands
        if state.is_soft:
            return self._get_soft_action(state)
        
        # Handle hard hands
        return self._get_hard_action(state)
    
    def _check_true_count_deviations(self, state: BlackjackState) -> Optional[BasicStrategyDecision]:
        """Check if true count warrants deviation from basic strategy"""
        
        key = (state.player_total, state.dealer_upcard, state.is_soft)
        if key not in self.true_count_deviations:
            return None
        
        deviations = self.true_count_deviations[key]
        
        # Find the appropriate deviation based on true count
        for threshold in sorted(deviations.keys(), reverse=True):
            if state.true_count >= threshold:
                return deviations[threshold]
        
        return None
    
    def _get_hard_action(self, state: BlackjackState) -> int:
        """Get action for hard totals"""
        
        total = state.player_total
        dealer = state.dealer_upcard
        
        # Handle very low totals (always hit)
        if total <= 7:
            return BasicStrategyDecision.HIT.value
        
        # Handle very high totals (always stand)  
        if total >= 18:
            return BasicStrategyDecision.STAND.value
        
        # Use lookup table for 8-17
        row = total - 8
        col = dealer - 1 if dealer <= 10 else 9  # Ace = index 9
        
        if 0 <= row < len(self.hard_strategy) and 0 <= col < 10:
            action = self.hard_strategy[row][col]
            
            # Check if action is legal in current context
            if action == BasicStrategyDecision.DOUBLE.value and state.num_cards > 2:
                return BasicStrategyDecision.HIT.value  # Can't double after first two cards
            
            if action == BasicStrategyDecision.SURRENDER.value:
                if (state.num_cards > 2 or 
                    self.rules.surrender_type.value == 'no_surrender'):
                    return BasicStrategyDecision.HIT.value  # Can't surrender
            
            return action
        
        # Default to hit for edge cases
        return BasicStrategyDecision.HIT.value
    
    def _get_soft_action(self, state: BlackjackState) -> int:
        """Get action for soft hands"""
        
        total = state.player_total
        dealer = state.dealer_upcard
        
        # Handle edge cases
        if total <= 12:  # Very low soft totals (impossible in normal play)
            return BasicStrategyDecision.HIT.value
        
        if total >= 20:  # Soft 20+ (A,9+)
            return BasicStrategyDecision.STAND.value
        
        # Use lookup table for soft 13-19
        row = total - 13
        col = dealer - 1 if dealer <= 10 else 9  # Ace = index 9
        
        if 0 <= row < len(self.soft_strategy) and 0 <= col < 10:
            action = self.soft_strategy[row][col]
            
            # Check if double is legal
            if action == BasicStrategyDecision.DOUBLE.value and state.num_cards > 2:
                return BasicStrategyDecision.HIT.value
            
            return action
        
        # Default to hit
        return BasicStrategyDecision.HIT.value
    
    def _get_pair_action(self, state: BlackjackState) -> int:
        """Get action for pairs"""
        
        # Extract card value from player total for pairs
        # This is a simplified approach - in a full implementation,
        # we'd track the actual card values
        if state.player_total == 2:  # A,A (counted as 2 if both aces as 1)
            card_value = 1
        elif state.player_total == 12 and state.is_soft:  # A,A (one ace as 11)
            card_value = 1  
        elif state.player_total % 2 == 0:
            card_value = state.player_total // 2
        else:
            # Edge case - default to not splitting
            return self._get_hard_action(state)
        
        dealer = state.dealer_upcard
        
        # Convert card value to table index
        if card_value == 1:  # Aces
            row = 0
        elif card_value <= 10:
            row = card_value - 1
        else:
            row = 9  # 10-value cards
        
        col = dealer - 1 if dealer <= 10 else 9  # Ace = index 9
        
        if 0 <= row < len(self.pair_strategy) and 0 <= col < 10:
            action = self.pair_strategy[row][col]
            
            # Check if split is legal (under resplit limit)
            if action == BasicStrategyDecision.SPLIT.value:
                if state.split_count >= self.rules.max_resplits:
                    # Can't split anymore, use non-pair strategy
                    if state.is_soft:
                        return self._get_soft_action(state)
                    else:
                        return self._get_hard_action(state)
            
            return action
        
        # Default to non-pair strategy
        if state.is_soft:
            return self._get_soft_action(state)
        else:
            return self._get_hard_action(state)
    
    def get_action_probabilities(self, state: BlackjackState) -> np.ndarray:
        """
        Get action probabilities for given state.
        
        For basic strategy, this returns a one-hot vector for the chosen action.
        """
        action = self.get_action(state)
        probs = np.zeros(5)
        probs[action] = 1.0
        return probs
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of strategy configuration"""
        return {
            'strategy_type': 'basic_strategy',
            'rules': self.rules.to_dict(),
            'use_true_count': self.use_true_count,
            'num_deviations': len(self.true_count_deviations) if self.use_true_count else 0,
            'description': 'Rule-correct basic strategy for 6-deck, S17, 3:2, DAS game'
        }


def create_basic_strategy_baseline(rules: BlackjackRules = V1_RULES, 
                                 use_true_count: bool = False) -> BasicStrategyBaseline:
    """
    Factory function to create basic strategy baseline.
    
    Args:
        rules: Blackjack rules configuration
        use_true_count: Whether to use true count deviations
        
    Returns:
        Configured basic strategy baseline
    """
    return BasicStrategyBaseline(rules=rules, use_true_count=use_true_count)
