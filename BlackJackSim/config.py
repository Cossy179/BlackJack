"""
Blackjack Rule Configuration for Phase 0

This module defines the fixed rule set for v1 training and evaluation.
All rules are locked in for consistency during training.
"""

from dataclasses import dataclass
from enum import Enum


class DealerRule(Enum):
    """Dealer rule for soft 17"""
    HITS_SOFT_17 = "hits_soft_17"
    STANDS_SOFT_17 = "stands_soft_17"


class SurrenderType(Enum):
    """Surrender options"""
    LATE_SURRENDER = "late_surrender"
    EARLY_SURRENDER = "early_surrender" 
    NO_SURRENDER = "no_surrender"


class PeekRule(Enum):
    """Dealer peek rules"""
    PEEK = "peek"  # US style - dealer peeks for blackjack
    ENHC = "enhc"  # European No Hole Card


@dataclass
class BlackjackRules:
    """Fixed rule set for v1 training and evaluation"""
    
    # Deck configuration
    num_decks: int = 6
    penetration: float = 0.75  # Deal 75% of shoe before reshuffling
    
    # Dealer rules
    dealer_rule: DealerRule = DealerRule.STANDS_SOFT_17
    peek_rule: PeekRule = PeekRule.PEEK
    
    # Payout rules  
    blackjack_payout: float = 1.5  # 3:2 payout (standard)
    
    # Player options
    double_after_split: bool = True
    max_resplits: int = 3  # Maximum number of times can resplit (4 hands total)
    can_resplit_aces: bool = False
    surrender_type: SurrenderType = SurrenderType.LATE_SURRENDER
    
    # Action availability
    can_double_any_two: bool = True  # Can double on any first two cards
    can_hit_split_aces: bool = False  # Split aces receive only one card
    
    def to_dict(self) -> dict:
        """Convert rules to dictionary for logging/serialization"""
        return {
            'num_decks': self.num_decks,
            'penetration': self.penetration,
            'dealer_rule': self.dealer_rule.value,
            'peek_rule': self.peek_rule.value,
            'blackjack_payout': self.blackjack_payout,
            'double_after_split': self.double_after_split,
            'max_resplits': self.max_resplits,
            'can_resplit_aces': self.can_resplit_aces,
            'surrender_type': self.surrender_type.value,
            'can_double_any_two': self.can_double_any_two,
            'can_hit_split_aces': self.can_hit_split_aces,
        }


# Global rule configuration for v1
V1_RULES = BlackjackRules()


def get_reward_for_result(result_type: str, rules: BlackjackRules = V1_RULES) -> float:
    """
    Calculate reward based on hand result and rules.
    
    This implements the expected value per round optimization target.
    """
    rewards = {
        'WIN': 1.0,
        'DOUBLEWIN': 2.0,
        'DOUBLELOSS': -2.0,
        'BLACKJACK': rules.blackjack_payout,
        'LOST': -1.0,
        'PUSH': 0.0,
        'SURRENDER': -0.5,
    }
    return rewards.get(result_type, 0.0)


def get_legal_actions(hand_cards: list, is_first_decision: bool, is_pair: bool, 
                     has_split_already: bool = False, split_count: int = 0,
                     rules: BlackjackRules = V1_RULES) -> list:
    """
    Return legal action mask for current state.
    
    Returns list of booleans: [can_stay, can_hit, can_double, can_split, can_surrender]
    
    Actions are indexed as:
    0: STAY
    1: HIT  
    2: DOUBLE
    3: SPLIT
    4: SURRENDER
    """
    # Initialize all actions as illegal
    legal = [False, False, False, False, False]
    
    # STAY and HIT are almost always legal (except when already busted)
    hand_total = sum(min(card, 10) if card > 1 else 11 for card in hand_cards)
    # Adjust for aces
    aces = sum(1 for card in hand_cards if card == 1)
    while hand_total > 21 and aces > 0:
        hand_total -= 10
        aces -= 1
    
    if hand_total <= 21:
        legal[0] = True  # STAY
        legal[1] = True  # HIT
    
    # DOUBLE - only on first decision if allowed
    if is_first_decision and rules.can_double_any_two:
        legal[2] = True
        
    # SPLIT - only if pair on first decision and under split limit
    if is_pair and is_first_decision and split_count < rules.max_resplits:
        legal[3] = True
        
    # SURRENDER - only on first decision if allowed  
    if (is_first_decision and 
        rules.surrender_type != SurrenderType.NO_SURRENDER):
        legal[4] = True
    
    return legal
