"""
Phase 1 State Representation

Compact numeric feature vector for blackjack RL training.
Includes all information needed for optimal decision making.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .BlackJack import Hand, Table, Player, Dealer
from .config import BlackjackRules, V1_RULES


@dataclass
class BlackjackState:
    """
    Compact state representation for blackjack RL.
    
    Features (12 total):
    1. player_total: Current hand total (4-31)
    2. is_soft: Whether hand has usable ace (0/1)
    3. is_pair: Whether initial 2 cards were pair (0/1) 
    4. num_cards: Number of cards in hand (2-12)
    5. is_doubled: Whether bet has been doubled (0/1)
    6. dealer_upcard: Dealer's face-up card (1-10)
    7. split_count: Number of splits this round (0-3)
    8. subhand_index: Index of current sub-hand (0-3)
    9. total_subhands: Total number of sub-hands (1-4)
    10. decks_remaining: Fraction of shoe remaining (0.0-1.0)
    11. running_count: Card counting running count (-52 to +52)
    12. true_count: True count (running/decks_remaining) (-20 to +20)
    """
    
    # Hand-specific features
    player_total: int          # 4-31 (busted hands capped at 31)
    is_soft: bool             # 0/1
    is_pair: bool             # 0/1 (only for initial 2 cards)
    num_cards: int            # 2-12 
    is_doubled: bool          # 0/1
    
    # Table context
    dealer_upcard: int        # 1-10 (Ace=1)
    split_count: int          # 0-3 (max splits allowed)
    subhand_index: int        # 0-3 (current hand being played)
    total_subhands: int       # 1-4 (total hands after splits)
    
    # Shoe context  
    decks_remaining: float    # 0.0-1.0 (fraction of shoe left)
    running_count: int        # Card counting running count
    true_count: float         # True count for composition-dependent play
    
    def to_array(self, normalize: bool = True) -> np.ndarray:
        """Convert state to numpy array for RL algorithms"""
        features = np.array([
            self.player_total,
            int(self.is_soft),
            int(self.is_pair),
            self.num_cards,
            int(self.is_doubled),
            self.dealer_upcard,
            self.split_count,
            self.subhand_index,
            self.total_subhands,
            self.decks_remaining,
            self.running_count,
            self.true_count
        ], dtype=np.float32)
        
        if normalize:
            return normalize_state_features_from_array(features)
        else:
            return features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for analysis"""
        return {
            'player_total': self.player_total,
            'is_soft': self.is_soft,
            'is_pair': self.is_pair,
            'num_cards': self.num_cards,
            'is_doubled': self.is_doubled,
            'dealer_upcard': self.dealer_upcard,
            'split_count': self.split_count,
            'subhand_index': self.subhand_index,
            'total_subhands': self.total_subhands,
            'decks_remaining': self.decks_remaining,
            'running_count': self.running_count,
            'true_count': self.true_count
        }
    
    @classmethod
    def from_table_state(cls, table: Table, hand_index: int = 0) -> 'BlackjackState':
        """
        Extract state representation from table state.
        
        Args:
            table: Current table state
            hand_index: Index of hand to extract state for (for splits)
        """
        player = table.players[0]  # Assume single player
        hand = player.hands[hand_index]
        dealer = table.dealer
        
        # Hand features
        player_total = min(hand.getHandTotal(), 31)  # Cap at 31
        is_soft = hand.isSoftHand()
        is_pair = (len(hand.cards) == 2 and hand.cards[0] == hand.cards[1])
        num_cards = len(hand.cards)
        is_doubled = (hand.handType.name == 'DOUBLE')
        
        # Table context
        dealer_upcard = dealer.hand.cards[0] if dealer.hand.cards else 1
        split_count = hand.split_count
        subhand_index = hand_index
        total_subhands = len(player.hands)
        
        # Shoe context
        total_cards = table.rules.num_decks * 52
        cards_remaining = len(table.shoe.cards)
        decks_remaining = cards_remaining / 52.0
        
        # Card counting (basic Hi-Lo system)
        running_count = table.shoe.get_running_count() if hasattr(table.shoe, 'get_running_count') else 0
        true_count = running_count / max(decks_remaining, 0.5) if decks_remaining > 0 else 0
        
        return cls(
            player_total=player_total,
            is_soft=is_soft,
            is_pair=is_pair,
            num_cards=num_cards,
            is_doubled=is_doubled,
            dealer_upcard=dealer_upcard,
            split_count=split_count,
            subhand_index=subhand_index,
            total_subhands=total_subhands,
            decks_remaining=decks_remaining,
            running_count=running_count,
            true_count=true_count
        )


class StateExtractor:
    """Utility class for extracting states from various blackjack environments"""
    
    def __init__(self, rules: BlackjackRules = V1_RULES):
        self.rules = rules
    
    def extract_from_gym_info(self, obs: Tuple, info: Dict[str, Any]) -> BlackjackState:
        """Extract state from Gym environment observation and info"""
        
        # Current gym obs format: (total, dealer_face, usable_ace, is_pair, num_cards, is_doubled, split_count)
        total, dealer_face, usable_ace, is_pair, num_cards, is_doubled, split_count = obs
        
        # Extract additional info
        decks_remaining = info.get('decks_remaining', 1.0)
        true_count = info.get('true_count', 0.0)
        
        # Calculate running count from true count
        running_count = int(true_count * max(decks_remaining, 0.5))
        
        return BlackjackState(
            player_total=total,
            is_soft=bool(usable_ace),
            is_pair=bool(is_pair),
            num_cards=num_cards,
            is_doubled=bool(is_doubled),
            dealer_upcard=dealer_face,
            split_count=split_count,
            subhand_index=0,  # Gym env doesn't track this yet
            total_subhands=1,  # Gym env doesn't support splits yet
            decks_remaining=decks_remaining,
            running_count=running_count,
            true_count=true_count
        )
    
    def extract_from_table(self, table: Table, hand_index: int = 0) -> BlackjackState:
        """Extract state directly from table"""
        return BlackjackState.from_table_state(table, hand_index)


def normalize_state_features_from_array(features: np.ndarray) -> np.ndarray:
    """
    Normalize state features to [-1, 1] range for neural networks.
    
    Returns normalized feature vector.
    """
    # Normalization: (feature - min) / (max - min) * 2 - 1 to get [-1, 1]
    # For features already in [0,1], just scale to [-1, 1]
    
    # Feature ranges:
    # player_total: 4-31 -> normalize to [-1, 1]
    # is_soft: 0-1 -> scale to [-1, 1] 
    # is_pair: 0-1 -> scale to [-1, 1]
    # num_cards: 2-12 -> normalize to [-1, 1]
    # is_doubled: 0-1 -> scale to [-1, 1]
    # dealer_upcard: 1-10 -> normalize to [-1, 1] 
    # split_count: 0-3 -> normalize to [-1, 1]
    # subhand_index: 0-3 -> normalize to [-1, 1]
    # total_subhands: 1-4 -> normalize to [-1, 1]
    # decks_remaining: 0-1 -> scale to [-1, 1]
    # running_count: -52 to +52 -> normalize to [-1, 1]
    # true_count: -20 to +20 -> normalize to [-1, 1]
    
    mins = np.array([4, 0, 0, 2, 0, 1, 0, 0, 1, 0, -52, -20], dtype=np.float32)
    maxs = np.array([31, 1, 1, 12, 1, 10, 3, 3, 4, 1, 52, 20], dtype=np.float32)
    
    # Normalize to [0, 1] first
    normalized = (features - mins) / (maxs - mins)
    
    # Scale to [-1, 1]
    normalized = normalized * 2.0 - 1.0
    
    # Clip to ensure bounds
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def normalize_state_features(state: BlackjackState) -> np.ndarray:
    """
    Normalize state features to [-1, 1] range for neural networks.
    
    Returns normalized feature vector.
    """
    features = state.to_array(normalize=False)
    return normalize_state_features_from_array(features)


def get_feature_names() -> List[str]:
    """Get list of feature names for analysis and debugging"""
    return [
        'player_total',
        'is_soft', 
        'is_pair',
        'num_cards',
        'is_doubled',
        'dealer_upcard',
        'split_count',
        'subhand_index',
        'total_subhands',
        'decks_remaining',
        'running_count',
        'true_count'
    ]


def get_feature_dimensions() -> int:
    """Get the dimensionality of the state representation"""
    return 12
