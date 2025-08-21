import random
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .BlackJack import Table, Player, HandResults, HandType, PlayOptions
from .config import BlackjackRules, V1_RULES, get_reward_for_result
from .state_representation import BlackjackState, StateExtractor, get_feature_dimensions
from .basic_strategy import BasicStrategyBaseline


class BlackjackSimEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = None, 
                 rules: BlackjackRules = V1_RULES, use_compact_state: bool = False) -> None:
        super().__init__()

        self.rules = rules
        self.use_compact_state = use_compact_state
        self.state_extractor = StateExtractor(rules)
        
        # Create baseline for sanity checks and demonstration data
        self.basic_strategy = BasicStrategyBaseline(rules, use_true_count=True)
        
        # Actions: 0=STAY, 1=HIT, 2=DOUBLE, 3=SPLIT, 4=SURRENDER
        self.action_space = spaces.Discrete(5)

        if use_compact_state:
            # Phase 1 compact state representation (12 features)
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, 
                shape=(get_feature_dimensions(),), 
                dtype=np.float32
            )
        else:
            # Legacy observation space for backward compatibility
            self.observation_space = spaces.Tuple(
                (
                    spaces.Discrete(32),    # player total
                    spaces.Discrete(11),    # dealer face card  
                    spaces.Discrete(2),     # usable ace
                    spaces.Discrete(2),     # is pair
                    spaces.Discrete(11),    # number of cards (2-12)
                    spaces.Discrete(2),     # is doubled
                    spaces.Discrete(4),     # split count (0-3)
                )
            )

        self.render_mode = render_mode
        self._table: Optional[Table] = None
        self._done: bool = False
        self._last_info: Dict[str, Any] = {}
        self._seed_value: Optional[int] = None
        if seed is not None:
            self.seed(seed)

    def seed(self, seed: Optional[int] = None) -> None:
        self._seed_value = 0 if seed is None else int(seed)
        random.seed(self._seed_value)

    def _obs(self):
        assert self._table is not None
        
        if self.use_compact_state:
            # Extract full state representation
            state = self.state_extractor.extract_from_table(self._table)
            return state.to_array()
        else:
            # Legacy observation format
            player_hand = self._table.players[0].hands[0]
            dealer_face = self._table.dealer.hand.cards[0]

            # Use the improved hand calculation methods
            total = player_hand.getHandTotal()
            soft_hand = player_hand.isSoftHand()
            is_pair = (len(player_hand.cards) == 2 and 
                       player_hand.cards[0] == player_hand.cards[1])
            num_cards = len(player_hand.cards)
            is_doubled = player_hand.handType == HandType.DOUBLE
            split_count = player_hand.split_count

            # Cap values to satisfy observation space
            capped_total = min(total, 31)
            capped_cards = min(num_cards, 10)  # 2-12 cards -> 0-10 index
            capped_splits = min(split_count, 3)
            
            return (int(capped_total), int(dealer_face), int(1 if soft_hand else 0),
                    int(1 if is_pair else 0), int(capped_cards), int(1 if is_doubled else 0),
                    int(capped_splits))

    def _compute_reward(self) -> float:
        assert self._table is not None
        hand = self._table.players[0].hands[0]
        if hand.result is None:
            return 0.0
        return get_reward_for_result(hand.result.name, self.rules)
        
    def _get_legal_action_mask(self) -> np.ndarray:
        """Return legal action mask for current state"""
        assert self._table is not None
        hand = self._table.players[0].hands[0]
        legal_actions = hand.getLegalActions()
        return np.array(legal_actions, dtype=bool)
    
    def _get_blackjack_state(self) -> BlackjackState:
        """Get structured blackjack state representation"""
        assert self._table is not None
        return self.state_extractor.extract_from_table(self._table)
    
    def _get_basic_strategy_action(self) -> int:
        """Get basic strategy recommendation for current state"""
        if self._table is None:
            return 0
        
        state = self._get_blackjack_state()
        return self.basic_strategy.get_action(state)
    
    def get_basic_strategy_baseline(self) -> BasicStrategyBaseline:
        """Get the basic strategy baseline for external use"""
        return self.basic_strategy

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)

        self._done = False
        self._last_info = {}

        self._table = Table(self.rules)
        self._table.players.append(Player(self.rules))

        # Deal first two cards to player and dealer
        self._table.dealFirstTwoCards()

        # If immediate terminal (e.g., dealer BJ and player not), we will finish on first step with STAY
        observation = self._obs()
        info = self._collect_info()
        return observation, info

    def step(self, action: int):
        if self._done:
            raise RuntimeError("Environment step called after done=True. Call reset().")
        assert self._table is not None

        player_hand = self._table.players[0].hands[0]

        # Check for immediate blackjack on first action
        if len(player_hand.cards) == 2 and player_hand.isBlackJack():
            if not self._table.dealer.hand.isBlackJack():
                player_hand.result = HandResults.BLACKJACK
            else:
                player_hand.result = HandResults.PUSH
            self._done = True
            terminated = True
            truncated = False
            reward = self._compute_reward()
            observation = self._obs()
            info = self._collect_info()
            return observation, reward, terminated, truncated, info

        # Get legal actions mask
        legal_mask = self._get_legal_action_mask()
        
        # Normalize action
        action = int(action)
        if action < 0 or action >= self.action_space.n or not legal_mask[action]:
            # Default to STAY if action is invalid or illegal
            action = 0

        # Mark that first decision has been made
        if player_hand.is_first_decision:
            player_hand.is_first_decision = False

        # Map action to simulator effects
        if action == 1:  # HIT
            player_hand.addCard(self._table.shoe.getCard())
            if player_hand.getHandTotal() > 21:
                # Player busts
                if player_hand.handType == HandType.DOUBLE:
                    player_hand.result = HandResults.DOUBLELOSS
                else:
                    player_hand.result = HandResults.LOST
                self._done = True
        elif action == 2:  # DOUBLE
            player_hand.addCard(self._table.shoe.getCard())
            player_hand.handType = HandType.DOUBLE
            if player_hand.getHandTotal() > 21:
                player_hand.result = HandResults.DOUBLELOSS
            else:
                self._dealer_play_and_score()
            self._done = True
        elif action == 3:  # SPLIT - Note: Basic env doesn't fully support splits yet
            # For now, treat as STAY if split is selected
            self._dealer_play_and_score()
            self._done = True
        elif action == 4:  # SURRENDER
            player_hand.result = HandResults.SURRENDER
            self._done = True
        else:  # STAY (action == 0)
            self._dealer_play_and_score()
            self._done = True

        terminated = self._done
        truncated = False
        reward = 0.0
        if terminated:
            # Always ensure we have a result by the end
            if player_hand.result is None:
                self._table.recordHandResults()
            reward = self._compute_reward()

        observation = self._obs()
        info = self._collect_info()
        return observation, reward, terminated, truncated, info

    def _dealer_play_and_score(self) -> None:
        assert self._table is not None
        
        # Check if dealer has blackjack first
        if self._table.dealer.hand.isBlackJack():
            self._table.recordHandResults()
            return
            
        # Dealer plays according to rules
        while True:
            dealer_choice = self._table.dealer.hand.dealerPlay()
            if dealer_choice == PlayOptions.STAY:
                break
            if dealer_choice == PlayOptions.HIT:
                self._table.dealer.hand.addCard(self._table.shoe.getCard())
        
        # Score results using existing logic
        self._table.recordHandResults()

    def _collect_info(self) -> Dict[str, Any]:
        assert self._table is not None
        hand = self._table.players[0].hands[0]
        dealer_hand = self._table.dealer.hand
        
        info: Dict[str, Any] = {
            "player_cards": list(hand.cards),
            "dealer_cards": list(dealer_hand.cards),
            "player_total": hand.getHandTotal(),
            "dealer_total": dealer_hand.getHandTotal() if self._done else dealer_hand.cards[0],
            "player_is_soft": hand.isSoftHand(),
            "dealer_is_soft": dealer_hand.isSoftHand() if self._done else False,
            "player_blackjack": hand.isBlackJack(),
            "dealer_blackjack": dealer_hand.isBlackJack(),
            "hand_type": hand.handType.name,
            "result": hand.result.name if hand.result is not None else None,
            "legal_actions": self._get_legal_action_mask().tolist(),
            "rules": self.rules.to_dict(),
            "true_count": self._table.shoe.get_true_count(),
            "decks_remaining": len(self._table.shoe.cards) / 52,
            "basic_strategy_action": self._get_basic_strategy_action(),
            "blackjack_state": self._get_blackjack_state().to_dict(),
        }
        self._last_info = info
        return info

    def render(self):
        if self._table is None:
            return None
        
        hand = self._table.players[0].hands[0]
        dealer = self._table.dealer.hand
        
        print(f"🎰 BLACKJACK GAME 🎰")
        print(f"Dealer: {dealer.cards} (total: {dealer.getHandTotal()}, {'soft' if dealer.isSoftHand() else 'hard'})")
        print(f"Player: {hand.cards} (total: {hand.getHandTotal()}, {'soft' if hand.isSoftHand() else 'hard'})")
        
        if hand.isBlackJack():
            print("🎉 PLAYER BLACKJACK!")
        if dealer.isBlackJack():
            print("💸 DEALER BLACKJACK!")
            
        if hand.result:
            result_emojis = {
                "WIN": "🎉 WIN",
                "DOUBLEWIN": "🎊 DOUBLE WIN", 
                "DOUBLELOSS": "💸 DOUBLE LOSS",
                "BLACKJACK": "🔥 BLACKJACK WIN",
                "LOST": "💸 LOSS",
                "PUSH": "🤝 PUSH",
                "SURRENDER": "🏳️ SURRENDER"
            }
            print(f"Result: {result_emojis.get(hand.result.name, hand.result.name)}")
            
        obs = self._obs()
        print(f"Observation: (total={obs[0]}, dealer_face={obs[1]}, usable_ace={obs[2]})")
        print("-" * 50)

    def close(self):
        self._table = None

