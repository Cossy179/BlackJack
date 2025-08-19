Phase 0 — Objectives, rules, and reward

Fix the single rule set you care about for v1 (decks, dealer stands or hits on 17, blackjack payout, double-after-split, resplit policy, surrender type, peek vs ENHC, penetration). Keep this constant for train and eval; vary later for robustness tests.

Optimize expected value per round as the primary target. The episode reward is the net units won or lost after all sub-hands finish (for example: normal win or loss is one unit; doubles count as two; blackjack uses your table payout; surrender loses half a unit, etc.). Track win, push, and loss percentages as secondary metrics.

Ensure the environment returns, at every decision, a mask of legal actions for stand, hit, double, split, and surrender.

Phase 1 — State representation and baselines

Use a compact numeric feature vector representing only the current sub-hand and minimal table and shoe context. Include player total, whether the hand is soft, whether it is a pair, number of cards in the hand, whether the bet has been doubled, the dealer upcard, how many splits occurred this round, the index of the current sub-hand and the total number of player sub-hands, decks remaining, running count, and true count.

Build a rule-correct basic-strategy baseline aligned with your exact rules. Optionally add a few true-count-conditioned overrides for borderline decisions. This baseline serves for sanity checks and as a source of demonstration data.

Phase 2 — Imitation pretraining (warm start)

Generate a large batch of gameplay using only the baseline policy. Record observations, the legal-action masks the environment provided, and the chosen baseline actions. Ensure the dataset includes a broad range of true-count buckets so the agent sees varied composition scenarios.

Train a supervised classifier that maps observations to one of the five actions while respecting the mask. Use cross-entropy loss applied only to legal actions. The goal is to reproduce the baseline reliably and to produce a stable starting point for reinforcement learning.

Phase 3 — Core reinforcement learner (Rainbow-style DQN)

Use a value-based agent with the following components: double Q-learning to reduce overestimation, a dueling architecture that separates state value and advantages, distributional value learning across discrete atoms (the C51 approach) to model the return distribution, prioritized replay to sample more informative transitions, parameter noise (NoisyNets) to drive exploration without an epsilon schedule, and multi-step returns to propagate outcome information more quickly.

Set practical starting hyperparameters in words: a discount factor of one because episodes are short; a large replay buffer capable of holding on the order of a million transitions; a batch size in the low thousands; an adaptive learning rate in the low e-4 range with gradient clipping; soft target-network updates of a small coefficient each optimization step; three-step returns; and prioritized replay that begins moderately biased toward high-error samples and anneals importance-sampling corrections toward one over training.

Enforce the action mask during both action selection and training targets by eliminating illegal actions from consideration. This single detail prevents the learner from chasing impossible choices and is crucial for stability.

Phase 4 — Curriculum to accelerate convergence

Start with hit, stand, and double only; train until performance stabilizes.

Enable surrender and continue training.

Enable splits with a practical cap on resplits; begin with aces and eights, then allow general pairs.

Keep the same network and optimizer throughout; simply expand the set of legal actions as rules unlock.

Phase 5 — Optional separate bet-sizing policy

Treat play and bet decisions as separate problems. After the play policy stabilizes, introduce a small bet-sizing policy that conditions on true count and shoe depth and chooses a wager from a small discrete set of units. Tie this policy’s reward to bankroll growth per hand, or to an approximation of the Kelly criterion. Train this head with a reliable policy-gradient method while holding the playing policy fixed. If you prefer simplicity, begin with a deterministic count-based schedule and skip learning here.

Phase 6 — Evaluation protocol (win rate and strategy depth)

Use very large deterministic evaluations with fixed random seeds and at least several million rounds per run. Report expected value per round and its confidence interval, win, push, and loss percentages and their intervals, blackjack frequency, the average number of hands per round, the rates of doubles, splits, and surrenders, and the distribution of true counts encountered during evaluation.

Produce granular, per-state analytics: group by player total, softness, pair status, and dealer upcard, and also by true-count buckets. For each group, report the action chosen by the policy, the empirical win rate, and the empirical expected value.

Run counterfactual analyses: for a large random subset of decision points, force each legal action separately and measure the resulting expected values. This reveals whether the policy’s chosen action is locally optimal and where it leaves value on the table.

Extract policy tables in the classic chart format: two grids, one for soft totals and one for hard totals, each indexed by player total and dealer upcard, with cells labeled by the chosen action. Generate these charts at neutral count and at several representative count buckets to visualize learned deviations from basic strategy.

Distill the learned policy into a compact, human-readable approximation by training a shallow decision tree to imitate the policy’s choices over a large sample. Report the imitation accuracy and, crucially, the expected-value gap between the distilled strategy and the original policy. This gives an “explainable card” and a way to audit decisions.

Phase 7 — Robustness checks and ablations

Vary individual rules one at a time while freezing the policy and measure expected-value changes. Compare dealer stands versus hits on 17, blackjack payout variations, double-after-split on and off, late surrender on and off, resplit limits, and penetration changes. Sensible directional shifts confirm the agent learned rule-aware play.

Evaluate with and without true-count features to quantify the value of shoe awareness.

Run sensitivity tests on mask integrity by intentionally toggling subsets of actions off in evaluation and verifying the agent degrades gracefully.

Phase 8 — Reproducibility, performance, and stopping

Fix seeds for the environment, the replay buffer sampling, and the neural network initialization. Record all configuration settings in a single manifest file per run.

Track training with rolling estimates of expected value and win rate on fixed evaluation suites. Stop when both flatten across multiple evaluations, the policy charts stabilize, counterfactual gaps shrink to noise, and ablations behave as expected.

Optimize simulator throughput early: keep observation construction lean, ensure settlement logic is efficient, and prefer long contiguous runs over frequent resets.

Phase 9 — Deliverables you should produce

Trained playing policy weights and, if used, a trained bet-sizing policy.

A strategy report containing global metrics with confidence intervals, per-state tables, policy charts for soft and hard totals at several count buckets, counterfactual action value summaries, and a section explaining notable deviations from basic strategy.

The distilled decision-tree strategy with its accuracy and expected-value gap relative to the full policy.

A reproducibility bundle with seeds, configuration, environment rule settings, and a precise description of the reward mapping used.

A short “operations” note that defines when to refresh the policy, how to re-evaluate after any rule change, and how to monitor expected value and risk over time.

Practical starting values and expectations (text only)

• Network capacity: three hidden layers with a few hundred units each using rectified activations; a dueling value head and an advantage head over five actions; a distributional output with a few dozen atoms spanning a conservative value range that covers losses from surrenders through gains from multiple doubles and blackjacks.
• Training length: tens to low hundreds of millions of environment steps for the play head when using the curriculum and imitation warm start; evaluation every roughly one million steps with multi-million-round test suites.
• Replay and returns: a large prioritized replay, multi-step targets of small depth to speed propagation, and soft target updates for stability.
• Exploration: parameter noise rather than an epsilon schedule, supplemented by the prioritized sampler to keep learning focused.
• Masking: strict use of the legal-action mask during both action choice and value computation; this cannot be optional.
• Objective: always judge success primarily by expected value per round under the fixed rules. Use win rate and push rate as diagnostics, not as the optimization target.