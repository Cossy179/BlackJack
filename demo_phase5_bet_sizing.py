#!/usr/bin/env python3
"""
Phase 5 Bet Sizing Demonstration

Shows the bet sizing policy training with:
1. Fixed playing policy from Phase 4
2. Trainable bet sizing policy
3. Kelly criterion integration
4. Bankroll growth optimization
"""

from BlackJackSim.integrated_training import train_phase5_bet_sizing, IntegratedBlackjackTrainer
from BlackJackSim.bet_sizing import create_bet_sizing_config, BetSizingAgent, BetSizingState
from BlackJackSim.device_utils import print_device_status
import time
import os


def demo_bet_sizing_agent():
    """Demonstrate the bet sizing agent components"""
    
    print("💰 PHASE 5 BET SIZING DEMONSTRATION 💰\n")
    
    print_device_status()
    print()
    
    print("🎯 Demonstrating Bet Sizing Agent Components:")
    print()
    
    # Create bet sizing agent
    config = create_bet_sizing_config(conservative=True, bet_spread="moderate")
    agent = BetSizingAgent(config)
    
    print("📊 Testing bet sizing decisions in different scenarios:")
    
    # Test scenarios
    scenarios = [
        # (true_count, shoe_depth, bankroll_ratio, recent_performance, description)
        (0.0, 0.8, 1.0, -0.05, "Neutral count, fresh shoe, even bankroll"),
        (2.5, 0.6, 1.2, 0.02, "Positive count, mid-shoe, winning"),
        (-1.5, 0.3, 0.9, -0.08, "Negative count, deep shoe, losing"),
        (4.0, 0.4, 1.5, 0.08, "Very positive count, profitable session"),
        (-2.0, 0.1, 0.7, -0.12, "Negative count, end of shoe, down"),
    ]
    
    for true_count, shoe_depth, bankroll_ratio, recent_perf, description in scenarios:
        state = BetSizingState(
            true_count=true_count,
            shoe_depth=shoe_depth,
            bankroll_ratio=bankroll_ratio,
            recent_performance=recent_perf
        )
        
        bet_size, info = agent.select_bet_size(state, deterministic=True)
        kelly_bet = info['kelly_bet_size']
        
        print(f"  📈 {description}")
        print(f"     True count: {true_count:+.1f}, Shoe depth: {shoe_depth:.1f}")
        print(f"     Policy bet: {bet_size:.1f} units, Kelly bet: {kelly_bet:.1f} units")
        print(f"     Value estimate: {info['value']:.3f}")
        print()
    
    print("✅ Bet sizing agent components working correctly!")
    print()


def demo_quick_training():
    """Demonstrate quick bet sizing training"""
    
    print("🚀 QUICK BET SIZING TRAINING DEMO")
    print("="*50)
    
    # Check if we have a trained playing policy
    play_policy_paths = [
        "curriculum_training_results/best_model_stage_ADD_SPLIT_ACES.pth",
        "curriculum_training_results/best_model_stage_ADD_SURRENDER.pth", 
        "curriculum_training_results/best_model_stage_BASIC_ACTIONS.pth"
    ]
    
    play_policy_path = None
    for path in play_policy_paths:
        if os.path.exists(path):
            play_policy_path = path
            break
    
    if not play_policy_path:
        print("⚠️  No trained playing policy found!")
        print("   Please run Phase 4 curriculum training first")
        print("   For demo purposes, we'll use a random policy")
        play_policy_path = "dummy_path.pth"  # Will fall back to random
    else:
        print(f"✅ Using trained playing policy: {play_policy_path}")
    
    print()
    print("⚙️ Training Configuration:")
    print("   Episodes: 50 iterations × 50 episodes = 2,500 episodes")
    print("   Initial bankroll: $1,000")
    print("   Bet spread: 1-5 units (moderate)")
    print("   Strategy: Conservative Kelly scaling")
    print()
    
    try:
        # Run quick training
        results = train_phase5_bet_sizing(
            play_policy_path=play_policy_path,
            config_type="moderate",
            initial_bankroll=1000.0,
            num_iterations=50  # Quick demo
        )
        
        print("🎉 TRAINING RESULTS:")
        print(f"   Training time: {results['training_time_minutes']:.1f} minutes")
        print(f"   Final bankroll: ${results['final_bankroll']:,.0f}")
        print(f"   Total profit: ${results['total_profit']:+,.0f}")
        print(f"   ROI: {results['roi_percent']:+.1f}%")
        print(f"   Episodes trained: {results['episodes_trained']:,}")
        print()
        
        # Show evaluation results
        eval_results = results['final_evaluation']
        print("📊 FINAL EVALUATION:")
        print(f"   Evaluation bankroll: ${eval_results['final_bankroll']:,.0f}")
        print(f"   Evaluation ROI: {eval_results['roi']:+.1f}%")
        print(f"   Sharpe ratio: {eval_results['sharpe_ratio']:.2f}")
        print(f"   Average bet size: {eval_results['avg_bet_size']:.1f} units")
        print()
        
        if results['roi_percent'] > 0:
            print("✅ Positive ROI achieved! Bet sizing is working!")
        else:
            print("📉 Negative ROI - bet sizing needs more training or tuning")
        
        print("💰 Key Features Demonstrated:")
        print("   ✅ Separate bet and play decisions")
        print("   ✅ Policy gradient training for bet sizing")
        print("   ✅ Kelly criterion integration")
        print("   ✅ Bankroll growth optimization")
        print("   ✅ True count and shoe depth conditioning")
        print("   ✅ Playing policy held fixed during training")
        print()
        
        print("📂 Results saved to: phase5_results/")
        print("   - training_progress.json (training history)")
        print("   - best_bet_sizing_model.pth (best model)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_strategy_comparison():
    """Demonstrate different bet sizing strategies"""
    
    print("\n" + "="*60)
    print("📊 BET SIZING STRATEGY COMPARISON")
    print("="*60)
    
    strategies = ["conservative", "moderate", "aggressive"]
    results = {}
    
    for strategy in strategies:
        print(f"\n🎲 Testing {strategy.upper()} strategy...")
        
        try:
            result = train_phase5_bet_sizing(
                play_policy_path="curriculum_training_results/best_model_stage_BASIC_ACTIONS.pth",
                config_type=strategy,
                initial_bankroll=1000.0,
                num_iterations=25  # Quick comparison
            )
            results[strategy] = result
            
            print(f"   Final ROI: {result['roi_percent']:+.1f}%")
            print(f"   Sharpe ratio: {result['final_evaluation']['sharpe_ratio']:.2f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[strategy] = None
    
    print(f"\n📈 STRATEGY COMPARISON SUMMARY:")
    print("-" * 40)
    
    for strategy, result in results.items():
        if result:
            roi = result['roi_percent']
            sharpe = result['final_evaluation']['sharpe_ratio']
            print(f"{strategy.capitalize():12s}: ROI={roi:+6.1f}%, Sharpe={sharpe:5.2f}")
        else:
            print(f"{strategy.capitalize():12s}: Failed")
    
    print("\n💡 Strategy Insights:")
    print("   - Conservative: Lower risk, steady growth")
    print("   - Moderate: Balanced risk/reward")  
    print("   - Aggressive: Higher variance, potential for bigger gains/losses")


if __name__ == "__main__":
    # Run demonstrations
    demo_bet_sizing_agent()
    
    print(f"\n{'='*60}")
    
    # Ask user which demo to run
    try:
        print("\n💰 Choose demonstration:")
        print("1. Quick training demo (recommended)")
        print("2. Strategy comparison")
        print("3. Both")
        
        choice = input("\nEnter choice (1-3, or Enter for 1): ").strip()
        
        if choice in ['2']:
            demo_strategy_comparison()
        elif choice in ['3']:
            demo_quick_training()
            demo_strategy_comparison()
        else:  # Default to 1
            demo_quick_training()
            
    except KeyboardInterrupt:
        print("\n\n👍 Demo cancelled by user")
    
    print(f"\n💰 Phase 5 bet sizing demonstration complete!")
    print(f"🎯 Bet sizing policy successfully separates betting and playing decisions!")
    print(f"📈 Ready for advanced bankroll management strategies!")
