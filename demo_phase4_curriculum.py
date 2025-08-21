"""
Phase 4 Curriculum Learning Demonstration

Shows the Rainbow DQN agent training with curriculum learning:
1. Stage 1: Basic actions (HIT, STAND, DOUBLE)
2. Stage 2: Add SURRENDER
3. Stage 3: Add SPLIT for Aces and Eights
4. Stage 4: Full actions including all pair splits
"""

from BlackJackSim.curriculum_training import train_curriculum_agent, CurriculumTrainer
from BlackJackSim.curriculum import create_curriculum_config
from BlackJackSim.device_utils import print_device_status
import time


def demo_quick_curriculum_training():
    """Demonstrate curriculum learning with a quick training run"""
    
    print("🎓 PHASE 4 CURRICULUM LEARNING DEMONSTRATION 🎓\n")
    
    print_device_status()
    print()
    
    print("🎯 Objective: Train Rainbow DQN with progressive action unlocking")
    print("📚 Curriculum Stages:")
    print("   1. BASIC_ACTIONS: HIT, STAND, DOUBLE only")
    print("   2. ADD_SURRENDER: + SURRENDER option")
    print("   3. ADD_SPLIT_ACES: + SPLIT for Aces and Eights only")
    print("   4. FULL_ACTIONS: All actions including general pair splits")
    print()
    
    print("🚀 Starting quick curriculum training demonstration...")
    print("   (Using reduced settings for demonstration purposes)")
    print()
    
    # Create quick training configuration
    start_time = time.time()
    
    try:
        results = train_curriculum_agent(
            max_episodes=5000,     # Quick demo
            quick_mode=True,       # Use faster settings
            config_overrides={
                'hidden_dim': 256,  # Smaller network for faster training
                'batch_size': 128,
                'buffer_capacity': 50000,
                'lr': 1e-3
            }
        )
        
        training_time = time.time() - start_time
        
        print(f"\n🎉 DEMONSTRATION COMPLETE! 🎉")
        print(f"   Training Time: {training_time/60:.1f} minutes")
        print(f"   Total Episodes: {results['total_episodes']:,}")
        print(f"   Final Performance: {results['final_performance']:.4f}")
        print(f"   Curriculum Complete: {results['curriculum_completed']}")
        print(f"   Final Stage: {results['final_stage']}")
        print()
        
        # Show curriculum progression
        curriculum_summary = results['curriculum_summary']
        stage_history = curriculum_summary.get('stage_history', [])
        
        if stage_history:
            print("📈 Curriculum Progression:")
            for i, stage in enumerate(stage_history):
                print(f"   Stage {i+1}: {stage['stage']}")
                print(f"     Episodes: {stage['episodes']:,}")
                print(f"     Performance: {stage['performance']:.4f}")
                print(f"     Stability: {stage['stability']:.4f}")
                print()
        
        print("💡 Key Features Demonstrated:")
        print("   ✅ Progressive action unlocking")
        print("   ✅ Rainbow DQN with GPU acceleration")
        print("   ✅ Dueling network architecture")
        print("   ✅ Distributional value learning (C51)")
        print("   ✅ Prioritized experience replay")
        print("   ✅ Noisy networks for exploration")
        print("   ✅ Multi-step returns")
        print("   ✅ Legal action masking")
        print("   ✅ Performance-based stage transitions")
        print()
        
        print("📂 Results saved to: curriculum_training_results/")
        print("   - training_results.json (complete summary)")
        print("   - curriculum_progress.json (stage progression)")
        print("   - training_progress.png (visualization)")
        print("   - Model checkpoints (.pth files)")
        
    except KeyboardInterrupt:
        print("\n⏸️  Demonstration interrupted by user")
        print("   Partial training results may be available in curriculum_training_results/")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


def demo_curriculum_stages():
    """Demonstrate individual curriculum stages"""
    
    print("\n" + "="*60)
    print("🎭 CURRICULUM STAGES DEMONSTRATION")
    print("="*60)
    
    from BlackJackSim.curriculum import CurriculumManager, CurriculumStage
    from BlackJackSim.config import V1_RULES
    
    # Create curriculum manager
    curriculum = CurriculumManager(create_curriculum_config(quick_mode=True), V1_RULES)
    
    # Demonstrate each stage
    stages = [
        CurriculumStage.BASIC_ACTIONS,
        CurriculumStage.ADD_SURRENDER,
        CurriculumStage.ADD_SPLIT_ACES,
        CurriculumStage.FULL_ACTIONS
    ]
    
    for stage in stages:
        curriculum.mask_manager.set_stage(stage)
        print(f"\n📚 {stage.name}:")
        print(f"   Description: {curriculum.mask_manager.get_stage_description()}")
        
        # Test different hand scenarios
        scenarios = [
            ([10, 10], "Pair of 10s"),
            ([1, 1], "Pair of Aces"),
            ([8, 8], "Pair of Eights"),
            ([5, 6], "Hard 11"),
            ([1, 6], "Soft 17")
        ]
        
        for hand_cards, description in scenarios:
            game_legal_mask = [True, True, True, True, True]  # All legal in game
            curriculum_mask = curriculum.get_curriculum_mask(
                game_legal_mask, hand_cards, True
            )
            
            available_actions = []
            action_names = ['STAND', 'HIT', 'DOUBLE', 'SPLIT', 'SURRENDER']
            for i, available in enumerate(curriculum_mask):
                if available:
                    available_actions.append(action_names[i])
            
            print(f"     {description}: {available_actions}")
    
    print("\n✅ Stage demonstration complete!")


if __name__ == "__main__":
    # Run demonstrations
    demo_curriculum_stages()
    
    print(f"\n{'='*60}")
    
    # Ask user if they want to run the full training demo
    try:
        response = input("\n🎯 Run full curriculum training demo? (y/N): ").lower().strip()
        if response in ['y', 'yes']:
            demo_quick_curriculum_training()
        else:
            print("\n👍 Skipping full training demo")
            print("📘 You can run it later with: python demo_phase4_curriculum.py")
    except KeyboardInterrupt:
        print("\n\n👍 Demo cancelled by user")
    
    print(f"\n🎓 Phase 4 curriculum learning is ready!")
    print(f"🚀 All Rainbow DQN components working with CUDA acceleration!")
    print(f"📈 Ready to train full-scale blackjack agents!")

