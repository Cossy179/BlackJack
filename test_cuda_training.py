"""
Test CUDA Training

Quick test to verify that our imitation learning works with CUDA acceleration.
"""

import torch
import time
import numpy as np
from BlackJackSim.data_generation import generate_demonstration_dataset
from BlackJackSim.imitation_learning import train_imitation_model


def test_cuda_training():
    """Test imitation learning with CUDA acceleration"""
    
    print("🎯 TESTING CUDA TRAINING 🎯\n")
    
    # Check CUDA availability
    print("=== CUDA Status ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Generate demonstration data
    print("=== Generating Demonstration Data ===")
    episodes, analysis = generate_demonstration_dataset(
        num_episodes=200,
        target_steps=1000,
        seed=42
    )
    
    print(f"Generated: {analysis['num_episodes']} episodes, {analysis['total_steps']} steps")
    print(f"Action distribution: {analysis['action_distribution']}")
    print()
    
    # Test training with timing
    print("=== Training with CUDA ===")
    start_time = time.time()
    
    model, trainer = train_imitation_model(
        episodes=episodes,
        train_split=0.8,
        batch_size=128,  # Larger batch for GPU
        num_epochs=20,
        learning_rate=1e-3
    )
    
    training_time = time.time() - start_time
    
    print(f"\n⏱️ Training completed in {training_time:.2f} seconds")
    print(f"Final training accuracy: {trainer.train_accuracies[-1]:.4f}")
    print(f"Final validation accuracy: {trainer.val_accuracies[-1]:.4f}")
    
    # Test model performance
    print("\n=== Model Performance Test ===")
    
    # Generate random batch for speed test
    device = next(model.parameters()).device
    batch_size = 1000
    
    # Random observations and legal masks
    observations = torch.randn(batch_size, 12).to(device)
    legal_masks = torch.randint(0, 2, (batch_size, 5)).float().to(device)
    # Ensure at least one legal action per sample
    legal_masks[:, 0] = 1  # STAND always legal
    
    # Time inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):  # 100 batches
            logits = model(observations)
            # Apply legal mask
            masked_logits = logits.clone()
            masked_logits[legal_masks == 0] = -1e9
            probs = torch.softmax(masked_logits, dim=-1)
    
    inference_time = time.time() - start_time
    total_predictions = 100 * batch_size
    
    print(f"Inference speed: {total_predictions / inference_time:.0f} predictions/second")
    print(f"Device used: {device}")
    
    # Memory usage
    if device.type == 'cuda':
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {memory_used:.2f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
    
    print("\n✅ CUDA training test completed successfully!")


def compare_cpu_vs_gpu():
    """Compare training speed between CPU and GPU"""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping comparison")
        return
    
    print("\n🏁 CPU vs GPU SPEED COMPARISON 🏁\n")
    
    # Generate small dataset for quick comparison
    episodes, _ = generate_demonstration_dataset(
        num_episodes=100,
        target_steps=500,
        seed=42
    )
    
    results = {}
    
    for device_name in ['cpu', 'cuda']:
        print(f"=== Training on {device_name.upper()} ===")
        
        start_time = time.time()
        
        # Force device for this test
        original_device_check = torch.cuda.is_available
        if device_name == 'cpu':
            torch.cuda.is_available = lambda: False
        
        try:
            model, trainer = train_imitation_model(
                episodes=episodes,
                train_split=0.8,
                batch_size=64,
                num_epochs=10,
                learning_rate=1e-3
            )
            
            training_time = time.time() - start_time
            final_accuracy = trainer.train_accuracies[-1]
            
            results[device_name] = {
                'time': training_time,
                'accuracy': final_accuracy
            }
            
            print(f"Time: {training_time:.2f}s, Accuracy: {final_accuracy:.4f}")
            
        finally:
            # Restore original function
            torch.cuda.is_available = original_device_check
        
        print()
    
    # Compare results
    if 'cpu' in results and 'cuda' in results:
        speedup = results['cpu']['time'] / results['cuda']['time']
        print(f"🚀 GPU Speedup: {speedup:.2f}x faster than CPU")
        print(f"CPU time: {results['cpu']['time']:.2f}s")
        print(f"GPU time: {results['cuda']['time']:.2f}s")


if __name__ == "__main__":
    test_cuda_training()
    compare_cpu_vs_gpu()
