"""
CUDA Integration Test

Comprehensive test to verify all components use CUDA/GPU with CPU fallback.
"""

import torch
import numpy as np
import time
from BlackJackSim.device_utils import (
    device_manager, get_device, is_cuda_available, 
    print_device_status, get_memory_info, clear_gpu_cache
)
from BlackJackSim.data_generation import generate_demonstration_dataset
from BlackJackSim.imitation_learning import (
    BlackjackPolicyNet, ImitationTrainer, BlackjackDataset, 
    train_imitation_model
)
from BlackJackSim.state_representation import get_feature_dimensions


def test_device_manager():
    """Test centralized device manager"""
    print("=== Testing Device Manager ===")
    
    print(f"Device: {get_device()}")
    print(f"CUDA available: {is_cuda_available()}")
    
    # Test device info
    device_info = device_manager.device_info
    print(f"Device info: {device_info}")
    
    # Test tensor creation
    x = device_manager.tensor([1, 2, 3, 4, 5])
    y = device_manager.randn(3, 3)
    z = device_manager.zeros(2, 4)
    
    print(f"Tensor devices: {x.device}, {y.device}, {z.device}")
    assert x.device.type == get_device().type, "Tensor should be on correct device type"
    
    # Test numpy conversion
    np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    tensor = device_manager.from_numpy(np_array)
    print(f"Numpy->Tensor device: {tensor.device}")
    
    print("✅ Device manager validated\n")


def test_neural_network_device():
    """Test neural network device handling"""
    print("=== Testing Neural Network Device Handling ===")
    
    # Create model
    model = BlackjackPolicyNet()
    print(f"Model device before: {next(model.parameters()).device}")
    
    # Move to device using device manager
    model = device_manager.to_device(model)
    print(f"Model device after: {next(model.parameters()).device}")
    
    # Test forward pass
    batch_size = 32
    observations = device_manager.randn(batch_size, 12)
    
    with torch.no_grad():
        logits = model(observations)
        print(f"Input device: {observations.device}")
        print(f"Output device: {logits.device}")
    
    assert observations.device.type == get_device().type, "Input should be on correct device type"
    assert logits.device.type == get_device().type, "Output should be on correct device type"
    
    # Test action prediction
    single_obs = device_manager.randn(12)
    legal_mask = device_manager.tensor([1, 1, 0, 1, 0], dtype=torch.float32)
    
    action = model.predict_action(single_obs, legal_mask)
    print(f"Predicted action: {action}")
    assert isinstance(action, int), "Action should be integer"
    
    print("✅ Neural network device handling validated\n")


def test_dataset_device():
    """Test dataset device handling"""
    print("=== Testing Dataset Device Handling ===")
    
    # Create sample data
    num_samples = 100
    observations = np.random.randn(num_samples, 12).astype(np.float32)
    actions = np.random.randint(0, 5, num_samples)
    legal_masks = np.random.randint(0, 2, (num_samples, 5)).astype(np.float32)
    
    # Create dataset
    dataset = BlackjackDataset(observations, actions, legal_masks)
    
    # Check device placement
    sample = dataset[0]
    print(f"Sample observation device: {sample['observation'].device}")
    print(f"Sample action device: {sample['action'].device}")
    print(f"Sample legal_mask device: {sample['legal_mask'].device}")
    
    # All should be on the same device type
    target_device = get_device()
    assert sample['observation'].device.type == target_device.type
    assert sample['action'].device.type == target_device.type
    assert sample['legal_mask'].device.type == target_device.type
    
    print("✅ Dataset device handling validated\n")


def test_training_device():
    """Test training with device handling"""
    print("=== Testing Training Device Handling ===")
    
    # Generate small dataset
    episodes, _ = generate_demonstration_dataset(
        num_episodes=50,
        target_steps=200,
        seed=42
    )
    
    print_device_status()
    
    # Train model
    start_time = time.time()
    model, trainer = train_imitation_model(
        episodes=episodes,
        num_epochs=5,
        batch_size=32,
        learning_rate=1e-3
    )
    training_time = time.time() - start_time
    
    # Check model device
    model_device = next(model.parameters()).device
    print(f"Trained model device: {model_device}")
    assert model_device.type == get_device().type, "Model should be on correct device type"
    
    # Test memory usage
    if is_cuda_available():
        memory_info = get_memory_info()
        print(f"GPU memory usage: {memory_info}")
    
    print(f"Training time: {training_time:.2f}s")
    print("✅ Training device handling validated\n")


def test_device_switching():
    """Test switching between devices"""
    print("=== Testing Device Switching ===")
    
    original_device = get_device()
    print(f"Original device: {original_device}")
    
    # Test switching to CPU
    device_manager.set_device('cpu')
    cpu_device = get_device()
    print(f"After switching to CPU: {cpu_device}")
    assert cpu_device.type == 'cpu', "Should switch to CPU"
    
    # Create tensor on CPU
    cpu_tensor = device_manager.tensor([1, 2, 3])
    print(f"CPU tensor device: {cpu_tensor.device}")
    assert cpu_tensor.device.type == 'cpu', "Tensor should be on CPU"
    
    # Switch back to original device
    device_manager.set_device(original_device)
    restored_device = get_device()
    print(f"After switching back: {restored_device}")
    assert restored_device.type == original_device.type, "Should restore original device type"
    
    print("✅ Device switching validated\n")


def test_memory_management():
    """Test GPU memory management"""
    print("=== Testing Memory Management ===")
    
    if not is_cuda_available():
        print("CUDA not available, skipping memory tests")
        return
    
    # Reset memory stats
    device_manager.reset_peak_memory()
    
    # Create large tensors
    large_tensors = []
    for i in range(5):
        tensor = device_manager.randn(1000, 1000)
        large_tensors.append(tensor)
    
    memory_info = get_memory_info()
    print(f"Memory after creating tensors: {memory_info}")
    
    # Clear references
    del large_tensors
    
    # Clear cache
    clear_gpu_cache()
    
    memory_info_after = get_memory_info()
    print(f"Memory after clearing: {memory_info_after}")
    
    # Memory should be reduced
    assert memory_info_after['allocated_gb'] < memory_info['allocated_gb'], "Memory should be freed"
    
    print("✅ Memory management validated\n")


def test_full_pipeline():
    """Test full pipeline with device handling"""
    print("=== Testing Full Pipeline ===")
    
    print("Initial device status:")
    print_device_status()
    
    # 1. Generate data
    print("1. Generating demonstration data...")
    episodes, analysis = generate_demonstration_dataset(
        num_episodes=100,
        target_steps=400,
        seed=42
    )
    
    # 2. Train model
    print("2. Training model...")
    start_time = time.time()
    model, trainer = train_imitation_model(
        episodes=episodes,
        num_epochs=10,
        batch_size=64,
        learning_rate=1e-3
    )
    training_time = time.time() - start_time
    
    # 3. Test inference
    print("3. Testing inference...")
    num_predictions = 1000
    
    # Create test data
    test_obs = device_manager.randn(num_predictions, 12)
    test_masks = device_manager.randint(0, 2, (num_predictions, 5)).float()
    test_masks[:, 0] = 1  # Ensure STAND is always legal
    
    # Time inference
    start_time = time.time()
    with torch.no_grad():
        logits = model(test_obs)
        masked_logits = logits.clone()
        masked_logits[test_masks == 0] = -1e9
        predictions = torch.argmax(masked_logits, dim=-1)
    inference_time = time.time() - start_time
    
    print(f"Training time: {training_time:.2f}s")
    print(f"Inference time: {inference_time:.4f}s for {num_predictions} predictions")
    print(f"Inference speed: {num_predictions/inference_time:.0f} predictions/second")
    
    # 4. Final device status
    print("4. Final device status:")
    print_device_status()
    
    if is_cuda_available():
        memory_info = get_memory_info()
        print(f"Final GPU memory: {memory_info['allocated_gb']:.2f} GB")
    
    print("✅ Full pipeline validated\n")


def main():
    """Run all CUDA integration tests"""
    print("🚀 CUDA INTEGRATION TESTS 🚀\n")
    
    try:
        test_device_manager()
        test_neural_network_device()
        test_dataset_device()
        test_training_device()
        test_device_switching()
        test_memory_management()
        test_full_pipeline()
        
        print("🎉 ALL CUDA INTEGRATION TESTS PASSED! 🎉")
        print("\n📋 Summary:")
        print(f"   Device: {get_device()}")
        print(f"   CUDA Available: {is_cuda_available()}")
        
        if is_cuda_available():
            info = device_manager.device_info
            memory = get_memory_info()
            print(f"   GPU: {info['name']}")
            print(f"   Memory: {memory['allocated_gb']:.2f}/{info['memory_gb']:.1f} GB")
        
        print("\n✅ All components now use CUDA with CPU fallback!")
        print("✅ Ready for Phase 3 Rainbow DQN implementation!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
