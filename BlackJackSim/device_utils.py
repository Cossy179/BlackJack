"""
Device Management Utilities

Centralized CUDA/GPU device detection and management for consistent 
device handling across all components.
"""

import torch
import numpy as np
from typing import Optional, Union, Any
import warnings


class DeviceManager:
    """
    Centralized device management for CUDA/CPU operations.
    
    Provides consistent device selection, memory management, and 
    tensor operations across all blackjack RL components.
    """
    
    _instance = None
    _device = None
    _device_info = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize device detection and selection"""
        self._detect_and_select_device()
        self._print_device_info()
    
    def _detect_and_select_device(self):
        """Detect and select the best available device"""
        if torch.cuda.is_available():
            # Check if CUDA is actually usable
            try:
                test_tensor = torch.tensor([1.0]).cuda()
                _ = test_tensor + test_tensor  # Simple operation test
                self._device = torch.device('cuda')
                self._device_info = {
                    'type': 'cuda',
                    'name': torch.cuda.get_device_name(0),
                    'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                    'compute_capability': torch.cuda.get_device_properties(0).major,
                    'device_count': torch.cuda.device_count()
                }
            except Exception as e:
                warnings.warn(f"CUDA detected but not usable: {e}. Falling back to CPU.")
                self._device = torch.device('cpu')
                self._device_info = {'type': 'cpu'}
        else:
            self._device = torch.device('cpu')
            self._device_info = {'type': 'cpu'}
    
    def _print_device_info(self):
        """Print device information"""
        if self._device_info['type'] == 'cuda':
            print(f"🚀 GPU Acceleration Enabled!")
            print(f"   Device: {self._device_info['name']}")
            print(f"   Memory: {self._device_info['memory_gb']:.1f} GB")
            print(f"   Compute: {self._device_info['compute_capability']}.x")
            if self._device_info['device_count'] > 1:
                print(f"   GPUs Available: {self._device_info['device_count']}")
        else:
            print(f"💻 Using CPU (CUDA not available)")
    
    @property
    def device(self) -> torch.device:
        """Get the selected device"""
        return self._device
    
    @property
    def device_type(self) -> str:
        """Get device type string ('cuda' or 'cpu')"""
        return self._device.type
    
    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA"""
        return self._device.type == 'cuda'
    
    @property
    def device_info(self) -> dict:
        """Get device information"""
        return self._device_info.copy()
    
    def to_device(self, tensor_or_module: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
        """Move tensor or module to the selected device"""
        return tensor_or_module.to(self._device)
    
    def tensor(self, data: Any, **kwargs) -> torch.Tensor:
        """Create tensor on the selected device"""
        return torch.tensor(data, device=self._device, **kwargs)
    
    def zeros(self, *args, **kwargs) -> torch.Tensor:
        """Create zeros tensor on the selected device"""
        return torch.zeros(*args, device=self._device, **kwargs)
    
    def ones(self, *args, **kwargs) -> torch.Tensor:
        """Create ones tensor on the selected device"""
        return torch.ones(*args, device=self._device, **kwargs)
    
    def randn(self, *args, **kwargs) -> torch.Tensor:
        """Create random normal tensor on the selected device"""
        return torch.randn(*args, device=self._device, **kwargs)
    
    def randint(self, *args, **kwargs) -> torch.Tensor:
        """Create random integer tensor on the selected device"""
        return torch.randint(*args, device=self._device, **kwargs)
    
    def from_numpy(self, array: np.ndarray, **kwargs) -> torch.Tensor:
        """Convert numpy array to tensor on the selected device"""
        return torch.from_numpy(array).to(self._device, **kwargs)
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA"""
        if self.is_cuda:
            torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage information"""
        if self.is_cuda:
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'cached_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
                'total_gb': self._device_info['memory_gb']
            }
        else:
            return {'type': 'cpu', 'memory_tracking': 'not_available'}
    
    def reset_peak_memory(self):
        """Reset peak memory statistics"""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
    
    def set_device(self, device: Union[str, torch.device, None] = None):
        """
        Override device selection (for testing or specific requirements).
        
        Args:
            device: Device to use ('cuda', 'cpu', or torch.device object)
                   If None, will auto-detect best device
        """
        if device is None:
            self._detect_and_select_device()
        else:
            if isinstance(device, str):
                device = torch.device(device)
            
            # Validate device is available
            if device.type == 'cuda' and not torch.cuda.is_available():
                warnings.warn("CUDA requested but not available. Using CPU.")
                device = torch.device('cpu')
            
            self._device = device
            if device.type == 'cuda':
                self._device_info = {
                    'type': 'cuda',
                    'name': torch.cuda.get_device_name(0),
                    'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                    'compute_capability': torch.cuda.get_device_properties(0).major,
                    'device_count': torch.cuda.device_count()
                }
            else:
                self._device_info = {'type': 'cpu'}


# Global device manager instance
device_manager = DeviceManager()


# Convenience functions for easy access
def get_device() -> torch.device:
    """Get the current device"""
    return device_manager.device


def get_device_type() -> str:
    """Get device type string"""
    return device_manager.device_type


def is_cuda_available() -> bool:
    """Check if CUDA is being used"""
    return device_manager.is_cuda


def to_device(tensor_or_module: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
    """Move tensor or module to the current device"""
    return device_manager.to_device(tensor_or_module)


def create_tensor(data: Any, **kwargs) -> torch.Tensor:
    """Create tensor on the current device"""
    return device_manager.tensor(data, **kwargs)


def from_numpy(array: np.ndarray, **kwargs) -> torch.Tensor:
    """Convert numpy array to tensor on current device"""
    return device_manager.from_numpy(array, **kwargs)


def clear_gpu_cache():
    """Clear GPU cache"""
    device_manager.clear_cache()


def get_memory_info() -> dict:
    """Get memory usage information"""
    return device_manager.get_memory_usage()


def print_device_status():
    """Print current device status and memory usage"""
    info = device_manager.device_info
    memory = device_manager.get_memory_usage()
    
    print(f"📱 Device Status:")
    print(f"   Type: {info['type']}")
    
    if info['type'] == 'cuda':
        print(f"   GPU: {info['name']}")
        print(f"   Memory: {memory['allocated_gb']:.2f}/{info['memory_gb']:.1f} GB")
        print(f"   Peak: {memory['max_allocated_gb']:.2f} GB")
    
    print(f"   Device: {device_manager.device}")


# Auto-initialize on import
def initialize_device_manager():
    """Initialize device manager and print status"""
    global device_manager
    device_manager = DeviceManager()
    return device_manager


# Initialize when module is imported
if device_manager._device is None:
    initialize_device_manager()

