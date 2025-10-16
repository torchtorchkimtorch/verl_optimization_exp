#!/usr/bin/env python3
"""
Test script for GPU monitoring functionality.
"""

import time
import torch
import argparse
from verl.utils.gpu_monitor import GPUResourceMonitor


def create_gpu_load(device_id: int, duration: int = 10):
    """Create some GPU load for testing."""
    print(f"Creating GPU load on device {device_id} for {duration} seconds...")
    
    device = torch.device(f'cuda:{device_id}')
    
    # Create some tensors and do computations
    size = 1000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    start_time = time.time()
    while time.time() - start_time < duration:
        # Matrix multiplication to create GPU load
        c = torch.matmul(a, b)
        # Add some memory allocation
        temp = torch.randn(size, size, device=device)
        del temp
        time.sleep(0.1)
    
    print(f"GPU load test completed on device {device_id}")


def test_monitoring():
    """Test GPU monitoring functionality."""
    print("🧪 Testing GPU monitoring functionality...")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot test GPU monitoring.")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"📊 Found {num_gpus} GPU(s)")
    
    # Test basic monitoring
    print("\n1️⃣ Testing basic GPU monitoring...")
    monitor = GPUResourceMonitor(
        monitor_interval=0.5,
        log_to_wandb=False,
        log_to_console=True,
        device_ids=[0] if num_gpus > 0 else []
    )
    
    monitor.start_monitoring()
    time.sleep(2)
    
    # Get initial metrics
    initial_metrics = monitor.get_latest_metrics()
    if initial_metrics:
        print("✓ Successfully collected initial metrics")
        print(f"  🔥 GPU 0 utilization: {initial_metrics.get('gpu_0_gpu_utilization_pct', 'N/A')}%")
        print(f"  💾 GPU 0 memory (PyTorch): {initial_metrics.get('gpu_0_memory_allocated_gb', 'N/A')} GB")
        print(f"  💾 GPU 0 memory (NVML): {initial_metrics.get('gpu_0_memory_utilization_pct', 'N/A'):.1f}%")
        print(f"  🌡️  GPU 0 temperature: {initial_metrics.get('gpu_0_temperature_c', 'N/A')}°C")
        print(f"  ⚡ GPU 0 power: {initial_metrics.get('gpu_0_power_usage_w', 'N/A')}W")
        print(f"  🖥️  System: CPU {initial_metrics.get('cpu_utilization_pct', 'N/A'):.1f}% | RAM {initial_metrics.get('system_memory_utilization_pct', 'N/A'):.1f}%")
    else:
        print("❌ Failed to collect initial metrics")
        monitor.stop_monitoring()
        return False
    
    # Test with GPU load
    print("\n2️⃣ Testing with GPU load...")
    if num_gpus > 0:
        import threading
        load_thread = threading.Thread(target=create_gpu_load, args=(0, 5))
        load_thread.start()
        
        time.sleep(6)  # Wait for load test to complete
        load_thread.join()
        
        # Get metrics after load
        final_metrics = monitor.get_latest_metrics()
        if final_metrics:
            print("✓ Successfully collected metrics during load")
            print(f"  🔥 GPU 0 utilization: {final_metrics.get('gpu_0_gpu_utilization_pct', 'N/A')}%")
            print(f"  💾 GPU 0 memory (PyTorch): {final_metrics.get('gpu_0_memory_allocated_gb', 'N/A')} GB")
            print(f"  💾 GPU 0 memory (NVML): {final_metrics.get('gpu_0_memory_utilization_pct', 'N/A'):.1f}%")
        
    monitor.stop_monitoring()
    
    # Test metrics history
    print("\n3️⃣ Testing metrics history...")
    history = monitor.get_metrics_history()
    print(f"✓ Collected {len(history)} metric samples")
    
    if len(history) > 0:
        print(f"  First sample timestamp: {history[0].get('timestamp', 'N/A')}")
        print(f"  Last sample timestamp: {history[-1].get('timestamp', 'N/A')}")
    
    print("\n✅ GPU monitoring test completed successfully!")
    return True


def test_wandb_integration():
    """Test WandB integration (optional)."""
    print("\n🔗 Testing WandB integration...")
    
    try:
        import wandb
        
        # Initialize a test run
        wandb.init(
            project="gpu_monitoring_test",
            name=f"test_run_{int(time.time())}",
            mode="offline"  # Use offline mode for testing
        )
        
        monitor = GPUResourceMonitor(
            monitor_interval=1.0,
            log_to_wandb=True,
            log_to_console=False
        )
        
        monitor.start_monitoring()
        time.sleep(3)
        monitor.stop_monitoring()
        
        wandb.finish()
        print("✅ WandB integration test completed")
        return True
        
    except ImportError:
        print("⚠️  WandB not available, skipping integration test")
        return True
    except Exception as e:
        print(f"❌ WandB integration test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test GPU monitoring functionality")
    parser.add_argument("--test-wandb", action="store_true",
                       help="Test WandB integration")
    parser.add_argument("--duration", type=int, default=10,
                       help="Duration for GPU load test (seconds)")
    
    args = parser.parse_args()
    
    print("🚀 Starting GPU monitoring tests...\n")
    
    # Test basic monitoring
    success = test_monitoring()
    
    # Test WandB integration if requested
    if args.test_wandb and success:
        success = test_wandb_integration()
    
    if success:
        print("\n🎉 All tests passed!")
        print("\n📋 Next steps:")
        print("1. Run your GRPO training with +trainer.enable_gpu_monitoring=true")
        print("2. Check WandB dashboard for real-time GPU metrics")
        print("3. Use scripts/monitor_gpu_realtime.py for standalone monitoring")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
