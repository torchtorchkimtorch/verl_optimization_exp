"""
Real-time GPU resource monitoring utilities for VERL training.
"""

import logging
import threading
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from verl.utils.device import get_torch_device

logger = logging.getLogger(__name__)


class GPUResourceMonitor:
    """Real-time GPU resource monitoring class."""
    
    def __init__(self, 
                 monitor_interval: float = 1.0,
                 log_to_wandb: bool = True,
                 log_to_console: bool = False,
                 device_ids: Optional[List[int]] = None):
        """
        Initialize GPU resource monitor.
        
        Args:
            monitor_interval: Monitoring interval in seconds
            log_to_wandb: Whether to log metrics to wandb
            log_to_console: Whether to log metrics to console
            device_ids: List of GPU device IDs to monitor. If None, monitors all available GPUs
        """
        self.monitor_interval = monitor_interval
        self.log_to_wandb = log_to_wandb
        self.log_to_console = log_to_console
        self.device_ids = device_ids
        
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
        
        # Initialize NVML if available
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_available = True
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
                self.nvml_available = False
        else:
            self.nvml_available = False
            logger.warning("pynvml not available. Install with: pip install pynvml")
        
        # Get device information
        self._setup_devices()
    
    def _setup_devices(self):
        """Setup device information."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available")
            self.device_ids = []
            return
        
        if self.device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
        
        logger.info(f"Monitoring GPUs: {self.device_ids}")
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get current GPU metrics for all monitored devices."""
        metrics = {}
        
        for device_id in self.device_ids:
            device_metrics = self._get_single_gpu_metrics(device_id)
            for key, value in device_metrics.items():
                metrics[f"gpu_{device_id}_{key}"] = value
        
        return metrics
    
    def _get_single_gpu_metrics(self, device_id: int) -> Dict[str, float]:
        """Get metrics for a single GPU device."""
        metrics = {}
        
        try:
            # PyTorch memory metrics
            with torch.cuda.device(device_id):
                total_memory = torch.cuda.get_device_properties(device_id).total_memory
                allocated_memory = torch.cuda.memory_allocated(device_id)
                reserved_memory = torch.cuda.memory_reserved(device_id)
                
                metrics.update({
                    'memory_allocated_gb': allocated_memory / (1024**3),
                    'memory_reserved_gb': reserved_memory / (1024**3),
                    'memory_total_gb': total_memory / (1024**3),
                    'memory_utilization_pct': (allocated_memory / total_memory) * 100,
                })
        except Exception as e:
            logger.warning(f"Failed to get PyTorch metrics for GPU {device_id}: {e}")
        
        # NVML metrics (more detailed)
        if self.nvml_available:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics['gpu_utilization_pct'] = util.gpu
                metrics['memory_utilization_nvml_pct'] = util.memory
                
                # Memory info from NVML
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics['memory_used_nvml_gb'] = mem_info.used / (1024**3)
                metrics['memory_free_nvml_gb'] = mem_info.free / (1024**3)
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics['temperature_c'] = temp
                except:
                    pass
                
                # Power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    metrics['power_usage_w'] = power
                except:
                    pass
                
                # Clock speeds
                try:
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    metrics['graphics_clock_mhz'] = graphics_clock
                    metrics['memory_clock_mhz'] = memory_clock
                except:
                    pass
                    
            except Exception as e:
                logger.warning(f"Failed to get NVML metrics for GPU {device_id}: {e}")
        
        return metrics
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get system-level metrics."""
        metrics = {}
        
        if PSUTIL_AVAILABLE:
            try:
                # CPU usage
                metrics['cpu_utilization_pct'] = psutil.cpu_percent(interval=None)
                
                # Memory usage
                memory = psutil.virtual_memory()
                metrics['system_memory_used_gb'] = memory.used / (1024**3)
                metrics['system_memory_total_gb'] = memory.total / (1024**3)
                metrics['system_memory_utilization_pct'] = memory.percent
                
            except Exception as e:
                logger.warning(f"Failed to get system metrics: {e}")
        
        return metrics
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                timestamp = time.time()
                
                # Get GPU metrics
                gpu_metrics = self._get_gpu_metrics()
                
                # Get system metrics
                system_metrics = self._get_system_metrics()
                
                # Combine all metrics
                all_metrics = {
                    'timestamp': timestamp,
                    **gpu_metrics,
                    **system_metrics
                }
                
                # Store metrics
                self.metrics_history.append(all_metrics)
                
                # Log to console if enabled
                if self.log_to_console:
                    self._log_to_console(all_metrics)
                
                # Log to wandb if enabled and available
                if self.log_to_wandb:
                    self._log_to_wandb(all_metrics)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.monitor_interval)
    
    def _log_to_console(self, metrics: Dict[str, float]):
        """Log metrics to console."""
        gpu_summary = []
        for device_id in self.device_ids:
            gpu_util = metrics.get(f'gpu_{device_id}_gpu_utilization_pct', 0)
            mem_util = metrics.get(f'gpu_{device_id}_memory_utilization_pct', 0)
            gpu_summary.append(f"GPU{device_id}: {gpu_util:.1f}%/{mem_util:.1f}%")
        
        logger.info(f"GPU Monitor - {' | '.join(gpu_summary)}")
    
    def _log_to_wandb(self, metrics: Dict[str, float]):
        """Log metrics to wandb."""
        try:
            import wandb
            if wandb.run is not None:
                # Remove timestamp for wandb logging
                wandb_metrics = {k: v for k, v in metrics.items() if k != 'timestamp'}
                wandb.log(wandb_metrics)
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if not self.monitoring:
            logger.warning("Monitoring not started")
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("GPU monitoring stopped")
    
    def get_latest_metrics(self) -> Optional[Dict[str, float]]:
        """Get the latest metrics."""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]
    
    def get_metrics_history(self) -> List[Dict[str, float]]:
        """Get all metrics history."""
        return self.metrics_history.copy()


# Global monitor instance
_global_monitor: Optional[GPUResourceMonitor] = None


def start_gpu_monitoring(monitor_interval: float = 1.0,
                        log_to_wandb: bool = True,
                        log_to_console: bool = False,
                        device_ids: Optional[List[int]] = None) -> GPUResourceMonitor:
    """
    Start global GPU monitoring.
    
    Args:
        monitor_interval: Monitoring interval in seconds
        log_to_wandb: Whether to log metrics to wandb
        log_to_console: Whether to log metrics to console
        device_ids: List of GPU device IDs to monitor
    
    Returns:
        GPUResourceMonitor instance
    """
    global _global_monitor
    
    if _global_monitor is not None:
        logger.warning("Global GPU monitoring already started")
        return _global_monitor
    
    _global_monitor = GPUResourceMonitor(
        monitor_interval=monitor_interval,
        log_to_wandb=log_to_wandb,
        log_to_console=log_to_console,
        device_ids=device_ids
    )
    _global_monitor.start_monitoring()
    
    return _global_monitor


def stop_gpu_monitoring():
    """Stop global GPU monitoring."""
    global _global_monitor
    
    if _global_monitor is not None:
        _global_monitor.stop_monitoring()
        _global_monitor = None


def get_gpu_monitor() -> Optional[GPUResourceMonitor]:
    """Get the global GPU monitor instance."""
    return _global_monitor
