#!/usr/bin/env python3
"""
Standalone GPU monitoring script for real-time visualization.
Can be run independently or alongside VERL training.
"""

import argparse
import time
import signal
import sys
from typing import Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from verl.utils.gpu_monitor import GPUResourceMonitor


class StandaloneGPUMonitor:
    """Standalone GPU monitor with optional wandb integration."""
    
    def __init__(self, 
                 monitor_interval: float = 1.0,
                 wandb_project: Optional[str] = None,
                 wandb_run_name: Optional[str] = None,
                 console_output: bool = True,
                 device_ids: Optional[list] = None):
        
        self.monitor_interval = monitor_interval
        self.console_output = console_output
        self.running = False
        
        # Initialize wandb if requested
        self.use_wandb = False
        if wandb_project and WANDB_AVAILABLE:
            try:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name or f"gpu_monitor_{int(time.time())}",
                    tags=["gpu_monitoring", "standalone"]
                )
                self.use_wandb = True
                print(f"✓ WandB initialized: {wandb_project}/{wandb_run_name}")
            except Exception as e:
                print(f"✗ Failed to initialize WandB: {e}")
        
        # Initialize GPU monitor
        self.monitor = GPUResourceMonitor(
            monitor_interval=monitor_interval,
            log_to_wandb=self.use_wandb,
            log_to_console=console_output,
            device_ids=device_ids
        )
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n📊 Received signal {signum}, shutting down GPU monitor...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start monitoring."""
        print("🚀 Starting GPU monitoring...")
        print(f"📊 Monitoring interval: {self.monitor_interval}s")
        print(f"🖥️  Monitoring GPUs: {self.monitor.device_ids}")
        print(f"📈 WandB logging: {'✓' if self.use_wandb else '✗'}")
        print(f"🖨️  Console output: {'✓' if self.console_output else '✗'}")
        print("Press Ctrl+C to stop monitoring\n")
        
        self.running = True
        self.monitor.start_monitoring()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop monitoring."""
        if self.running:
            print("\n🛑 Stopping GPU monitoring...")
            self.running = False
            self.monitor.stop_monitoring()
            
            if self.use_wandb:
                try:
                    wandb.finish()
                    print("✓ WandB session finished")
                except:
                    pass
            
            print("✓ GPU monitoring stopped")
    
    def get_summary(self):
        """Get monitoring summary."""
        history = self.monitor.get_metrics_history()
        if not history:
            return "No metrics collected yet"
        
        latest = history[-1]
        summary = []
        summary.append(f"📊 Monitoring Summary (Total samples: {len(history)})")
        summary.append("=" * 50)
        
        for device_id in self.monitor.device_ids:
            gpu_util = latest.get(f'gpu_{device_id}_gpu_utilization_pct', 0)
            mem_util = latest.get(f'gpu_{device_id}_memory_utilization_pct', 0)
            mem_used = latest.get(f'gpu_{device_id}_memory_allocated_gb', 0)
            mem_total = latest.get(f'gpu_{device_id}_memory_total_gb', 0)
            temp = latest.get(f'gpu_{device_id}_temperature_c', 0)
            power = latest.get(f'gpu_{device_id}_power_usage_w', 0)
            
            summary.append(f"GPU {device_id}:")
            summary.append(f"  🔥 Utilization: {gpu_util:.1f}%")
            summary.append(f"  💾 Memory: {mem_used:.1f}/{mem_total:.1f} GB ({mem_util:.1f}%)")
            if temp > 0:
                summary.append(f"  🌡️  Temperature: {temp:.1f}°C")
            if power > 0:
                summary.append(f"  ⚡ Power: {power:.1f}W")
            summary.append("")
        
        return "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(description="Real-time GPU monitoring with optional WandB logging")
    parser.add_argument("--interval", type=float, default=1.0, 
                       help="Monitoring interval in seconds (default: 1.0)")
    parser.add_argument("--wandb-project", type=str, 
                       help="WandB project name for logging")
    parser.add_argument("--wandb-run-name", type=str,
                       help="WandB run name (default: auto-generated)")
    parser.add_argument("--no-console", action="store_true",
                       help="Disable console output")
    parser.add_argument("--gpus", type=str,
                       help="Comma-separated list of GPU IDs to monitor (default: all)")
    parser.add_argument("--summary-interval", type=int, default=30,
                       help="Print summary every N seconds (default: 30)")
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    device_ids = None
    if args.gpus:
        try:
            device_ids = [int(x.strip()) for x in args.gpus.split(',')]
        except ValueError:
            print("❌ Invalid GPU IDs format. Use comma-separated integers (e.g., '0,1,2')")
            sys.exit(1)
    
    # Check dependencies
    if args.wandb_project and not WANDB_AVAILABLE:
        print("❌ WandB not available. Install with: pip install wandb")
        sys.exit(1)
    
    # Create and start monitor
    monitor = StandaloneGPUMonitor(
        monitor_interval=args.interval,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        console_output=not args.no_console,
        device_ids=device_ids
    )
    
    # Start monitoring in a separate thread for summary printing
    import threading
    
    def print_summary():
        while monitor.running:
            time.sleep(args.summary_interval)
            if monitor.running:
                print(monitor.get_summary())
    
    if not args.no_console:
        summary_thread = threading.Thread(target=print_summary, daemon=True)
        summary_thread.start()
    
    # Start main monitoring
    monitor.start()


if __name__ == "__main__":
    main()
