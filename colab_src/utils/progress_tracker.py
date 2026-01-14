"""Real-time training progress tracking and visualization."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class TrainingProgressTracker:
    """Track training progress with batch/epoch metrics and ETA estimation."""
    
    def __init__(self, output_dir=".", name="training_progress"):
        """
        Initialize progress tracker.
        
        Args:
            output_dir: Directory to save training history JSON
            name: Experiment name for logging
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.output_dir / "training_history.json"
        self.name = name
        
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "epoch_time": [],
            "best_epoch": 0,
            "best_val_loss": float('inf'),
        }
        
        self.batch_losses = []
        self.epoch_start_time = None
        self.training_start_time = None
        self.total_batches_per_epoch = 0
        self.current_batch = 0
        
    def start(self):
        """Initialize training timer."""
        self.training_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"🚀 Starting {self.name}")
        print(f"{'='*70}\n")
        
    def epoch_start(self, total_batches):
        """Start epoch timer.
        
        Args:
            total_batches: Total batches in this epoch
        """
        self.epoch_start_time = time.time()
        self.batch_losses = []
        self.current_batch = 0
        self.total_batches_per_epoch = total_batches
        
    def log_batch(self, loss, batch_idx=None, log_every=50):
        """Log batch loss.
        
        Args:
            loss: Loss value for this batch
            batch_idx: Batch index (auto-increment if None)
            log_every: Print progress every N batches
        """
        if batch_idx is not None:
            self.current_batch = batch_idx
        else:
            self.current_batch += 1
            
        self.batch_losses.append(loss)
        
        if (self.current_batch + 1) % log_every == 0:
            avg_loss = np.mean(self.batch_losses[-log_every:])
            elapsed = time.time() - self.epoch_start_time
            time_per_batch = elapsed / (self.current_batch + 1)
            eta_sec = time_per_batch * (self.total_batches_per_epoch - self.current_batch - 1)
            
            print(f"Batch {self.current_batch+1:5d}/{self.total_batches_per_epoch:5d} "
                  f"| Loss: {avg_loss:.4f} | ETA: {int(eta_sec)}s")
    
    def epoch_end(self, epoch, train_loss, val_loss=None):
        """Log epoch completion.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss for epoch
            val_loss: Validation loss for epoch (optional)
        """
        epoch_time = time.time() - self.epoch_start_time
        
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["epoch_time"].append(epoch_time)
        
        if val_loss is not None:
            self.history["val_loss"].append(val_loss)
            is_best = val_loss < self.history["best_val_loss"]
            if is_best:
                self.history["best_val_loss"] = val_loss
                self.history["best_epoch"] = epoch
        else:
            is_best = train_loss < self.history["best_val_loss"]
            if is_best:
                self.history["best_val_loss"] = train_loss
                self.history["best_epoch"] = epoch
        
        # ETA calculation
        total_elapsed = time.time() - self.training_start_time
        avg_time_per_epoch = total_elapsed / (epoch + 1)
        
        indicator = " 🌟" if is_best else ""
        print(f"\nEpoch {epoch+1:3d} | Train: {train_loss:.4f}", end="")
        if val_loss is not None:
            print(f" | Val: {val_loss:.4f}{indicator}", end="")
        print(f" | Time: {epoch_time:.1f}s | ETA: {int(avg_time_per_epoch * (50 - epoch - 1))}s")
        
        # Save history
        self._save_history()
    
    def summary(self, total_epochs=None):
        """Print final training summary.
        
        Args:
            total_epochs: Total epochs trained (for calculating statistics)
        """
        if self.training_start_time is None:
            return
            
        total_time = time.time() - self.training_start_time
        
        print(f"\n{'='*70}")
        print(f"✅ Training Complete: {self.name}")
        print(f"{'='*70}")
        print(f"Total time: {int(total_time)}s ({total_time/60:.1f} minutes)")
        
        if self.history["train_loss"]:
            min_train = min(self.history["train_loss"])
            max_train = max(self.history["train_loss"])
            final_train = self.history["train_loss"][-1]
            improvement = (max_train - min_train) / max_train * 100
            
            print(f"\nTrain Loss:")
            print(f"  Min:  {min_train:.4f}")
            print(f"  Max:  {max_train:.4f}")
            print(f"  Final: {final_train:.4f}")
            print(f"  Improvement: {improvement:.1f}%")
        
        if self.history["val_loss"]:
            min_val = min(self.history["val_loss"])
            max_val = max(self.history["val_loss"])
            final_val = self.history["val_loss"][-1]
            
            print(f"\nValidation Loss:")
            print(f"  Min:  {min_val:.4f}")
            print(f"  Max:  {max_val:.4f}")
            print(f"  Final: {final_val:.4f}")
            print(f"  Best epoch: {self.history['best_epoch']}")
        
        print(f"{'='*70}\n")
    
    def _save_history(self):
        """Save history to JSON file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)


def monitor_training_live(history_file="training_history.json", update_interval=5):
    """
    Monitor training progress in real-time by reading history file.
    Auto-updates every N seconds. Press Ctrl+C to stop.
    
    Args:
        history_file: Path to training history JSON
        update_interval: Seconds between updates
    """
    history_path = Path(history_file)
    
    if not history_path.exists():
        print(f"❌ History file not found: {history_file}")
        return
    
    print(f"\n📊 Monitoring training (updating every {update_interval}s)...")
    print("Press Ctrl+C to stop monitoring\n")
    
    prev_epochs = 0
    
    try:
        while True:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            current_epochs = len(history.get("epoch", []))
            
            if current_epochs > prev_epochs or prev_epochs == 0:
                # Progress bar
                if history.get("train_loss"):
                    epochs = history["epoch"]
                    total_epochs = 50  # Assume 50 total epochs
                    progress = min(len(epochs) / total_epochs, 1.0)
                    bar_length = 40
                    filled = int(bar_length * progress)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    print(f"Progress: [{bar}] {len(epochs)}/50 epochs")
                    
                    # Show recent 5 epochs
                    print(f"\nRecent epochs:")
                    for i, epoch in enumerate(epochs[-5:]):
                        train = history["train_loss"][i - 5] if i - 5 >= 0 else history["train_loss"][0]
                        is_best = epoch == history.get("best_epoch")
                        indicator = " 🌟" if is_best else ""
                        print(f"  Epoch {epoch+1}: Loss={train:.4f}{indicator}")
                    
                    # Plot loss curves
                    plt.figure(figsize=(12, 4))
                    
                    plt.subplot(1, 2, 1)
                    plt.plot(history["epoch"], history["train_loss"], 'b-', label='Train Loss', linewidth=2)
                    if history.get("val_loss"):
                        plt.plot(history["epoch"], history["val_loss"], 'r-', label='Val Loss', linewidth=2)
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Training Progress')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Improvement graph
                    plt.subplot(1, 2, 2)
                    min_loss = min(history["train_loss"])
                    improvement = [(max(history["train_loss"]) - l) / max(history["train_loss"]) * 100 
                                  for l in history["train_loss"]]
                    plt.plot(history["epoch"], improvement, 'g-', linewidth=2)
                    plt.xlabel('Epoch')
                    plt.ylabel('Improvement (%)')
                    plt.title('Loss Improvement Over Baseline')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.show()
                    
                prev_epochs = current_epochs
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped")
