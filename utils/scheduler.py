# training/scheduler.py
import torch
import numpy as np

class TeacherForcingScheduler:
    def __init__(self, initial_ratio, final_ratio, decay_epochs, decay_type='linear', total_epochs=None):
        """
        Manages the teacher forcing ratio over epochs.

        Args:
            initial_ratio (float): Starting teacher forcing ratio (e.g., 1.0).
            final_ratio (float): Target teacher forcing ratio after decay (e.g., 0.0).
            decay_epochs (int): Number of epochs over which to decay the ratio.
            decay_type (str): 'linear' or 'exponential'.
            total_epochs (int, optional): Total number of training epochs. Used by some decay types
                                          or for validation. If None, decay_epochs is the primary control.
        """
        self.initial_ratio = initial_ratio
        self.current_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.decay_epochs = decay_epochs
        self.decay_type = decay_type.lower()
        self.total_epochs = total_epochs if total_epochs else decay_epochs

        if self.decay_type not in ['linear', 'exponential']:
            raise ValueError("decay_type must be 'linear' or 'exponential'")

    def step(self, current_epoch):
        """
        Update the teacher forcing ratio based on the current epoch.

        Args:
            current_epoch (int): The current training epoch (0-indexed).
        """
        if current_epoch >= self.decay_epochs:
            self.current_ratio = self.final_ratio
        else:
            if self.decay_type == 'linear':
                self.current_ratio = self.initial_ratio - \
                                     (self.initial_ratio - self.final_ratio) * (current_epoch / self.decay_epochs)
            elif self.decay_type == 'exponential':
                # Decay such that ratio = final_ratio at decay_epochs
                # initial_ratio * (decay_rate ^ epoch)
                # final_ratio = initial_ratio * (decay_rate ^ decay_epochs)
                # decay_rate = (final_ratio / initial_ratio) ^ (1 / decay_epochs)
                if self.initial_ratio == 0: # Avoid division by zero if initial is 0
                     self.current_ratio = self.final_ratio
                elif self.final_ratio == self.initial_ratio: # No decay needed
                    self.current_ratio = self.initial_ratio
                else:
                    # Ensure final_ratio / initial_ratio is positive for root calculation
                    # This might happen if final_ratio is 0. Add a small epsilon.
                    ratio_div = (self.final_ratio + 1e-9) / (self.initial_ratio + 1e-9)
                    if ratio_div <=0: # Should not happen with positive ratios or epsilon
                        decay_rate = 0 
                    else:
                        decay_rate = ratio_div ** (1.0 / self.decay_epochs)
                    self.current_ratio = self.initial_ratio * (decay_rate ** current_epoch)
            
            # Ensure ratio is within bounds
            self.current_ratio = max(self.final_ratio, min(self.initial_ratio, self.current_ratio))
            
        return self.current_ratio

    def get_ratio(self):
        return self.current_ratio

# Example of how to use PyTorch's built-in LR schedulers
# You would typically instantiate these in your train.py script.

def get_learning_rate_scheduler(optimizer, scheduler_type='steplr', step_size=10, gamma=0.1, patience=5, min_lr=1e-6):
    """
    Factory function for PyTorch learning rate schedulers.

    Args:
        optimizer: The PyTorch optimizer.
        scheduler_type (str): 'steplr', 'reduce_on_plateau'.
        step_size (int): For StepLR, period of learning rate decay.
        gamma (float): For StepLR, multiplicative factor of learning rate decay.
        patience (int): For ReduceLROnPlateau, number of epochs with no improvement after which learning rate will be reduced.
        min_lr (float): For ReduceLROnPlateau, a lower bound on the learning rate.

    Returns:
        A PyTorch learning rate scheduler object.
    """
    scheduler_type = scheduler_type.lower()
    if scheduler_type == 'steplr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=patience, min_lr=min_lr)
    # Add more schedulers as needed (e.g., CosineAnnealingLR)
    else:
        print(f"Unsupported scheduler_type: {scheduler_type}. No LR scheduler will be used.")
        return None

if __name__ == '__main__':
    # Test TeacherForcingScheduler
    print("--- Testing TeacherForcingScheduler ---")
    epochs = 20
    decay_epochs_tf = 15
    
    print("\nLinear Decay:")
    tf_scheduler_linear = TeacherForcingScheduler(initial_ratio=1.0, final_ratio=0.1, decay_epochs=decay_epochs_tf, decay_type='linear')
    for epoch in range(epochs):
        ratio = tf_scheduler_linear.step(epoch)
        print(f"Epoch {epoch+1}/{epochs}: Teacher Forcing Ratio = {ratio:.4f}")

    print("\nExponential Decay:")
    tf_scheduler_exp = TeacherForcingScheduler(initial_ratio=1.0, final_ratio=0.01, decay_epochs=decay_epochs_tf, decay_type='exponential')
    for epoch in range(epochs):
        ratio = tf_scheduler_exp.step(epoch)
        print(f"Epoch {epoch+1}/{epochs}: Teacher Forcing Ratio = {ratio:.4f}")
    
    # Test Learning Rate Scheduler Factory
    print("\n--- Testing Learning Rate Scheduler Factory ---")
    dummy_model = torch.nn.Linear(10,1)
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)

    lr_scheduler_step = get_learning_rate_scheduler(dummy_optimizer, scheduler_type='steplr', step_size=5, gamma=0.5)
    print(f"StepLR Scheduler: {lr_scheduler_step}")
    for epoch in range(15):
        # In a real loop, optimizer.step() would be called before scheduler.step()
        # For ReduceLROnPlateau, scheduler.step(validation_loss) is called.
        if isinstance(lr_scheduler_step, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler_step.step(np.random.rand()) # Dummy validation loss
        else:
            lr_scheduler_step.step()
        print(f"Epoch {epoch+1}, LR: {dummy_optimizer.param_groups[0]['lr']:.6f}")

    # Reset optimizer for ReduceLROnPlateau
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)
    lr_scheduler_plateau = get_learning_rate_scheduler(dummy_optimizer, scheduler_type='reduce_on_plateau', gamma=0.5, patience=2)
    print(f"\nReduceLROnPlateau Scheduler: {lr_scheduler_plateau}")
    dummy_val_losses = [1.0, 0.9, 0.95, 0.96, 0.8, 0.82, 0.83, 0.7, 0.6]
    for epoch, val_loss in enumerate(dummy_val_losses):
        lr_scheduler_plateau.step(val_loss)
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.2f}, LR: {dummy_optimizer.param_groups[0]['lr']:.6f}")