import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import math
import numpy as np
from copy import deepcopy

class TabularFedLTHPruner:
    """
    Federated Lottery Ticket Hypothesis Pruner for Tabular/Dense Networks
    Adapted for MLP architectures used in crop yield prediction
    """
    
    def __init__(self, start_ratio=0.2, end_ratio=0.8, device='cpu', 
                 min_increase=0.05, structured_pruning=True):
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.ratio = start_ratio
        self.device = device
        self.min_increase = min_increase
        self.unpruned_flag = True
        self.structured_pruning = structured_pruning
        
    def unstructured_prune(self, model, skip_first_layer=True, skip_last_layer=True):
        """
        Apply unstructured pruning to Linear layers in MLP
        """
        parameters_to_prune = []
        layer_names = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Skip first layer (input) and last layer (output) if specified
                if skip_first_layer and 'network.0' in name:
                    continue
                if skip_last_layer and name == 'network.' + str(len(list(model.named_modules())) - 2):
                    continue
                    
                parameters_to_prune.append((module, 'weight'))
                layer_names.append(name)
        
        if parameters_to_prune:
            parameters_to_prune = tuple(parameters_to_prune)
            
            # Apply L1 unstructured pruning globally
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.ratio,
            )
            
            print(f"Applied unstructured pruning with ratio {self.ratio:.3f} to {len(parameters_to_prune)} layers")
            
            # Update ratio for next iteration
            if self.ratio < self.end_ratio:
                self.ratio = min(self.ratio + self.min_increase, self.end_ratio)
    
    def structured_prune(self, model, weight_with_mask, train_loader, criterion, output_dim=1):
        """
        Apply structured pruning to create a smaller network
        For regression, we maintain the output dimension as 1
        """
        if not self.structured_pruning:
            return model, 0.0
        
        # Calculate current sparsity
        current_sparsity = self.check_sparsity(model)
        
        # Only do structured pruning if sparsity is high enough
        if current_sparsity < 50:  # Less than 50% remaining weights
            print("Skipping structured pruning - not enough sparsity yet")
            return model, current_sparsity/100
        
        # Create a new smaller model
        try:
            new_model = self._create_pruned_model(model, sparsity_target=current_sparsity/100)
            
            # Transfer important weights
            self._transfer_weights(model, new_model, weight_with_mask)
            
            # Move to correct device
            if next(model.parameters()).is_cuda:
                new_model = new_model.cuda(next(model.parameters()).device)
            
            return new_model, current_sparsity/100
            
        except Exception as e:
            print(f"Structured pruning failed: {e}")
            print("Continuing with unstructured pruning only")
            return model, current_sparsity/100
    
    def _create_pruned_model(self, original_model, sparsity_target=0.5):
        """
        Create a structurally smaller model based on sparsity
        """
        from models.crop_yield_model import CropYieldMLP
        
        # Get original architecture more robustly
        original_dims = []
        linear_layers = [(name, module) for name, module in original_model.named_modules() 
                        if isinstance(module, torch.nn.Linear)]
        
        for name, layer in linear_layers[:-1]:  # Exclude output layer
            original_dims.append(layer.out_features)
        
        if not original_dims:
            raise ValueError("No linear layers found in model")
        
        # Calculate new dimensions based on sparsity - be more conservative
        reduction_factor = max(0.5, 1 - sparsity_target * 0.5)  # Less aggressive reduction
        new_dims = [max(8, int(dim * reduction_factor)) for dim in original_dims]  # Minimum 8 neurons
        
        # Get input dimension
        if hasattr(original_model, 'input_dim'):
            input_dim = original_model.input_dim
        else:
            input_dim = linear_layers[0][1].in_features if linear_layers else 12  # Default
        
        print(f"Creating pruned model: {original_dims} -> {new_dims}")
        
        # Create new model with reduced dimensions
        new_model = CropYieldMLP(
            input_dim=input_dim,
            hidden_dims=new_dims,
            dropout_rate=0.2
        )
        
        return new_model
    
    def _transfer_weights(self, old_model, new_model, weight_with_mask):
        """
        Transfer important weights from old model to new model
        """
        old_state = old_model.state_dict()
        new_state = new_model.state_dict()
        
        # Transfer weights layer by layer
        old_linear_layers = [(name, module) for name, module in old_model.named_modules() 
                           if isinstance(module, nn.Linear)]
        new_linear_layers = [(name, module) for name, module in new_model.named_modules() 
                           if isinstance(module, nn.Linear)]
        
        for (old_name, old_layer), (new_name, new_layer) in zip(old_linear_layers, new_linear_layers):
            old_weight_key = old_name + '.weight'
            new_weight_key = new_name + '.weight'
            old_bias_key = old_name + '.bias'
            new_bias_key = new_name + '.bias'
            
            if old_weight_key in old_state and new_weight_key in new_state:
                old_weight = old_state[old_weight_key]
                new_weight_shape = new_state[new_weight_key].shape
                
                # Transfer top-k important weights based on magnitude
                if old_weight.shape[0] >= new_weight_shape[0] and old_weight.shape[1] >= new_weight_shape[1]:
                    # Calculate importance scores (L1 norm of weights)
                    importance = torch.norm(old_weight, p=1, dim=1)
                    _, top_indices = torch.topk(importance, new_weight_shape[0])
                    
                    # Transfer weights
                    new_state[new_weight_key] = old_weight[top_indices, :new_weight_shape[1]]
                    
                    # Transfer corresponding biases
                    if old_bias_key in old_state and new_bias_key in new_state:
                        new_state[new_bias_key] = old_state[old_bias_key][top_indices]
        
        new_model.load_state_dict(new_state)
    
    def check_sparsity(self, model, skip_first_layer=True, skip_last_layer=True):
        """
        Check sparsity of Linear layers in the model
        """
        total_params = 0
        zero_params = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Skip first and last layers if specified
                if skip_first_layer and 'network.0' in name:
                    continue
                if skip_last_layer and 'network.' + str(len(list(model.named_modules())) - 2) in name:
                    continue
                
                # Check for pruned weights (with mask) or regular weights
                if hasattr(module, 'weight_mask'):
                    weight = module.weight_orig * module.weight_mask
                else:
                    weight = module.weight
                
                total_params += weight.numel()
                zero_params += float(torch.sum(weight == 0))
        
        if total_params > 0:
            sparsity_percentage = 100 * zero_params / total_params
            remaining_percentage = 100 - sparsity_percentage
            print(f'* Remaining weights = {remaining_percentage:.2f}% (Sparsity: {sparsity_percentage:.2f}%)')
            return remaining_percentage
        else:
            return 100.0
    
    def remove_prune(self, model, skip_first_layer=True, skip_last_layer=True):
        """
        Remove pruning masks and make pruning permanent
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Skip first and last layers if specified
                if skip_first_layer and 'network.0' in name:
                    continue
                if skip_last_layer and 'network.' + str(len(list(model.named_modules())) - 2) in name:
                    continue
                
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
                if hasattr(module, 'bias_mask'):
                    prune.remove(module, 'bias')

def pruning_model_dense(model, px, skip_first_layer=True, skip_last_layer=True):
    """
    Apply unstructured pruning to dense/linear layers
    """
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip first layer (input features) and last layer (output) if specified
            if skip_first_layer and 'network.0' in name:
                continue
            if skip_last_layer and 'network.' + str(len(list(model.named_modules())) - 2) in name:
                continue
                
            parameters_to_prune.append((module, 'weight'))
    
    if parameters_to_prune:
        parameters_to_prune = tuple(parameters_to_prune)
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=px,
        )

def check_sparsity_dense(model, skip_first_layer=True, skip_last_layer=True):
    """
    Check sparsity for dense/linear layers
    """
    total_params = 0
    zero_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if skip_first_layer and 'network.0' in name:
                continue
            if skip_last_layer and 'network.' + str(len(list(model.named_modules())) - 2) in name:
                continue
            
            if hasattr(module, 'weight_mask'):
                weight = module.weight_orig * module.weight_mask
            else:
                weight = module.weight
                
            total_params += weight.numel()
            zero_params += float(torch.sum(weight == 0))
    
    if total_params > 0:
        remaining_percentage = 100 * (1 - zero_params / total_params)
        print(f'* Remaining weight = {remaining_percentage:.2f}%')
        return remaining_percentage
    else:
        return 100.0

# Legacy functions for compatibility
def pruning_model(model, px, conv1=False):
    """Legacy function - redirects to dense pruning"""
    return pruning_model_dense(model, px, skip_first_layer=not conv1)

def check_sparsity(model, conv1=True):
    """Legacy function - redirects to dense sparsity check"""
    return check_sparsity_dense(model, skip_first_layer=conv1)

if __name__ == '__main__':
    # Test the pruning utilities
    from models.crop_yield_model import CropYieldMLP
    
    model = CropYieldMLP(input_dim=21, hidden_dims=[128, 64, 32])
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test pruning
    pruner = TabularFedLTHPruner(start_ratio=0.2, end_ratio=0.8)
    initial_sparsity = pruner.check_sparsity(model)
    
    pruner.unstructured_prune(model)
    final_sparsity = pruner.check_sparsity(model)
    
    print(f"Sparsity increased from {100-initial_sparsity:.2f}% to {100-final_sparsity:.2f}%")
