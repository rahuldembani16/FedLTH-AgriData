import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
import random
from models.crop_yield_model import CropYieldDataPreprocessor

class CropYieldDataset(Dataset):
    """
    Custom Dataset for Crop Yield tabular data
    """
    def __init__(self, csv_path=None, dataframe=None, preprocessor=None, is_train=True):
        if csv_path is not None:
            self.df = pd.read_csv(csv_path)
        elif dataframe is not None:
            self.df = dataframe.copy()
        else:
            raise ValueError("Either csv_path or dataframe must be provided")
        
        self.preprocessor = preprocessor
        self.is_train = is_train
        
        if self.preprocessor is None:
            self.preprocessor = CropYieldDataPreprocessor()
            if self.is_train:
                self.preprocessor.fit(self.df)
        
        # Transform the data
        self.features, self.targets = self.preprocessor.transform(self.df)
        
        # Convert to tensors
        self.features = torch.FloatTensor(self.features)
        if self.targets is not None:
            self.targets = torch.FloatTensor(self.targets)
        
        # Store classes for compatibility (dummy classes for regression)
        self.classes = ['regression_task']  # Single class for regression
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx], torch.tensor(0.0)  # Dummy target
    
    def get_preprocessor(self):
        return self.preprocessor

class CropYieldPartitioner:
    """
    Custom partitioner for crop yield data supporting various federated learning scenarios
    """
    def __init__(self, targets, num_clients, partition="iid", alpha=0.5, min_samples_per_client=50, seed=42):
        self.targets = targets
        self.num_clients = num_clients
        self.partition = partition
        self.alpha = alpha  # Dirichlet concentration parameter
        self.min_samples_per_client = min_samples_per_client
        self.seed = seed
        
        np.random.seed(seed)
        random.seed(seed)
        
        self.client_dict = self._partition_data()
    
    def _partition_data(self) -> Dict[int, List[int]]:
        """Partition data based on the specified strategy"""
        n_data = len(self.targets)
        indices = np.arange(n_data)
        
        if self.partition == "iid":
            return self._iid_partition(indices)
        elif self.partition == "dirichlet":
            return self._dirichlet_partition(indices)
        elif self.partition == "geographic":
            return self._geographic_partition(indices)
        elif self.partition == "crop_based":
            return self._crop_based_partition(indices)
        else:
            raise ValueError(f"Partition {self.partition} not implemented")
    
    def _iid_partition(self, indices: np.ndarray) -> Dict[int, List[int]]:
        """IID partition - randomly distribute data"""
        np.random.shuffle(indices)
        client_dict = {}
        
        samples_per_client = len(indices) // self.num_clients
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            if i == self.num_clients - 1:  # Last client gets remaining data
                end_idx = len(indices)
            else:
                end_idx = (i + 1) * samples_per_client
            
            client_dict[i] = indices[start_idx:end_idx].tolist()
        
        return client_dict
    
    def _dirichlet_partition(self, indices: np.ndarray) -> Dict[int, List[int]]:
        """
        Dirichlet partition based on yield ranges (simulating non-IID distribution)
        """
        # Create yield-based bins for non-IID distribution
        targets_array = np.array(self.targets)
        n_bins = min(10, self.num_clients)  # Create bins based on yield ranges
        
        # Sort indices by target values and create bins
        sorted_indices = indices[np.argsort(targets_array)]
        bin_size = len(sorted_indices) // n_bins
        
        bins = []
        for i in range(n_bins):
            start_idx = i * bin_size
            if i == n_bins - 1:
                end_idx = len(sorted_indices)
            else:
                end_idx = (i + 1) * bin_size
            bins.append(sorted_indices[start_idx:end_idx])
        
        # Use Dirichlet distribution to assign bin proportions to clients
        client_dict = {i: [] for i in range(self.num_clients)}
        
        for bin_indices in bins:
            # Generate Dirichlet distribution
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # Assign samples to clients based on proportions
            np.random.shuffle(bin_indices)
            start_idx = 0
            
            for client_id in range(self.num_clients):
                if client_id == self.num_clients - 1:
                    # Last client gets remaining samples
                    client_samples = bin_indices[start_idx:]
                else:
                    n_samples = int(len(bin_indices) * proportions[client_id])
                    client_samples = bin_indices[start_idx:start_idx + n_samples]
                    start_idx += n_samples
                
                client_dict[client_id].extend(client_samples.tolist())
        
        # Ensure minimum samples per client
        self._ensure_min_samples(client_dict, indices)
        
        return client_dict
    
    def _geographic_partition(self, indices: np.ndarray) -> Dict[int, List[int]]:
        """
        Geographic partition - simulate regional federated learning
        This assumes we have region information in the original dataset
        """
        # This is a simplified version - in practice, you'd use actual geographic data
        np.random.shuffle(indices)
        return self._iid_partition(indices)  # Fallback to IID for now
    
    def _crop_based_partition(self, indices: np.ndarray) -> Dict[int, List[int]]:
        """
        Crop-based partition - clients specialize in different crops
        """
        # This would require crop type information from the original dataset
        # For now, fallback to Dirichlet
        return self._dirichlet_partition(indices)
    
    def _ensure_min_samples(self, client_dict: Dict[int, List[int]], all_indices: np.ndarray):
        """Ensure each client has minimum number of samples"""
        for client_id in range(self.num_clients):
            if len(client_dict[client_id]) < self.min_samples_per_client:
                # Need to redistribute samples
                shortage = self.min_samples_per_client - len(client_dict[client_id])
                
                # Find clients with excess samples
                for donor_id in range(self.num_clients):
                    if donor_id != client_id and len(client_dict[donor_id]) > self.min_samples_per_client + shortage:
                        # Transfer samples
                        samples_to_transfer = client_dict[donor_id][-shortage:]
                        client_dict[donor_id] = client_dict[donor_id][:-shortage]
                        client_dict[client_id].extend(samples_to_transfer)
                        break
    
    def __getitem__(self, index):
        """Get client data indices"""
        return self.client_dict[index]
    
    def __len__(self):
        """Get number of clients"""
        return self.num_clients

def create_crop_yield_datasets(csv_path, test_size=0.2, random_state=42):
    """
    Create train and test datasets for crop yield data
    """
    # Load full dataset
    df = pd.read_csv(csv_path)
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Create and fit preprocessor on training data
    preprocessor = CropYieldDataPreprocessor()
    preprocessor.fit(train_df)
    
    # Create datasets
    train_dataset = CropYieldDataset(dataframe=train_df, preprocessor=preprocessor, is_train=True)
    test_dataset = CropYieldDataset(dataframe=test_df, preprocessor=preprocessor, is_train=False)
    
    return train_dataset, test_dataset, preprocessor

if __name__ == '__main__':
    # Test the dataset and partitioner
    try:
        csv_path = 'dataset/crop_yield_10k_records.csv'
        train_dataset, test_dataset, preprocessor = create_crop_yield_datasets(csv_path)
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        print(f"Feature dimension: {train_dataset.features.shape[1]}")
        
        # Test partitioner
        partitioner = CropYieldPartitioner(
            targets=train_dataset.targets,
            num_clients=5,
            partition="dirichlet",
            alpha=0.5
        )
        
        print(f"Partitioner created for {len(partitioner)} clients")
        for i in range(len(partitioner)):
            client_data = partitioner[i]
            print(f"Client {i}: {len(client_data)} samples")
        
        # Test data loading
        sample_features, sample_target = train_dataset[0]
        print(f"Sample features shape: {sample_features.shape}")
        print(f"Sample target: {sample_target}")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        print("Make sure the dataset file exists in the correct location")
