from torch.utils.data import Dataset
import pandas as pd


class DatasetOpt(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = pd.read_csv(data)
        self.data.rename(columns={'temperature_optimum': 'label'}, inplace=True)
        self.default_ranges = [(None, 25), (25, 50), (50, 80), (80, None)]

    def calculate_weights(self):
        """Calculate weights as inverse frequency of samples in each temp range."""
        total_samples = len(self.data)
        weights = []

        for temp_min, temp_max in self.default_ranges:
            # Create mask for the range
            if temp_min is None:
                mask = self.data['label'] < temp_max
            elif temp_max is None:
                mask = self.data['label'] >= temp_min
            else:
                mask = (self.data['label'] >= temp_min) & (self.data['label'] < temp_max)

            # Count samples in this range
            range_count = mask.sum()
            if range_count == 0:
                weights.append(0.0)  # Avoid division by zero; no weight if empty
            else:
                # Inverse frequency weight (scaled by total samples)
                weight = total_samples / range_count
                weights.append(weight)

        # weights[2] = weights[2] * 4
        # weights[3] = weights[3] * 8

        # Optional: Normalize weights to sum to 1
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]

        return weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.loc[idx, ['accession', 'ec', 'label', 'organism', 'sequence', 'ogt']]
        # return self.data.loc[idx, ['accession', 'ec', 'organism', 'sequence']]


class DatasetStability(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = pd.read_csv(data)
        self.data.rename(columns={'stability': 'label'}, inplace=True)
        self.default_ranges = [(None, 45), (45, 70), (70, 100), (100, None)]

    def calculate_weights(self):
        total_samples = len(self.data)
        weights = []
        for temp_min, temp_max in self.default_ranges:
            if temp_min is None:
                mask = self.data['label'] < temp_max
            elif temp_max is None:
                mask = self.data['label'] >= temp_min
            else:
                mask = (self.data['label'] >= temp_min) & (self.data['label'] < temp_max)

            range_count = mask.sum()
            if range_count == 0:
                weights.append(0.0)
            else:
                weight = total_samples / range_count
                weights.append(weight)

        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]

        return weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.loc[idx, ['accession', 'ec', 'label', 'organism', 'sequence', 'ogt']]


class DatasetRange(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = pd.read_csv(data)
        self.data.rename(columns={'temperature_low': 'labels_low', 'temperature_high': 'labels_high'}, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.loc[idx, ['accession', 'ec', 'labels_low', 'labels_high', 'organism', 'sequence']]


def generate_mutated_seq(seq):
    amino_acid = 'ACDEFGHIKLMNPQRSTVWY'
    mutations = []
    mutated_seq = []
    for i, aa in enumerate(seq):
        for aa_ in amino_acid:
            if aa_ != aa:
                mutations.append(f'{aa}{i+1}{aa_}')
            else:
                mutations.append('-')
            mutated_seq.append(seq[:i] + aa_ + seq[i+1:])
    return pd.DataFrame({'mutation': mutations, 'sequence': mutated_seq})


#inference only
class DatasetMutation(Dataset):
    def __init__(self, seq):
        self.data = generate_mutated_seq(seq)
        self.data['label'] = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.loc[idx, ['mutation', 'sequence', 'label']]


def load_dataset(name):
    if name == 'opt':
        return DatasetOpt
    elif name == 'stability':
        return DatasetStability
    elif name == 'range':
        return DatasetRange
    elif name == 'mutation':
        return DatasetMutation
    else:
        raise ValueError('Invalid dataset name')