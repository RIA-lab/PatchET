import torch
import torch.nn as nn


class TemperatureRangeLoss(nn.Module):
    def __init__(self, lambda_penalty=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_penalty = lambda_penalty

    def forward(self, pred, label):
        pred_low, pred_high = pred[:, 0], pred[:, 1]
        label_low, label_high = label[:, 0], label[:, 1]
        mse_loss = self.mse(pred_low, label_low) + self.mse(pred_high, label_high)
        constraint_penalty = torch.relu(pred_low - pred_high).mean()
        loss = mse_loss + self.lambda_penalty * constraint_penalty
        return loss


class RMSELoss:
    def __init__(self):
        self.mse = nn.MSELoss()

    def __call__(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


class WeightedRMSELoss:
    def __init__(self, temp_ranges=None, weights=None):
        """
        temp_ranges: Optional list of tuples defining temperature ranges, e.g., [(None, 25), (25, 50), (50, 80), (80, None)].
                     If None, defaults to unweighted RMSE.
        weights: Optional list of weights corresponding to each range, e.g., [w1, w2, w3, w4].
                 If None, defaults to unweighted RMSE.
        """
        self.mse = nn.MSELoss(reduction='none')  # No reduction for weighted case
        self.is_weighted = temp_ranges is not None and weights is not None

        if self.is_weighted:
            assert len(temp_ranges) == len(weights), "Number of ranges must match number of weights"
            self.temp_ranges = temp_ranges
            self.weights = weights
        else:
            # For unweighted RMSE, we can use reduction='mean' directly later
            self.mse = nn.MSELoss(reduction='mean')

    def set_ranges_and_weights(self, temp_ranges, weights):
        self.is_weighted = temp_ranges is not None and weights is not None
        if self.is_weighted:
            assert len(temp_ranges) == len(weights), "Number of ranges must match number of weights"
            self.temp_ranges = temp_ranges
            self.weights = weights
            self.mse = nn.MSELoss(reduction='none')

    def __call__(self, y_pred, y_true):
        if self.is_weighted:
            # Weighted RMSE logic
            squared_error = self.mse(y_pred, y_true)
            weights = torch.zeros_like(y_true, dtype=torch.float)

            # Assign weights based on true temperature values
            for (temp_min, temp_max), weight in zip(self.temp_ranges, self.weights):
                if temp_min is None:
                    mask = (y_true < temp_max)
                elif temp_max is None:
                    mask = (y_true >= temp_min)
                else:
                    mask = (y_true >= temp_min) & (y_true < temp_max)
                weights[mask] = weight

            # Apply weights and compute RMSE
            weighted_squared_error = weights * squared_error
            return torch.sqrt(torch.mean(weighted_squared_error))
        else:
            # Unweighted RMSE logic
            mse = self.mse(y_pred, y_true)
            return torch.sqrt(mse)