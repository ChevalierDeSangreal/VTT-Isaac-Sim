import torch
import torch.nn as nn
import torch

class TrackAgileModuleVer0Dicision(nn.Module):
    def __init__(self, input_size=9+9, hidden_size=256, output_size=4, num_layers=2, device='cpu'):
        super(TrackAgileModuleVer0Dicision, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        torch.nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size).to(x[0].device)
        # print(x.shape, h0.shape)
        out, _ = self.gru(x, h0)
        # print(out.shape)
        out = self.fc(out[-1, :, :])
        # print(out.shape)
        out = torch.sigmoid(out) * 2 - 1
        return out
    
class TrackAgileModuleVer0ExtractorVer0(nn.Module):
    def __init__(self, device):
        super(TrackAgileModuleVer0ExtractorVer0, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=11, stride=11, padding=2)

        self.fc = nn.Sequential(
            nn.Linear(20 * 20, 80),
            nn.ReLU(),
            nn.Linear(80, 9)
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # For ReLU activations
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero (optional)
        self.device = device

    def forward(self, x, mask):
        x = torch.where(mask, x, torch.full_like(x, 333))
        x = -self.maxpooling(-x)
        x[x == 333] = 0
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

class DirectionPrediction(nn.Module):
    def __init__(self, device):
        super(DirectionPrediction, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # For ReLU activations
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero (optional)
        self.device = device
    def forward(self, x):
        x = self.fc(x)
        return x


class TrackAgileModuleVer0(nn.Module):
    """
    No velocity to output action
    """
    def __init__(self, device='cpu'):
        super(TrackAgileModuleVer0, self).__init__()
        self.device = device

        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer0Dicision(input_size=6+3,device=device).to(device)

        # Initialize Extractor module
        self.extractor_module = TrackAgileModuleVer0ExtractorVer0(device=device).to(device)

        self.directpred = DirectionPrediction(device=device).to(device)

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()

class TrackAgileModuleVer1(nn.Module):
    """
    Use velocity to output action
    """
    def __init__(self, device='cpu'):
        super(TrackAgileModuleVer1, self).__init__()
        self.device = device

        # Initialize Decision module
        self.decision_module = TrackAgileModuleVer0Dicision(input_size=9+3,device=device).to(device)

        # Initialize Extractor module
        self.extractor_module = TrackAgileModuleVer0ExtractorVer0(device=device).to(device)

        self.directpred = DirectionPrediction(device=device).to(device)

    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.eval()
