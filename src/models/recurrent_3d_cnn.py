"""
Recurrent 3D CNN with ConvLSTM for spatiotemporal tumor growth prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell for spatiotemporal feature learning.
    """
    
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Gates: input, forget, cell, output
        self.conv = nn.Conv3d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
    def forward(self, x, hidden_state):
        """
        Args:
            x: Input tensor (batch, channels, depth, height, width)
            hidden_state: Tuple of (h, c) hidden states
        
        Returns:
            h_next, c_next: Next hidden states
        """
        h, c = hidden_state
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Compute gates
        gates = self.conv(combined)
        
        # Split into individual gates
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        # Update cell state and hidden state
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, spatial_size):
        """Initialize hidden states."""
        depth, height, width = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_channels, depth, height, width).to(next(self.parameters()).device),
            torch.zeros(batch_size, self.hidden_channels, depth, height, width).to(next(self.parameters()).device)
        )


class Recurrent3DCNN(nn.Module):
    """
    Recurrent 3D CNN for tumor growth prediction on longitudinal scans.
    
    Uses ConvLSTM layers to capture spatiotemporal patterns in sequential 3D scans.
    """
    
    def __init__(self, in_channels=4, hidden_channels=[32, 64, 128], 
                 num_layers=3, output_channels=1, kernel_size=3):
        super(Recurrent3DCNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # Initial 3D convolution
        self.conv_in = nn.Conv3d(in_channels, hidden_channels[0], kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm3d(hidden_channels[0])
        
        # ConvLSTM layers
        self.convlstm_layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels[i] if i == 0 else hidden_channels[i-1]
            out_ch = hidden_channels[i]
            self.convlstm_layers.append(ConvLSTMCell(in_ch, out_ch, kernel_size))
        
        # Output convolution
        self.conv_out = nn.Conv3d(hidden_channels[-1], output_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, time_steps, channels, depth, height, width)
        
        Returns:
            output: Predicted tumor state (batch, output_channels, depth, height, width)
        """
        batch_size, time_steps, c, d, h, w = x.shape
        
        # Initialize hidden states for all layers
        hidden_states = []
        spatial_size = (d, h, w)
        for i in range(self.num_layers):
            hidden_states.append(
                self.convlstm_layers[i].init_hidden(batch_size, spatial_size)
            )
        
        # Process each time step
        for t in range(time_steps):
            input_t = x[:, t, :, :, :, :]
            
            # Initial convolution
            features = F.relu(self.bn_in(self.conv_in(input_t)))
            
            # Pass through ConvLSTM layers
            for i, convlstm in enumerate(self.convlstm_layers):
                h, c = hidden_states[i]
                h, c = convlstm(features, (h, c))
                hidden_states[i] = (h, c)
                features = h
        
        # Use final hidden state to generate prediction
        final_features = hidden_states[-1][0]
        output = self.conv_out(final_features)
        
        return output
    
    def predict_future(self, x, num_future_steps=1):
        """
        Predict tumor state for multiple future time steps.
        
        Args:
            x: Input tensor of shape (batch, time_steps, channels, depth, height, width)
            num_future_steps: Number of future time steps to predict
        
        Returns:
            predictions: List of predicted future states
        """
        batch_size, time_steps, c, d, h, w = x.shape
        
        # Initialize hidden states
        hidden_states = []
        spatial_size = (d, h, w)
        for i in range(self.num_layers):
            hidden_states.append(
                self.convlstm_layers[i].init_hidden(batch_size, spatial_size)
            )
        
        # Process input sequence
        for t in range(time_steps):
            input_t = x[:, t, :, :, :, :]
            features = F.relu(self.bn_in(self.conv_in(input_t)))
            
            for i, convlstm in enumerate(self.convlstm_layers):
                h, c = hidden_states[i]
                h, c = convlstm(features, (h, c))
                hidden_states[i] = (h, c)
                features = h
        
        # Generate future predictions
        predictions = []
        last_input = x[:, -1, :, :, :, :]
        
        for _ in range(num_future_steps):
            features = F.relu(self.bn_in(self.conv_in(last_input)))
            
            for i, convlstm in enumerate(self.convlstm_layers):
                h, c = hidden_states[i]
                h, c = convlstm(features, (h, c))
                hidden_states[i] = (h, c)
                features = h
            
            prediction = self.conv_out(features)
            predictions.append(prediction)
            
            # Use prediction as input for next step
            last_input = prediction
        
        return torch.stack(predictions, dim=1)
