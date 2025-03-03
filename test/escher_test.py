from escher import HistoryValueNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def test_history_value_network():
    input_size = 10
    layers = [32, 32, 16]
    model = HistoryValueNetwork(input_size, layers, activation=F.leaky_relu)

    # Create a mock input tensor
    test_input = torch.randn((1, input_size))

    # Run the model
    output = model(test_input)

    # Assertions
    assert output.shape == (1, 1), f"Unexpected output shape: {output.shape}"
    assert not torch.isnan(output).any(), "NaN values detected in output"

    print("Test passed!")

# Run test
test_history_value_network()