from escher import HistoryValueNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# python -m test.escher_test

def test_history_value_network():
    input_size = 32
    layers = [32, 16, 8]
    model = HistoryValueNetwork(input_size, layers, activation='leaky_relu')

    # Mock input tensor, size 1
    test_input = torch.randn((1, input_size))
    output = model((test_input, 1))
    assert output.shape == (1, 1), f"Unexpected output shape: {output.shape}"
    assert output.shape[0] == test_input.shape[0], "Batch size mismatch in output"
    assert not torch.isnan(output).any(), "NaN values detected in output"
    assert torch.isfinite(output).all(), "Non-finite values detected in output"

    # Different activation function
    for activation_fn in ['leaky_relu', 'relu', torch.sigmoid]:
        model = HistoryValueNetwork(input_size, layers, activation=activation_fn)
        output = model((test_input, 1))
        assert output.shape == (1, 1), f"Unexpected output shape for activation {activation_fn}: {output.shape}"

    # Mock input tensor, size 10
    test_input = torch.randn((10, input_size))  
    output = model((test_input, 1))
    assert output.shape == (10, 1), f"Unexpected output shape for batch size: {output.shape}"

    # Test with an input tensor of all zeros
    test_input_zeros = torch.zeros((1, input_size))
    output_zeros = model((test_input_zeros, 1))
    assert output_zeros.shape == (1, 1), f"Unexpected output shape for zeros input: {output_zeros.shape}"

    # Test with very large or very small values
    test_input_large = torch.full((1, input_size), 1e6)
    output_large = model((test_input_large, 1))
    assert output_large.shape == (1, 1), f"Unexpected output shape for large input: {output_large.shape}"

    print("Test passed!")

# Run test
test_history_value_network()