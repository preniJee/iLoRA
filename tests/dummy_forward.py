import torch
from model.peft.tuners.moelora import Linear

# Define a small MoeLoRA linear layer
in_features = 8
out_features = 8
r = 4
num_moe = 2
layer = Linear(
    adapter_name="default",
    in_features=in_features,
    out_features=out_features,
    r=r,
    num_moe=num_moe,
    gating="Dense",
    gate_weights=[],
)

# Dummy input tensor
x = torch.randn(1, 1, in_features)

# Constant gate weights (equal weighting of experts)
constant_gate = torch.full((1, num_moe), 1.0 / num_moe)
layer.gate_weights = [constant_gate]

with torch.no_grad():
    out = layer(x)

print("Output shape:", out.shape)
