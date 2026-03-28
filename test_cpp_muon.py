import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
#include <torch/extension.h>

torch::Tensor zeropower_via_newtonschulz5_cpp(torch::Tensor G, int steps, double eps) {
    double a = 3.4445, b = -4.7750, c = 2.0315;
    auto X = G.to(torch::kBFloat16);
    X = X / (X.norm() + eps);
    
    bool transposed = G.size(0) > G.size(1);
    if (transposed) {
        X = X.t();
    }
    
    for (int i = 0; i < steps; ++i) {
        auto A = torch::matmul(X, X.t());
        auto B = b * A + c * torch::matmul(A, A);
        X = a * X + torch::matmul(B, X);
    }
    
    if (transposed) {
        X = X.t();
    }
    return X;
}
"""

module = load_inline(
    name="muon_cpp",
    cpp_sources=cpp_source,
    functions=["zeropower_via_newtonschulz5_cpp"],
    verbose=True,
)

G = torch.randn(128, 128, device='cuda')
out = module.zeropower_via_newtonschulz5_cpp(G, 10, 1e-7)
print(out.shape)
