#pragma once
#include <torch/extension.h>

at::Tensor hamilton_product(at::Tensor inputs);
at::Tensor hamilton_product_grad(at::Tensor grads, at::Tensor mem, int inSize);