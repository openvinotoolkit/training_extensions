#include <torch/torch.h>
#include <iostream>

#include <vector>

at::Tensor qs_cuda_forward(
        at::Tensor input,
        at::Tensor scale,
        int level_low,
        int level_high);

std::vector<at::Tensor> qs_cuda_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor scale,
        int level_low,
        int level_high);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor qs_forward(
        at::Tensor input,
        at::Tensor scale,
        int level_low,
        int level_high) {
    CHECK_INPUT(input);
    CHECK_INPUT(scale);

    return qs_cuda_forward(input, scale, level_low, level_high);
}

std::vector<at::Tensor> qs_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor scale,
        int level_low,
        int level_high) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(scale);

    return qs_cuda_backward(grad_output, input, scale, level_low, level_high);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("QuantizeSymmetric_forward", &qs_forward, "QuantizeSymmetric forward");
  m.def("QuantizeSymmetric_backward", &qs_backward, "QuantizeSymmetric backward");
}
