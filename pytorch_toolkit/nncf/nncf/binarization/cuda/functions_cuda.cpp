#include <torch/torch.h>

#include <vector>

at::Tensor wb_cuda_forward(
        at::Tensor input,
        bool per_channel);

at::Tensor ab_cuda_forward(
        at::Tensor input,
        at::Tensor scale,
        at::Tensor thresholds);

std::vector<at::Tensor> ab_cuda_backward(
    at::Tensor grad_output,
    at::Tensor input,
    at::Tensor scale,
    at::Tensor output);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor wb_forward(
        at::Tensor input,
        bool per_channel) {
    CHECK_INPUT(input);

    return wb_cuda_forward(input, per_channel);
}

at::Tensor ab_forward(
        at::Tensor input,
        at::Tensor scale,
        at::Tensor thresholds) {
    CHECK_INPUT(input);
    CHECK_INPUT(scale);
    CHECK_INPUT(thresholds);

    return ab_cuda_forward(input, scale, thresholds);
}

std::vector<at::Tensor> ab_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor scale,
        at::Tensor output) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(scale);
    CHECK_INPUT(output);

    return ab_cuda_backward(grad_output, input, scale, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("WeightBinarize_forward", &wb_forward, "Weight binarize forward");
  m.def("ActivationBinarize_forward", &ab_forward, "Activation binarize forward");
  m.def("ActivationBinarize_backward", &ab_backward, "Activation binarize backward");
}
