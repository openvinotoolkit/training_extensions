#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>

#include <vector>

void sum_to_act_channels(at::Tensor& target_tensor)
{
    // Sum over N
    target_tensor = target_tensor.sum(0, /*keepdims=*/ true);

    // Sum over H, W and the rest
    auto dim_count = target_tensor.dim();
    for (int64_t dim_idx = 2; dim_idx < dim_count; dim_idx++)
    {
        target_tensor = target_tensor.sum(dim_idx, /*keepdims=*/ true);
    }
}


namespace {

template <typename scalar_t>
at::Tensor wb_cpu_forward(
        at::Tensor input,
        bool per_channel) {

    auto mask = input.gt(0);

    auto output = at::zeros_like(input);
    if (per_channel)
    {
        output = output.fill_(-1.0);
        output = output.masked_fill_(mask, 1.0);

        auto scale = input.clone();
        scale = scale.abs();
        auto dim_count = input.dim();
        for (int64_t dim_idx = 1; dim_idx < dim_count; dim_idx++)
        {
            scale = scale.mean(dim_idx, true);
        }
        output = output * scale;
    }
    else
    {
        auto scale = input.abs().mean();
        output = output.fill_(-scale);
        output = output.masked_fill_(mask, scale);
    }
    return output;
}

template <typename scalar_t>
at::Tensor ab_cpu_forward(
        at::Tensor input,
        at::Tensor scale,
        at::Tensor thresholds) {
    auto mask = input.gt(thresholds * scale);
    auto output = at::zeros_like(input);
    AT_CHECK(scale.numel() == 1); // only single-scale mode is supported for now
    output = output.masked_fill_(mask, scale[0]);
    return output;
}

template <typename scalar_t>
std::vector<at::Tensor> ab_cpu_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor scale,
        at::Tensor output) {

    auto grad_input = at::zeros_like(grad_output);
    auto grad_scale = at::zeros_like(scale);
    auto grad_thresholds = at::zeros_like(grad_output);

    // Input gradient
    auto mask_lower = input.lt(scale);
    auto mask_higher = input.gt(0);
    auto mask_range = mask_lower * mask_higher;
    grad_input.masked_scatter_(mask_range, grad_output);

    // Scale gradient
    auto err = (output - input) / scale;
    auto inv_mask_lower = at::ones_like(mask_lower);
    inv_mask_lower.masked_fill_(mask_lower, 0);
    err.masked_fill_(inv_mask_lower, 1);
    grad_scale = grad_output * err;
    grad_scale = grad_scale.sum().view_as(scale);

    // Threshold gradient
    grad_thresholds.masked_scatter_(mask_range, -grad_output);
    sum_to_act_channels(grad_thresholds);

    return {grad_input, grad_scale, grad_thresholds};
}

#define CHECK_CPU(x) AT_CHECK(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_INPUT(x) CHECK_CPU(x)

at::Tensor wb_forward(
        at::Tensor input,
        bool per_channel) {
    CHECK_INPUT(input);

    at::Tensor output;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "wb_cpu_forward", ([&] {
      output = wb_cpu_forward<scalar_t>(input, per_channel);
    }));

    return output;
}

at::Tensor ab_forward(
        at::Tensor input,
        at::Tensor scale,
        at::Tensor thresholds) {
    CHECK_INPUT(input);
    CHECK_INPUT(scale);
    CHECK_INPUT(thresholds);

    at::Tensor output;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "ab_cpu_forward", ([&] {
      output = ab_cpu_forward<scalar_t>(input, scale, thresholds);
    }));

    return output;
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

    std::vector<at::Tensor> retval;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "ab_cpu_forward", ([&] {
      retval = ab_cpu_backward<scalar_t>(grad_output, input, scale, output);
    }));

    return retval;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("WeightBinarize_forward", &wb_forward, "Weight binarize forward");
  m.def("ActivationBinarize_forward", &ab_forward, "Activation binarize forward");
  m.def("ActivationBinarize_backward", &ab_backward, "Activation binarize backward");
}
