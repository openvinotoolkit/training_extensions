#include <torch/torch.h>

#include <vector>

namespace {

template <typename scalar_t>
at::Tensor q_cpu_forward(
        at::Tensor input,
        at::Tensor scale,
        scalar_t level_low,
        scalar_t level_high) {
    scalar_t s = level_high / scale.data<scalar_t>()[0];

    auto output = input * s;
    output = output.clamp_(level_low, level_high);
    output = output.round_();
    output = output.div_(s);

    return output;
}

template <typename scalar_t>
std::vector<at::Tensor> q_cpu_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor scale,
        scalar_t level_low,
        scalar_t level_high) {
    scalar_t rscale = 1 / scale.data<scalar_t>()[0];
    scalar_t alpha = level_low / level_high;
    scalar_t low_scale = scale.data<scalar_t>()[0] * alpha;

    auto mask_hi = input.gt(scale.data<scalar_t>()[0]);
    auto mask_lo = input.lt(low_scale);

    auto output = q_cpu_forward<scalar_t>(input, scale, level_low, level_high);

    auto err = at::sub(output, input);
    err = err.mul_(rscale);
    err = err.masked_fill_(mask_hi, 1);
    err = err.masked_fill_(mask_lo, alpha);
    err = err.mul_(grad_output);

    auto grad_scale = err.sum().view_as(scale);

    auto grad_input = grad_output.clone();
    grad_input = grad_input.masked_fill_(mask_hi.add_(mask_lo), 0);

    return {grad_input, grad_scale};
}
}

at::Tensor qs_forward(
        at::Tensor input,
        at::Tensor scale,
        int level_low,
        int level_high) {
    AT_ASSERTM(!input.type().is_cuda(), "input must be a CPU tensor");
    AT_ASSERTM(!scale.type().is_cuda(), "scale must be a CPU tensor");

    at::Tensor output;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "qss_cpu_forward", ([&] {
      output = q_cpu_forward<scalar_t>(input, scale, level_low, level_high);
    }));

    return output;
}

std::vector<at::Tensor> qs_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor scale,
        int level_low,
        int level_high) {
    AT_ASSERTM(!grad_output.type().is_cuda(), "grad_output must be a CPU tensor");
    AT_ASSERTM(!input.type().is_cuda(), "input must be a CPU tensor");
    AT_ASSERTM(!scale.type().is_cuda(), "scale must be a CPU tensor");

    std::vector<at::Tensor> results;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "qs_cpu_backward", ([&] {
        results = q_cpu_backward<scalar_t>(grad_output, input, scale, level_low, level_high);
    }));

    return results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("QuantizeSymmetric_forward", &qs_forward, "QuantizeSymmetric forward");
  m.def("QuantizeSymmetric_backward", &qs_backward, "QuantizeSymmetric backward");
}