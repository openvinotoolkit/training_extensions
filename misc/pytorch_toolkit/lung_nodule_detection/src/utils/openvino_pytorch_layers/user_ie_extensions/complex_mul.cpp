// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "complex_mul.hpp"
// #include <ie_parallel.hpp>
#include <ie_common.h>

using namespace TemplateExtension;

//! [op:ctor]
ComplexMultiplication::ComplexMultiplication(
    const ov::Output<ov::Node>& inp0,
    const ov::Output<ov::Node>& inp1) : Op({inp0, inp1}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void ComplexMultiplication::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(1), outShape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> ComplexMultiplication::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");
    return std::make_shared<ComplexMultiplication>(new_args.at(0), new_args.at(1));
}
//! [op:copy]

//! [op:visit_attributes]
bool ComplexMultiplication::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool ComplexMultiplication::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float* inp0 = reinterpret_cast<float*>(inputs[0].data());
    const float* inp1 = reinterpret_cast<float*>(inputs[1].data());
    float* out = reinterpret_cast<float*>(outputs[0].data());

    size_t channels0 = inputs[0].get_shape()[1];
    size_t channels1 = inputs[1].get_shape()[1];
    size_t batch = inputs[0].get_shape()[0];
    size_t spatialSize = inputs[0].get_shape()[2] * inputs[0].get_shape()[3];

    // x1 = x_r * y_r - x_i * y_i
    // x2 = x_r * y_i + x_i * y_r
    if (channels0 == channels1)
        // InferenceEngine::parallel_for(channels0 * batch, [&](size_t ch) {
        for (size_t ch = 0; ch < channels0 * batch; ++ch) {
            for (int i = 0; i < spatialSize; ++i) {
                    int outIdx = (ch * spatialSize + i) * 2;
                    float real0 = inp0[outIdx];
                    float imag0 = inp0[outIdx + 1];
                    float real1 = inp1[outIdx];
                    float imag1 = inp1[outIdx + 1];
                    out[outIdx] = real0 * real1 - imag0 * imag1;
                    out[outIdx + 1] = real0 * imag1 + imag0 * real1;
            }
        }
    else if (channels1 == 1)
        // InferenceEngine::parallel_for(channels0 * batch, [&](size_t ch) {
        for (size_t ch = 0; ch < channels0 * batch; ++ch) {
            int b = ch / channels0;
            for (int i = 0; i < spatialSize; ++i) {
                int outIdx = (ch * spatialSize + i) * 2;
                int inpIdx = (b * spatialSize + i) * 2;
                float real0 = inp0[outIdx];
                float imag0 = inp0[outIdx + 1];
                float real1 = inp1[inpIdx];
                float imag1 = inp1[inpIdx + 1];
                out[outIdx] = real0 * real1 - imag0 * imag1;
                out[outIdx + 1] = real0 * imag1 + imag0 * real1;
            }
        }
    else
        IE_THROW() << "Wrong number of channels for second input!";

    return true;
}

bool ComplexMultiplication::has_evaluate() const {
    return true;
}
//! [op:evaluate]
