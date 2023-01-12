// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unpool.hpp"
// #include <ie_parallel.hpp>

using namespace TemplateExtension;

//! [op:ctor]
Unpool::Unpool(const ov::Output<ov::Node>& poolInp,
               const ov::Output<ov::Node>& poolOut,
               const ov::Output<ov::Node>& inp,
               const ov::Output<ov::Node>& shape) : Op({poolInp, poolOut, inp, shape}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void Unpool::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(3);
    auto poolInpShape = get_input_partial_shape(0).to_shape();
    outShape[0] = poolInpShape[0];  // Use only spatial dimensions from shape
    outShape[1] = poolInpShape[1];  // and restore batch and channels
    set_output_type(0, get_input_element_type(0), outShape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> Unpool::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 4, "Incorrect number of new arguments");
    return std::make_shared<Unpool>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}
//! [op:copy]

//! [op:visit_attributes]
bool Unpool::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool Unpool::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float* poolInp = reinterpret_cast<float*>(inputs[0].data());
    const float* poolOut = reinterpret_cast<float*>(inputs[1].data());
    const float* inp     = reinterpret_cast<float*>(inputs[2].data());
    float* out = reinterpret_cast<float*>(outputs[0].data());

    std::vector<size_t> poolInpDims = inputs[0].get_shape();
    std::vector<size_t> poolOutDims = inputs[1].get_shape();
    std::vector<size_t> inpDims = inputs[2].get_shape();
    std::vector<size_t> outDims = outputs[0].get_shape();

    const size_t batch    = poolInpDims[0];
    const size_t channels = poolInpDims[1];
    const size_t height   = poolInpDims[2];
    const size_t width    = poolInpDims[3];
    const size_t outHeight = outDims[2];
    const size_t outWidth  = outDims[3];
    const size_t poolOutHeight = poolOutDims[2];
    const size_t poolOutWidth  = poolOutDims[3];

    std::vector<bool> mask(inputs[1].get_size(), false);

    memset(out, 0, outputs[0].get_byte_size());
    // InferenceEngine::parallel_for(batch*channels, [&](size_t d) {
    for (size_t d = 0; d < batch * channels; ++d) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int poolOutIdx = (d * poolOutHeight + y / 2) * poolOutWidth + x / 2;
                int poolInpIdx = (d * height + y) * width + x;
                int dstIdx = d * outHeight * outWidth + (y * width + x);
                if (fabs(poolInp[poolInpIdx] - poolOut[poolOutIdx]) < 1e-5f && !mask[poolOutIdx]) {
                    out[dstIdx] = inp[poolOutIdx];
                    mask[poolOutIdx] = true;
                }
            }
        }
    }
    return true;
}

bool Unpool::has_evaluate() const {
    return true;
}
//! [op:evaluate]
