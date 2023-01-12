// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_conv.hpp"

using namespace TemplateExtension;

//! [op:ctor]
SparseConv::SparseConv(const ov::Output<ov::Node>& features,
                       const ov::Output<ov::Node>& inp_pos,
                       const ov::Output<ov::Node>& out_pos,
                       const ov::Output<ov::Node>& kernel,
                       const ov::Output<ov::Node>& offset) : Op({features, inp_pos, out_pos, kernel, offset}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void SparseConv::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(2);
    auto kernelShape = get_input_partial_shape(3);
    outShape[1] = kernelShape[4];
    set_output_type(0, get_input_element_type(0), outShape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> SparseConv::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 5, "Incorrect number of new arguments");
    return std::make_shared<SparseConv>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4));
}
//! [op:copy]

//! [op:visit_attributes]
bool SparseConv::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool SparseConv::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float* features = reinterpret_cast<float*>(inputs[0].data());
    const float* inpPos = reinterpret_cast<float*>(inputs[1].data());
    const float* outPos = reinterpret_cast<float*>(inputs[2].data());
    const float* kernel = reinterpret_cast<float*>(inputs[3].data());
    const float* offset = reinterpret_cast<float*>(inputs[4].data());
    float* out = reinterpret_cast<float*>(outputs[0].data());
    memset(out, 0, outputs[0].get_byte_size());

    size_t numInpPoints = inputs[1].get_shape()[0];
    const size_t numOutPoints = inputs[2].get_shape()[0];
    std::vector<size_t> kernelDims = inputs[3].get_shape();

    // Kernel layout is DxHxWxICxOH
    const int kd = kernelDims[0];
    const int kh = kernelDims[1];
    const int kw = kernelDims[2];
    const int IC = kernelDims[3];
    const int OC = kernelDims[4];

    // See https://github.com/isl-org/Open3D/blob/master/python/open3d/ml/torch/python/layers/convolutions.py
    float rw = kw * 0.51f;
    float rh = kh * 0.51f;
    float rd = kd * 0.51f;

    for (size_t i = 0; i < numInpPoints; ++i) {
        if (inpPos[i * 3] < 0) {
            numInpPoints = i;
            break;
        }
    }

    for (size_t i = 0; i < numOutPoints; ++i) {
        const float xi = outPos[i * 3] - offset[0];
        const float yi = outPos[i * 3 + 1] - offset[1];
        const float zi = outPos[i * 3 + 2] - offset[2];

        // Accumulate features which inside the kernel
        for (size_t j = 0; j < numInpPoints; ++j) {
            const float xj = inpPos[j * 3];
            const float yj = inpPos[j * 3 + 1];
            const float zj = inpPos[j * 3 + 2];

            if (xi - rw <= xj && xj <= xi + rw &&
                yi - rh <= yj && yj <= yi + rh &&
                zi - rd <= zj && zj <= zi + rd) {

                const int w = std::min(static_cast<int>(xj - xi + kw * 0.5f), kw - 1);
                const int h = std::min(static_cast<int>(yj - yi + kh * 0.5f), kh - 1);
                const int d = std::min(static_cast<int>(zj - zi + kd * 0.5f), kd - 1);

                const float* featuresOffset = features + j * IC;
                for (size_t ic = 0; ic < IC; ++ic) {
                    const float* kernelOffset = kernel + OC * (ic + IC * (w + kw * (h + kh * d)));
                    for (size_t oc = 0; oc < OC; ++oc) {
                        out[i * OC + oc] += kernelOffset[oc] * featuresOffset[ic];
                    }
                }
            }
        }
    }
    return true;
}

bool SparseConv::has_evaluate() const {
    return true;
}
//! [op:evaluate]
