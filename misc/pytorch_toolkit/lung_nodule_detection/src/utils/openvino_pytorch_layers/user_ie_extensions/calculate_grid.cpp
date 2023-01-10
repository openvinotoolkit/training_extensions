// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "calculate_grid.hpp"

using namespace TemplateExtension;

//! [op:ctor]
CalculateGrid::CalculateGrid(const ov::Output<ov::Node>& inp_pos) : Op({inp_pos}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void CalculateGrid::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), outShape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> CalculateGrid::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");
    return std::make_shared<CalculateGrid>(new_args.at(0));
}
//! [op:copy]

//! [op:visit_attributes]
bool CalculateGrid::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool CalculateGrid::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const float* inpPos = reinterpret_cast<float*>(inputs[0].data());
    float* out = reinterpret_cast<float*>(outputs[0].data());

    std::set<std::tuple<int, int, int> > outPos;

    const size_t numPoints = inputs[0].get_shape()[0];
    static const std::vector<std::vector<int> > filters {{-1, -1, -1}, {-1, -1, 0}, {-1, 0, -1},
                                                         {-1, 0, 0}, {0, -1, -1}, {0, -1, 0},
                                                         {0, 0, -1}, {0, 0, 0}};

    std::vector<int> pos(3);
    for (size_t i = 0; i < numPoints; ++i) {
        for (size_t j = 0; j < filters.size(); ++j) {
            bool isValid = true;
            for (size_t k = 0; k < 3; ++k) {
                int val = static_cast<int>(inpPos[i * 3 + k]) + filters[j][k];
                if (val < 0 || val % 2) {
                    isValid = false;
                    break;
                }
                pos[k] = val;
            }
            if (isValid)
                outPos.insert(std::make_tuple(pos[0], pos[1], pos[2]));
        }
    }

    int i = 0;
    for (const auto it : outPos) {
        out[i * 3] = 0.5f + std::get<0>(it);
        out[i * 3 + 1] = 0.5f + std::get<1>(it);
        out[i * 3 + 2] = 0.5f + std::get<2>(it);
        i += 1;
    }
    memset(out + i * 3, 0, sizeof(float) * 3 * (numPoints - i));
    out[i * 3] = -1.0f;
    return true;
}

bool CalculateGrid::has_evaluate() const {
    return true;
}
//! [op:evaluate]
