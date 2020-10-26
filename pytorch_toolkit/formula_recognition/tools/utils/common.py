"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import os

ENCODER_INPUTS = "imgs"
ENCODER_OUTPUTS = "row_enc_out,hidden,context,init_0"
DECODER_INPUTS = "dec_st_h,dec_st_c,output_prev,row_enc_out,tgt"
DECODER_OUTPUTS = "dec_st_h_t,dec_st_c_t,output,logit"


def read_net(model_xml, ie):
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    model = ie.read_network(model_xml, model_bin)

    return model


def export_model_if_not_yet(exporter, model, model_type, ir=False):
    assert model_type in ('encoder', 'decoder')
    result_model_exists = os.path.exists(model)
    if ir:
        model_xml = model.replace(".onnx", '.xml')
        model_bin = model.replace(".onnx", '.bin')
        result_model_exists = os.path.exists(model_xml) and os.path.exists(model_bin)
    if not result_model_exists:
        print(f"Model {model} does not exists, exporting it...")
        export_function_name = f"export_{model_type}{'_ir' if ir else ''}"
        getattr(exporter, export_function_name)()
