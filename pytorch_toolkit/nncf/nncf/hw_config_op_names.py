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


class HWConfigOpName:
    CONVOLUTION = "Convolution"
    DEPTHWISECONVOLUTION = "DepthWiseConvolution"
    MATMUL = "MatMul"
    ADD = "Add"
    MULTIPLY = "Multiply"
    MAXIMUM = "Maximum"
    LESS = "Less"
    LESSEQUAL = "LessEqual"
    GREATER = "Greater"
    GREATEREQUAL = "GreaterEqual"
    DIVIDE = "Divide"
    MINIMUM = "Minimum"
    EQUAL = "Equal"
    SUBTRACT = "Subtract"
    NOTEQUAL = "NotEqual"
    FLOORMOD = "FloorMod"
    LOGICALOR = "LogicalOr"
    LOGICALXOR = "LogicalXor"
    LOGICALAND = "LogicalAnd"
    LOGICALNOT = "LogicalNot"
    POWER = "Power"
    AVGPOOL = "AvgPool"
    REDUCEMEAN = "ReduceMean"
    MAXPOOL = "MaxPool"
    REDUCEMAX = "ReduceMax"
    INTERPOLATE = "Interpolate"
    MVN = "MVN"
    RESHAPE = "Reshape"
    CONCAT = "Concat"
    FLATTEN = "Flatten"
    SQUEEZE = "Squeeze"
    UNSQUEEZE = "Unsqueeze"
    SPLIT = "Split"
    CROP = "Crop"
    TRANSPOSE = "Transpose"
    TILE = "Tile"
