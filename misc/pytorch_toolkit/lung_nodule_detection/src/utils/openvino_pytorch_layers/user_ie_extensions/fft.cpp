// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fft.hpp"

#include <ie_common.h>
#include <details/ie_so_loader.h>
#include <opencv2/core/core_c.h>

using namespace TemplateExtension;

std::unique_ptr<InferenceEngine::details::SharedObjectLoader> so;
using cvCreateMatHeaderF = CvMat*(int, int, int);
using cvSetDataF = void(CvArr*, void*, int);
using cvReleaseMatF = void(CvMat**);
using cvDftF = void(const CvArr*, CvArr*, int, int);
using cvScaleF = void(const CvArr*, CvArr*, double, double);
using cvCloneMatF = CvMat*(const CvMat*);
using cvCopyF = void(const CvArr*, const CvArr*, const CvArr*);
using cvInitMatHeaderF = CvMat*(CvMat*, int, int, int, void*, int);
using cvGetRawDataF = void(const CvArr*, uchar**, int* step, CvSize* roi_size);
using cvReshapeF = CvMat*(const CvArr*, CvMat*, int, int);
using cvCreateDataF = void(CvArr*);
using cvReleaseDataF = void(CvArr*);

bool loadOpenCV() {
    static bool loaded = false;
    if (!loaded) {
        loaded = true;
        try {
#ifdef _WIN32
            so.reset(new InferenceEngine::details::SharedObjectLoader("opencv_core.dll"));
#elif defined(__APPLE__)
            so.reset(new InferenceEngine::details::SharedObjectLoader("libopencv_core.dylib"));
#else
            so.reset(new InferenceEngine::details::SharedObjectLoader("libopencv_core.so"));
#endif
        } catch (InferenceEngine::details::InferenceEngineException& ex) {
            return false;
        }
    }
    return loaded;
}

void fftshift(CvMat* src, bool inverse) {
    static auto cvCloneMat = reinterpret_cast<cvCloneMatF*>(so->get_symbol("cvCloneMat"));
    static auto cvCopy = reinterpret_cast<cvCopyF*>(so->get_symbol("cvCopy"));
    static auto cvInitMatHeader = reinterpret_cast<cvInitMatHeaderF*>(so->get_symbol("cvInitMatHeader"));
    static auto cvGetRawData = reinterpret_cast<cvGetRawDataF*>(so->get_symbol("cvGetRawData"));
    static auto cvReleaseMat = reinterpret_cast<cvReleaseMatF*>(so->get_symbol("cvReleaseMat"));


    // tl | tr        br | bl
    // ---+---   ->   ---+---
    // bl | br        tr | tl

    float* data;
    int step;
    CvSize size;
    cvGetRawData(src, (uchar**)&data, &step, &size);

    int height = size.height;
    int width = size.width;
    int h2 = height / 2;
    int w2 = width / 2;

    if (height % 2 || width % 2) {
        // Swap rows.
        CvMat* srcTop = new CvMat();
        CvMat* srcBot = new CvMat();
        CvMat* dstTop = new CvMat();
        CvMat* dstBot = new CvMat();
        int topH = inverse ? h2 : (h2 + height % 2);
        int botH = height - topH;
        cvInitMatHeader(srcTop, topH, width, CV_32FC2, data, step);
        cvInitMatHeader(srcBot, botH, width, CV_32FC2, data + topH * width * 2, step);
        cvInitMatHeader(dstTop, topH, width, CV_32FC2, data + botH * width * 2, step);
        cvInitMatHeader(dstBot, botH, width, CV_32FC2, data, step);

        CvMat* tmp = cvCloneMat(srcTop);
        cvCopy(srcBot, dstBot, 0);
        cvCopy(tmp, dstTop, 0);

        cvReleaseMat(&tmp);
        delete srcTop;
        delete srcBot;
        delete dstTop;
        delete dstBot;

        // Swap columns.
        CvMat* srcL = new CvMat();
        CvMat* srcR = new CvMat();
        CvMat* dstL = new CvMat();
        CvMat* dstR = new CvMat();
        int leftW = inverse ? w2 : (w2 + width % 2);
        int rightW = width - leftW;

        cvInitMatHeader(srcL, height, leftW, CV_32FC2, data, step);
        cvInitMatHeader(srcR, height, rightW, CV_32FC2, data + leftW * 2, step);
        cvInitMatHeader(dstL, height, leftW, CV_32FC2, data + rightW * 2, step);
        cvInitMatHeader(dstR, height, rightW, CV_32FC2, data, step);

        tmp = cvCloneMat(srcL);
        cvCopy(srcR, dstR, 0);
        cvCopy(tmp, dstL, 0);

        cvReleaseMat(&tmp);
        delete srcL;
        delete srcR;
        delete dstL;
        delete dstR;

        return;
    }

    CvMat* tl = new CvMat();
    CvMat* tr = new CvMat();
    CvMat* bl = new CvMat();
    CvMat* br = new CvMat();

    cvInitMatHeader(tl, h2, w2, CV_32FC2, data, step);
    cvInitMatHeader(tr, h2, w2, CV_32FC2, data + width, step);
    cvInitMatHeader(bl, h2, w2, CV_32FC2, data + height * width, step);
    cvInitMatHeader(br, h2, w2, CV_32FC2, data + height * width + width, step);

    CvArr* mask = 0;
    CvMat* tmp = cvCloneMat(tl);
    cvCopy(br, tl, mask);
    cvCopy(tmp, br, mask);

    cvCopy(tr, tmp, mask);
    cvCopy(bl, tr, mask);
    cvCopy(tmp, bl, mask);

    cvReleaseMat(&tmp);

    delete tl;
    delete tr;
    delete bl;
    delete br;
}

//! [op:ctor]
FFT::FFT(const ov::Output<ov::Node>& inp,
         const ov::Output<ov::Node>& dims,
         bool inverse,
         bool centered) : Op({inp, dims}) {
    loadOpenCV();
    constructor_validate_and_infer_types();
    this->inverse = inverse;
    this->centered = centered;
}
//! [op:ctor]

//! [op:validate]
void FFT::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), outShape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> FFT::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");
    return std::make_shared<FFT>(new_args.at(0), new_args.at(1), inverse, centered);
}
//! [op:copy]

//! [op:visit_attributes]
bool FFT::visit_attributes(ov::AttributeVisitor& visitor) {
    int inverse_i = static_cast<int>(inverse);
    int centered_i = static_cast<int>(centered);
    visitor.on_attribute("inverse", inverse_i);
    visitor.on_attribute("centered", centered_i);
    inverse = static_cast<bool>(inverse_i);
    centered = static_cast<bool>(centered_i);
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool FFT::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    static auto cvSetData = reinterpret_cast<cvSetDataF*>(so->get_symbol("cvSetData"));
    static auto cvCreateMatHeader = reinterpret_cast<cvCreateMatHeaderF*>(so->get_symbol("cvCreateMatHeader"));
    static auto cvDFT = reinterpret_cast<cvDftF*>(so->get_symbol("cvDFT"));
    static auto cvScale = reinterpret_cast<cvScaleF*>(so->get_symbol("cvConvertScale"));
    static auto cvReleaseMat = reinterpret_cast<cvReleaseMatF*>(so->get_symbol("cvReleaseMat"));
    static auto cvReshape = reinterpret_cast<cvReshapeF*>(so->get_symbol("cvReshape"));
    static auto cvCloneMat = reinterpret_cast<cvCloneMatF*>(so->get_symbol("cvCloneMat"));
    static auto cvCreateData = reinterpret_cast<cvCreateDataF*>(so->get_symbol("cvCreateData"));
    static auto cvReleaseData = reinterpret_cast<cvReleaseDataF*>(so->get_symbol("cvReleaseData"));
    static auto cvCopy = reinterpret_cast<cvCopyF*>(so->get_symbol("cvCopy"));

    float* inpData = reinterpret_cast<float*>(inputs[0].data());

    if (inputs[1].get_element_type() != ov::element::i32)
        IE_THROW() << "Unexpected dims type: " << inputs[1].get_element_type();

    int32_t* signalDimsData = reinterpret_cast<int32_t*>(inputs[1].data());
    float* outData = reinterpret_cast<float*>(outputs[0].data());
    std::vector<size_t> dims = inputs[0].get_shape();
    const size_t numSignalDims = inputs[1].get_shape()[0];

    if (!(dims.size() == 3 && (numSignalDims == 1 && signalDimsData[0] == 1) ||
          dims.size() == 4 && ((numSignalDims == 1 && signalDimsData[0] == 1) ||
                               (numSignalDims == 2 && signalDimsData[0] == 1 && signalDimsData[1] == 2)) ||
          dims.size() == 5 && ((numSignalDims == 2 && signalDimsData[0] == 1 && signalDimsData[1] == 2) ||
                               (numSignalDims == 2 && signalDimsData[0] == 2 && signalDimsData[1] == 3)))) {
        std::ostringstream ss;
        for (size_t i = 0; i < numSignalDims; ++i)
            ss << signalDimsData[i] << " ";
        IE_THROW() << "Unsupported configuration: Input dims " << dims.size() << " and signal dims " << ss.str();
    }

    const int batch = dims[0];

    if (dims.size() == 5 && numSignalDims == 2 && signalDimsData[0] == 1 && signalDimsData[1] == 2) {
        const int channels = dims[1];
        int rows = dims[2];
        int cols = dims[3];
        const int planeSize = channels * rows * cols;
        // InferenceEngine::parallel_for(batch * cols, [&](size_t d) {
        for (size_t d = 0; d < batch * cols; ++d) {
            int b = d / cols;
            int col = d % cols;
            // Copy a slice from input
            CvMat* inpSlice = cvCreateMatHeader(channels * rows, 1, CV_32FC2);
            CvMat* outSlice = cvCreateMatHeader(channels * rows, 1, CV_32FC2);
            cvSetData(inpSlice, reinterpret_cast<void*>(inpData + (b * planeSize + col) * 2), cols * 2 * sizeof(float));
            cvSetData(outSlice, reinterpret_cast<void*>(outData + (b * planeSize + col) * 2), cols * 2 * sizeof(float));

            CvMat* inp_col = cvCloneMat(inpSlice);

            CvMat inp_header, *inp;
            inp = cvReshape(inp_col, &inp_header, 2, channels);

            CvMat* out = cvCreateMatHeader(channels, rows, CV_32FC2);
            cvCreateData(out);

            if (centered)
                fftshift(inp, true);

            if (inverse)
                cvDFT(inp, out, CV_DXT_INVERSE, 0);
            else
                cvDFT(inp, out, CV_DXT_FORWARD, 0);
            cvScale(out, out, 1.0 / sqrtf(channels * rows), 0);

            if (centered)
                fftshift(out, false);

            CvMat out_col_header, *out_col;
            out_col = cvReshape(out, &out_col_header, 2, channels * rows);

            cvCopy(out_col, outSlice, 0);

            cvReleaseData(inp_col);
            cvReleaseMat(&inp_col);

            cvReleaseData(out);
            cvReleaseMat(&out);

            cvReleaseMat(&inpSlice);
            cvReleaseMat(&outSlice);
        }
    } else if (dims.size() == 5 && numSignalDims == 2 && signalDimsData[0] == 2 && signalDimsData[1] == 3) {
        const int channels = dims[1];
        int rows = dims[2];
        int cols = dims[3];
        int planeSize = rows * cols * 2;  // 2 is last dimension size
        // InferenceEngine::parallel_for(batch * channels, [&](size_t d) {
        for (size_t d = 0; d < batch * channels; ++d) {
            CvMat* inp = cvCreateMatHeader(rows, cols, CV_32FC2);
            CvMat* out = cvCreateMatHeader(rows, cols, CV_32FC2);
            cvSetData(inp, reinterpret_cast<void*>(inpData + d * planeSize), cols * 2 * sizeof(float));
            cvSetData(out, reinterpret_cast<void*>(outData + d * planeSize), cols * 2 * sizeof(float));

            if (centered)
                fftshift(inp, true);

            if (inverse)
                cvDFT(inp, out, CV_DXT_INVERSE, 0);
            else
                cvDFT(inp, out, CV_DXT_FORWARD, 0);
            cvScale(out, out, 1.0 / sqrtf(cols * rows), 0);

            if (centered)
                fftshift(out, false);

            cvReleaseMat(&inp);
            cvReleaseMat(&out);
        }
    } else if (dims.size() == 4 && numSignalDims == 2 && signalDimsData[0] == 1 && signalDimsData[1] == 2) {
        int rows = dims[1];
        int cols = dims[2];
        int planeSize = rows * cols * 2;  // 2 is last dimension size
        // InferenceEngine::parallel_for(batch, [&](size_t d) {
        for (size_t d = 0; d < batch; ++d) {
            CvMat* inp = cvCreateMatHeader(rows, cols, CV_32FC2);
            CvMat* out = cvCreateMatHeader(rows, cols, CV_32FC2);
            cvSetData(inp, reinterpret_cast<void*>(inpData + d * planeSize), cols * 2 * sizeof(float));
            cvSetData(out, reinterpret_cast<void*>(outData + d * planeSize), cols * 2 * sizeof(float));

            if (centered)
                fftshift(inp, true);

            if (inverse)
                cvDFT(inp, out, CV_DXT_INVERSE, 0);
            else
                cvDFT(inp, out, CV_DXT_FORWARD, 0);
            cvScale(out, out, 1.0 / sqrtf(cols * rows), 0);

            if (centered)
                fftshift(out, false);

            cvReleaseMat(&inp);
            cvReleaseMat(&out);
        }
    } else if (dims.size() == 4 && numSignalDims == 1 && signalDimsData[0] == 1) {
        int rows = dims[1];
        int cols = dims[2];

        const int planeSize = rows;
        // InferenceEngine::parallel_for(batch * cols, [&](size_t d) {
        for (size_t d = 0; d < batch * cols; ++d) {
            int b = d / cols;
            int col = d % cols;
            CvMat* inp = cvCreateMatHeader(rows, 1, CV_32FC2);
            CvMat* out = cvCreateMatHeader(rows, 1, CV_32FC2);
            cvSetData(inp, reinterpret_cast<void*>(inpData + (b * planeSize * cols + col) * 2), cols * 2 * sizeof(float));
            cvSetData(out, reinterpret_cast<void*>(outData + (b * planeSize * cols + col) * 2), cols * 2 * sizeof(float));

            if (centered)
                fftshift(inp, true);

            if (inverse)
                cvDFT(inp, out, CV_DXT_INVERSE, 0);
            else
                cvDFT(inp, out, CV_DXT_FORWARD, 0);
            cvScale(out, out, 1.0 / sqrtf(rows), 0);

            if (centered)
                fftshift(out, false);

            cvReleaseMat(&inp);
            cvReleaseMat(&out);
        }
    } else if (dims.size() == 3) {
        int rows = dims[0];
        int cols = dims[1];
        CvMat* inp = cvCreateMatHeader(rows, cols, CV_32FC2);
        CvMat* out = cvCreateMatHeader(rows, cols, CV_32FC2);
        cvSetData(inp, reinterpret_cast<void*>(inpData), cols * 2 * sizeof(float));
        cvSetData(out, reinterpret_cast<void*>(outData), cols * 2 * sizeof(float));

        if (inverse)
            cvDFT(inp, out, CV_DXT_INVERSE | CV_DXT_ROWS, 0);
        else
            cvDFT(inp, out, CV_DXT_FORWARD | CV_DXT_ROWS, 0);
        cvScale(out, out, 1.0 / sqrtf(cols), 0);

        cvReleaseMat(&inp);
        cvReleaseMat(&out);
    }
    return true;
}

bool FFT::has_evaluate() const {
    return true;
}
//! [op:evaluate]
