## Implemented Compression Methods

Each compression method receives its own hyperparameters that are organized as a dictionary and basically stored in a JSON file that is deserialized when the training starts. Compression methods can be applied separately or together producing sparse, quantized, or both sparse and quantized models. For more information about the configuration, refer to the samples.

- [Quantization](./compression_algorithms/Quantization.md)
  - Symmetric and asymmetric quantization modes
  - Signed and unsigned
  - Per tensor/per channel
  - Exports to OpenVINO-supported FakeQuantize ONNX nodes
  - Arbitrary bitwidth
  - Mixed-bitwidth quantization
  - Automatic bitwidth assignment based on HAWQ
  - Automatic quantization parameter selection and activation quantizer setup based on HW config preset
- [Binarization](./compression_algorithms/Binarization.md)
  - XNOR, DoReFa weight binarization
  - Scale/threshold based per-channel activation binarization
- [Sparsity](./compression_algorithms/Sparsity.md)
  - Magnitude sparsity
  - Regularization-based (RB) sparsity
- [Filter pruning](./compression_algorithms/Pruning.md)