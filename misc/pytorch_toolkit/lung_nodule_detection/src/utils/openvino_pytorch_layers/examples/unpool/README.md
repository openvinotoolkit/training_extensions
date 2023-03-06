Guide of how to enable PyTorch `nn.MaxUnpool2d` in Intel OpenVINO.


## Description
There are two problems with OpenVINO and MaxUnpool at the moment of this guide creation:

* OpenVINO does not have Unpooling kernels
* PyTorch -> ONNX conversion is unimplemented for `nn.MaxUnpool2d`

So following this guide you will learn
* How to perform PyTorch -> ONNX conversion for unsupported layers
* How to convert ONNX to OpenVINO Intermediate Respresentation (IR) with extensions
* How to write custom CPU layers in OpenVINO

## Get ONNX model

MaxUnpool layer in PyTorch takes two inputs - input `features` from any layer and `indices` after MaxPool layer:

```python
self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
self.unpool = nn.MaxUnpool2d(2, stride=2)

output, indices = self.pool(x)
# ...
unpooled = self.unpool(features, indices)
```

If your version of PyTorch does not support ONNX model conversion with MaxUnpool, replace every unpool layer definition
```python
self.unpool = nn.MaxUnpool2d(2, stride=2)
```
to
```python
self.unpool = Unpool2d()
```

where `Unpool2d` defined in [unpool.py](./unpool.py). Also, replace op usage from

```python
self.unpool(features, indices)
```
to
```python
self.unpool.apply(features, indices)
```

See complete example in [export_model.py](./export_model.py).
