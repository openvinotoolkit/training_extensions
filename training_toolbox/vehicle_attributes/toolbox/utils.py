def get_checkpoint_variable_names(ckpt):
  from tensorflow.python import pywrap_tensorflow
  reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
  vars_dict = reader.get_variable_to_shape_map()
  return [v for v in vars_dict.keys()]
