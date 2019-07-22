import tensorflow as tf

EMBEDDINGS_DIM = 256
WEIGHT_DECAY = 0.0001

def l2_normalized_embeddings(inputs):
    output = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    output = tf.keras.layers.Reshape([1, 1, output.shape[-1]])(output)
    output = tf.keras.layers.Conv2D(filters=EMBEDDINGS_DIM, kernel_size=1, padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=WEIGHT_DECAY))(output)
    output = tf.keras.layers.Flatten()(output)
    output = tf.nn.l2_normalize(output * 1000, axis=-1, epsilon=1e-13)
    return output

def keras_applications_mobilenetv2(inputs, alpha=1.0):
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

    base_model = MobileNetV2(alpha=alpha, include_top=False,
                             weights='imagenet', input_tensor=inputs)

    output = l2_normalized_embeddings(base_model.output)
    return tf.keras.Model(inputs, output)

def keras_applications_resnet50(inputs):
    from tensorflow.keras.applications.resnet50 import ResNet50

    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    output = l2_normalized_embeddings(base_model.output)

    return tf.keras.Model(inputs, output)
