# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from __future__ import print_function

import time
from abc import ABCMeta, abstractmethod
from datetime import datetime
from os.path import exists, join, basename

import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph  # pylint: disable=no-name-in-module

BASE_FILE_NAME = 'converted_model'
PB_FILE_NAME = '{}.pbtxt'.format(BASE_FILE_NAME)


class BaseMonitor(object):
    """Base class for network monitors, which allows to control the network behaviour.
    """

    __metaclass__ = ABCMeta

    def __init__(self, params, batch_size, num_gpu, log_dir, src_scope, snapshot_path, init_model_path):
        """Constructor.

        :param params: Dictionary with model parameters
        :param batch_size: Size of batch
        :param num_gpu: Number of GPU devices
        :param log_dir: Path to directory for logging
        :param src_scope: Name of source network scope to load variables from
        :param snapshot_path: Path to model snapshot
        :param init_model_path: Path to model weights to initialize from
        """

        self._params = params
        self._batch_size = batch_size
        self._num_gpu = num_gpu
        self._log_dir = log_dir
        self._src_scope = src_scope
        self._snapshot_path = snapshot_path
        self._init_model_path = init_model_path

        self._log_iter = 100
        self._save_iter = 1000
        self._max_num_saves = 1000
        self._train_prefetch_size = 1
        self._train_data_process_num_threads = 5
        self._test_prefetch_size = 1
        self._test_data_process_num_threads = 5
        self._max_grad_norm = 1.0

    @abstractmethod
    def _create_model(self, network_input, global_step=None, is_training=None,
                      merge_train_ops=False, merging_transit=False):
        """Abstract method for model creating.

        :param network_input: Network input tensors
        :param global_step: Training step variable
        :param is_training: Training indicator
        :param merge_train_ops: Whether to run with merged train Ops
        :param merging_transit: Whether to run in train Ops merging mode
        :return: Created model
        """

        pass

    @abstractmethod
    def _create_train_dataset(self, data_path):
        """Abstract method for training dataset creation.

        :param data_path: Path to data
        :return: Created dataset
        """

        pass

    @abstractmethod
    def _create_test_dataset(self, data_path):
        """Abstract method for evaluation dataset creation.

        :param data_path:
        :return: Created dataset
        """

        pass

    def _transfer_train_parameters(self, sess, model):
        """Loads available model params from the specified snapshot.

        :param sess: Session to load in
        :param model: Model to load variables
        :return:
        """

        model.load_available_variables(self._init_model_path, sess, self._src_scope,
                                       '{}/{}'.format(self._params.name, self._params.backbone))

    @abstractmethod
    def _get_out_node_names(self):
        """Returns network output node names.

        :return: List of names
        """

        pass

    @abstractmethod
    def _get_test_ops(self, model, network_inputs):
        """Returns list of operations to carry out model evaluation.

        :param model: Model for evaluation
        :param network_inputs: Network input tensors
        :return: List of operations
        """

        pass

    @abstractmethod
    def _test(self, sess, network_inputs, test_ops):
        """Carry out internal testing step.

        :param sess: Session
        :param network_inputs: Network input tensors
        :param test_ops: List of testing operations
        """

        pass

    @abstractmethod
    def _demo(self, sess, data_path, input_image, inference_ops, out_scale):
        """Carry out internal demonstration step.

        :param sess: Session
        :param data_path: Path to data to demonstrate on
        :param input_image: Input network tensor for images
        :param inference_ops: List of operations to carry out inference
        :param out_scale: Parameter to scale final image
        """

        pass

    def _get_extra_param_names(self):
        """Returns list of model-specific parameters names for restoring

        :return: List of names
        """

        return []

    @staticmethod
    def _log(message):
        """Prints logging message.

        :param message: Message to print
        """

        print(message)

    @staticmethod
    def _start_session(log_device_placement=False, allow_growth=True, allow_soft_placement=True):
        """Creates new session and initializes variables.

        :param log_device_placement: Whether to log device placement for variables
        :param allow_growth: Whether to allow growth of GPU memory usage
        :param allow_soft_placement: Whether to allow soft placement of data on GPU
        :return: New session
        """

        config = tf.ConfigProto()
        config.log_device_placement = log_device_placement
        config.gpu_options.allow_growth = allow_growth  # pylint: disable=no-member
        config.allow_soft_placement = allow_soft_placement

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        return sess

    @staticmethod
    def _close_session(sess):
        """Close specified session.

        :param sess: Session to close
        """

        sess.close()

    @staticmethod
    def _collect_summary():
        """Creates Op to collect summary operations.

        :return: Summary Op
        """

        train_summaries = \
            tf.get_collection('weights_summary') + \
            tf.get_collection('activation_summary') + \
            tf.get_collection('bn_summary') + \
            tf.get_collection('loss_summary') + \
            tf.get_collection('lr_summary') + \
            tf.get_collection('grad_summary') + \
            tf.get_collection('accuracy_summary')
        return tf.summary.merge(train_summaries)

    def _restore_train_parameters(self, sess, model):
        """Restores model parameters from the specified input for training.

        :param sess: Session to load in
        :param model: Model for parameters loading
        """

        self._log('\nParameters restoring...')
        if exists(self._snapshot_path + '.index'):
            loader = tf.train.Saver()
            loader.restore(sess, self._snapshot_path)
            self._log('Full model restored from: {}'.format(self._snapshot_path))
        elif exists(self._init_model_path + '.index') and self._src_scope != '':
            self._transfer_train_parameters(sess, model)
        else:
            self._log('No source to load train parameters')

    def _restore_inference_parameters(self, sess, model, ckpt_path=''):
        """Restores model parameters from the specified input for inference.

        :param sess: Session to load in
        :param model: Model for parameters loading
        :param ckpt_path: Extra parameter with snapshot path (can be empty)
        """

        model_path = ckpt_path if ckpt_path is not None and ckpt_path != '' else self._snapshot_path

        self._log('\nParameters restoring...')
        if exists(model_path + '.index'):
            extra_param_ends = self._get_extra_param_names()

            model.load_available_variables(model_path, sess, self._params.name, self._params.name,
                                           extra_ends=extra_param_ends)
        else:
            self._log('No source to load test parameters')

    def _multi_gpu_training(self, global_step, is_training_var, data_iterator):
        """Creates ops for multi-GPU training.

        :param global_step: Iteration variable
        :param is_training_var: Indicator variable of training mode
        :param data_iterator: Network input data iterator
        :return: Tuple of train and loss Ops and created models
        """

        def _average_gradients(tower_grads, max_grad_norm=None):
            """Creates Op to merge list of gradients according its name.

            :param tower_grads: List of gradients by tower
            :param max_grad_norm: Max gradient norm to clip
            :return: List of averaged gradients
            """

            averaged_grads = []
            for grad_and_vars in zip(*tower_grads):
                grads = []
                for grad_value, _ in grad_and_vars:
                    expanded_g = tf.expand_dims(grad_value, 0)

                    grads.append(expanded_g)

                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)

                if max_grad_norm is not None:
                    grad = tf.clip_by_norm(grad, max_grad_norm)

                var_name = grad_and_vars[0][1]

                grad_and_var = (grad, var_name)
                averaged_grads.append(grad_and_var)

            return averaged_grads

        optimizer = None
        towers = []
        tower_gradients = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(self._num_gpu):
                with tf.device('/gpu:{}'.format(i)):
                    with tf.name_scope('tower_{}'.format(i)):
                        tower_inputs = data_iterator.get_next()

                        tower = self._create_model(tower_inputs, global_step, is_training_var)
                        towers.append(tower)

                        tf.get_variable_scope().reuse_variables()

                        if optimizer is None:
                            optimizer = tower.create_optimizer()

                        tower_gradients.append(optimizer.compute_gradients(tower.total_loss))

        gradients = _average_gradients(tower_gradients, max_grad_norm=self._max_grad_norm)
        optimizer_op = optimizer.apply_gradients(gradients, global_step=global_step)

        extra_ops = tf.get_collection('train_extra_ops')
        with tf.control_dependencies(extra_ops):
            with tf.control_dependencies([optimizer_op]):
                train_op = tf.no_op(name='train_op')

        loss_op = tf.add_n([t.total_loss for t in towers]) / float(self._num_gpu)

        return train_op, loss_op, towers

    def _init_data(self, sess, data_init_op):
        """Initializes network input data.

        :param sess: Session to load
        :param data_init_op: Initialization Op
        """

        self._log('\nData loading...')
        sess.run(data_init_op)
        self._log('Finished.')

    @staticmethod
    def _get_iter_id(snapshot_path):
        """Parse snapshot name to extract iteration number.

        :param snapshot_path: Snapshot name
        :return: Iteration number
        """

        return int(basename(snapshot_path).split('-')[-1])

    def _optimize(self, sess, global_step, train_op, loss_op, summary_op, is_training):
        """Internal method to carry out model training procedure.

        :param sess: Session to run in.
        :param global_step: Step variable
        :param train_op: Operation to carry out training step
        :param loss_op: Operation to calculate loss scalar value
        :param summary_op: Operation to dump summary Ops
        :param is_training: Indicator of training mode
        """

        assert self._log_dir is not None and self._log_dir != ''

        summary_writer = tf.summary.FileWriter(self._log_dir, sess.graph)
        model_saver = tf.train.Saver(max_to_keep=self._max_num_saves)

        logfile_name = '{}_train_log.txt'.format(datetime.now()).replace(' ', '-').replace(':', '-')
        log_stream = open(join(self._log_dir, logfile_name), 'w')

        start_time = time.time()
        num_steps = 0

        self._log('\nOptimization...')
        init_train_step = sess.run(global_step)
        for train_step in xrange(init_train_step, self._params.max_train_steps):
            if train_step % self._log_iter == 0:
                _, total_loss, train_summary = sess.run([train_op, loss_op, summary_op], feed_dict={is_training: True})
                num_steps += 1

                end_time = time.time()
                mean_iter_time = float(end_time - start_time) / float(num_steps)
                num_steps = 0

                log_str = \
                    '{}: {}: Train loss: {:.3f} Time: {:.3f} s/iter' \
                        .format(datetime.now(), train_step, total_loss, mean_iter_time)
                self._log(log_str)
                log_stream.write(log_str + '\n')
                log_stream.flush()

                assert not np.isnan(total_loss), 'Model diverged with loss = NaN, step= {}'.format(train_step)

                summary_writer.add_summary(train_summary, train_step)

                start_time = time.time()
            else:
                _ = sess.run(train_op, feed_dict={is_training: True})
                num_steps += 1

            if train_step > 0 and train_step % self._save_iter == 0:
                checkpoint_path = join(self._log_dir, 'model.ckpt')
                model_saver.save(sess, checkpoint_path, global_step=train_step)

        log_stream.close()
        self._log('Finished.')

    def train(self, data_path):
        """Carry out model training procedure.

        :param data_path: Path to load training data
        """

        with tf.Graph().as_default():
            dataset = self._create_train_dataset(data_path)

            data_iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            data_init_op = data_iterator.make_initializer(dataset)

            is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')
            global_step = tf.Variable(0, trainable=False, name='GlobalStep')

            train_op, loss_op, towers = self._multi_gpu_training(global_step, is_training, data_iterator)
            tf.add_to_collection('loss_summary', tf.summary.scalar('total_loss', loss_op))

            summary_op = self._collect_summary()

            sess = self._start_session()
            self._restore_train_parameters(sess, towers[0])
            self._init_data(sess, data_init_op)
            self._optimize(sess, global_step, train_op, loss_op, summary_op, is_training)
            self._close_session(sess)

    def test(self, data_path):
        """Carry out model testing procedure.

        :param data_path: Path to load training data
        """

        with tf.Graph().as_default():
            dataset = self._create_test_dataset(data_path)

            data_iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            data_init_op = data_iterator.make_initializer(dataset)

            network_inputs = data_iterator.get_next()
            model = self._create_model(network_inputs)
            test_ops = self._get_test_ops(model, network_inputs)

            sess = self._start_session()
            self._restore_inference_parameters(sess, model)
            self._init_data(sess, data_init_op)
            self._test(sess, network_inputs, test_ops)
            self._close_session(sess)

    def eliminate_train_ops(self, output_path):
        """Carry out eliminating of all training nodes from the network.

        :param output_path: Path to save variables
        """

        with tf.Graph().as_default():
            input_shape = [1, self._params.image_size.h, self._params.image_size.w, self._params.image_size.c]
            input_image = tf.placeholder(tf.float32, input_shape, name='input')

            model = self._create_model([input_image], None, None, True, True)
            merge_op = model.get_merge_train_op()

            sess = self._start_session()
            self._restore_inference_parameters(sess, model)

            sess.run(merge_op, feed_dict={input_image: np.zeros(input_shape, dtype=np.float32)})

            saver = tf.train.Saver(max_to_keep=1)
            saver.save(sess, output_path, global_step=self._get_iter_id(self._snapshot_path))
            self._log('Model converted successfully.')

            self._close_session(sess)

    def save_model_graph(self, input_snapshot_path, output_directory_path):
        """Stores model graph to file.

        :param input_snapshot_path: Path to load variables
        :param output_directory_path: Path to save variables
        """

        with tf.Graph().as_default():
            input_shape = [1, self._params.image_size.h, self._params.image_size.w, self._params.image_size.c]
            input_image = tf.placeholder(tf.float32, input_shape, name='input')

            model = self._create_model([input_image], None, None, True, False)

            sess = self._start_session()
            self._restore_inference_parameters(sess, model, input_snapshot_path)

            tf.train.write_graph(sess.graph.as_graph_def(), output_directory_path, PB_FILE_NAME, as_text=True)
            self._log('Network graph is stored successfully.')

            self._close_session(sess)

    def freeze_model_graph(self, checkpoint_path, network_path, out_path):
        """Prepares model for inference and freezes it.

        :param checkpoint_path: Path to model checkpoint
        :param network_path: Path to network
        :param out_path: path to save out model
        """

        out_node_names = ','.join(self._get_out_node_names())
        freeze_graph.freeze_graph(network_path, '', False, checkpoint_path, out_node_names,
                                  'save/restore_all', 'save/Const:0', out_path, True, '')

    def demo(self, data_path, out_scale=1.0, deploy=False):
        """Outputs processed bt network image.

        :param data_path: Path to data
        :param out_scale: Parameter to scale output image
        :param deploy: Whether to run network in deploy mode
        """

        with tf.Graph().as_default():
            image_shape = [self._params.image_size.h, self._params.image_size.w, self._params.image_size.c]
            input_image = tf.placeholder(tf.float32, image_shape)
            network_inputs = [tf.expand_dims(input_image, axis=0)]

            model = self._create_model(network_inputs, None, None, deploy, False)
            inference_ops = self._get_test_ops(model, network_inputs)

            sess = self._start_session()
            self._restore_inference_parameters(sess, model)
            self._demo(sess, data_path, input_image, inference_ops, out_scale)
            self._close_session(sess)

    def performance(self):
        """Measures model properties like number of parameters and operations.
        """

        graph = tf.Graph()
        with graph.as_default():
            input_shape = [1, self._params.image_size.h, self._params.image_size.w, self._params.image_size.c]
            input_image = tf.placeholder(tf.float32, input_shape, name='input')

            _ = self._create_model([input_image], None, None, True, False)

            flops = \
                tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
            num_params = \
                tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())

            self._log('\nTotal stat:')
            if num_params is not None:
                self._log('MParams: {}'.format(1e-6 * float(num_params.total_parameters)))  # pylint: disable=no-member
            if flops is not None:
                self._log('GFlop: {}'.format(1e-9 * flops.total_float_ops))  # pylint: disable=no-member
