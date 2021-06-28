#!/usr/bin/env python
# coding:utf8

import codecs
import numpy as np
import tensorflow as tf
import util
#from ipdb import set_trace
from bert_base.bert import optimization
from tensorflow_probability import distributions as tfp
class ModelHelper(object):
    #VALID_MODEL_TYPE = ["ItemGoal", "ItemGoalFinetune", "CNNXDFXDF", "TransDfmAtt", "AttDfmAtt", "ItemGoalEmb"]
    """Define some common parameters and layers for model
    """
    def __init__(self, config):
        self.config = config
        self.logger = util.Logger(config)
    


    @staticmethod
    def _build_extreme_FM_quick(scope, nn_input, config, field_num, dim, out_size, layer_list = [100,100,50]):
        with tf.variable_scope(scope):
            hidden_nn_layers = []
            field_nums = []
            final_len = 0
            field_num = field_num
            nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), dim])
            field_nums.append(int(field_num))
            hidden_nn_layers.append(nn_input)
            final_result = []
            split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
            layer_list = [100,100,50]
            with tf.variable_scope("exfm_part", initializer=tf.contrib.layers.xavier_initializer(seed=config.data.shuffle_seed)) as scope:
                for idx, layer_size in enumerate(layer_list):
                    split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)
                    dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
                    dot_result_o = tf.reshape(dot_result_m, shape=[dim, -1, field_nums[0] * field_nums[-1]])
                    dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

                    filters = tf.get_variable(name="f_" + str(idx),
                                            shape=[1, field_nums[-1]*field_nums[0], layer_size],
                                            dtype=tf.float32)
                    # dot_result = tf.transpose(dot_result, perm=[0, 2, 1])
                    curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')

                    curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

                    if idx != len(layer_list) - 1:
                        next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                        final_len += int(layer_size / 2)
                    else:
                        direct_connect = curr_out
                        next_hidden = 0
                        final_len += layer_size
                    field_nums.append(int(layer_size / 2))

                    final_result.append(direct_connect)
                    hidden_nn_layers.append(next_hidden)

                result = tf.concat(final_result, axis=1)
                result = tf.reduce_sum(result, -1)

                w_nn_output = tf.get_variable(name='w_nn_output',
                                                shape=[final_len, out_size],
                                                dtype=tf.float32)
                b_nn_output = tf.get_variable(name='b_nn_output',
                                                shape=[out_size],
                                                dtype=tf.float32,
                                                initializer=tf.zeros_initializer())
                exFM_out = tf.nn.xw_plus_b(result, w_nn_output, b_nn_output)

                return exFM_out



    @staticmethod
    def _build_dnn(scope, embed_out, embed_layer_size, config, out_size, mode, layer_list=[400,400], activation_fn=tf.nn.relu):
        with tf.variable_scope(scope):
            w_fm_nn_input = embed_out
            last_layer_size = embed_layer_size
            hidden_nn_layers = []
            hidden_nn_layers.append(w_fm_nn_input)
            layer_idx = 0
            for layer_size in layer_list:
                curr_w_nn_layer = tf.get_variable(name=('w_nn_layer' + str(layer_idx)), shape=[last_layer_size, layer_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=config.data.shuffle_seed))
                curr_b_nn_layer = tf.get_variable(name=('b_nn_layer' + str(layer_idx)), shape=[layer_size], dtype=tf.float32, initializer=tf.zeros_initializer())
                curr_hidden_nn_layer = tf.nn.xw_plus_b(hidden_nn_layers[layer_idx], curr_w_nn_layer, curr_b_nn_layer)
                ####论文中使用identity优于relu
                if activation_fn is not None:
                    curr_hidden_nn_layer = activation_fn(curr_hidden_nn_layer) 
                    curr_hidden_nn_layer = tf.layers.batch_normalization(curr_hidden_nn_layer, training=(mode == tf.estimator.ModeKeys.TRAIN), name="W_bn-%s" % (layer_idx))
                hidden_nn_layers.append(curr_hidden_nn_layer)
                last_layer_size = layer_size
                layer_idx += 1

            w_nn_output = tf.get_variable(name='w_nn_output_dnn', shape=[last_layer_size, out_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=config.data.shuffle_seed))
            b_nn_output = tf.get_variable(name='b_nn_output_dnn', shape=[out_size], dtype=tf.float32, initializer=tf.zeros_initializer())
            nn_output = tf.nn.xw_plus_b(hidden_nn_layers[-1], w_nn_output, b_nn_output)
            return nn_output

    


    @staticmethod
    def attention(name, embedding, attention_dimension, config=None):
        #### attention reference from HAN
        attention_matrix = tf.contrib.layers.fully_connected(inputs=embedding, num_outputs=attention_dimension, activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(seed=config.data.shuffle_seed))
        attention_vector = tf.get_variable(name=name+"_attention_vector", shape=[attention_dimension], initializer=tf.contrib.layers.xavier_initializer(seed=(config.data.shuffle_seed)))
        alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(attention_matrix, attention_vector), axis=-1, keepdims=True), axis=1)
        attention_embedding = tf.reduce_sum(tf.multiply(embedding, alpha), axis=-2, keepdims=False)
        return attention_embedding


    

    @staticmethod
    def dropout(layer, dropout_keep_prob, is_train=True, seed=None):
        if is_train:
            layer = tf.nn.dropout(layer, keep_prob=dropout_keep_prob, seed=seed)
        return layer



    @staticmethod
    def _get_exports_dict(probability, pred):
        key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        exports = {
            key: tf.estimator.export.ClassificationOutput(scores=probability),
            "predictions": tf.estimator.export.ClassificationOutput(scores=probability),
            "pred": tf.estimator.export.PredictOutput(outputs=pred)
        }
        return exports

    def get_train_op(self, loss, learning_rate, optimizer="Adam"):
        def _get_train_op_func(variables):
            return tf.contrib.layers.optimize_loss(
                loss, global_step=tf.train.get_global_step(),
                learning_rate=learning_rate, optimizer=optimizer,
                clip_gradients=self.config.train.clip_gradients,
                variables=variables)
        variables = tf.trainable_variables()
        tf.logging.info("get_train_op: variables len %d," % len(variables))
        return _get_train_op_func(variables)





    def get_regression_estimator_spec(self, hidden_layer, mode, labels, params, minimum_value, maximum_value):
        loss, train_op = None, None
        metrics = {}
        hidden_layer_weight = tf.get_variable(
            name="hidden_layer_weight", shape=[hidden_layer.shape[1], 1],
            initializer=tf.contrib.layers.xavier_initializer(seed=self.config.data.shuffle_seed))
        hidden_layer_bias = tf.Variable(tf.zeros([1]), name="hidden_layer_bias")
        _y = tf.matmul(hidden_layer, hidden_layer_weight) + hidden_layer_bias
        if None != minimum_value:
            _y = tf.maximum(_y, minimum_value)
        if None != maximum_value:
            _y = tf.minimum(_y, maximum_value)
        if mode != tf.estimator.ModeKeys.PREDICT:
            loss = tf.losses.mean_squared_error(labels, _y)
            if self.config.train.l2_lambda > 0:
                l2_losses = tf.add_n(
                        [tf.nn.l2_loss(v) for v in tf.trainable_variables() 
                            if 'bias' not in v.name.lower() and 
                            'global_step' not in v.name.lower() and 
                            'bert' not in v.name.lower()]) * self.config.train.l2_lambda
                loss = loss + l2_losses
            if abs(self.config.train.decay_rate - 1) > util.EPS:
                learning_rate = tf.train.exponential_decay(
                    self.config.train.learning_rate, tf.train.get_global_step(),
                    self.config.train.decay_steps, self.config.train.decay_rate,
                    staircase=True, name="learning_rate")
            else:
                learning_rate = self.config.train.learning_rate

            if mode == tf.estimator.ModeKeys.TRAIN:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = self.get_train_op(loss, learning_rate, optimizer=self.config.optimizer.optimizer)
            metrics = {
                "MAE": tf.metrics.mean_absolute_error(labels, _y),
                "MSE": tf.metrics.mean_squared_error(labels, _y)
            }
        exports = self._get_exports_dict(_y, hidden_layer)
        return tf.estimator.EstimatorSpec(mode, predictions=_y, loss=loss, train_op=train_op, eval_metric_ops=metrics, export_outputs=exports)








