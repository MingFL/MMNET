#!/usr/bin/env python
# coding:utf8

import tensorflow as tf
from model.embedding_layer import EmbeddingLayer
from model.model_helper import ModelHelper
from bert_base.bert import modeling
from model.tcn import TemporalConvNet
import numpy as np


class MMNetEstimator(tf.estimator.Estimator):
    def __init__(self, data_processor, model_params):
        self.seed = None 
        config = data_processor.config
        embedding_layer = EmbeddingLayer(config)
        model_helper = ModelHelper(config)

        def _model_fn(features, labels, mode, params):
            padding_map = { 
                            "title_token": [config.fixed_len_feature.title_token_padding_begin, 
                                            config.fixed_len_feature.title_token_padding_end],
                            "title_char": [config.fixed_len_feature.title_char_padding_begin, 
                                            config.fixed_len_feature.title_char_padding_end],
                            "content_token": [config.fixed_len_feature.content_token_padding_begin, 
                                            config.fixed_len_feature.content_token_padding_end],
                            "content_char": [config.fixed_len_feature.content_char_padding_begin, 
                                            config.fixed_len_feature.content_char_padding_end],
                          }
            hidden_layer_list = []
            dense_dim = config.embedding_layer.dense_embedding_dimension
            for feature_name in params["feature_names"]:
                if feature_name not in ['title_token', 'title_char', 'content_token', 'content_char']:
                    continue
                with tf.variable_scope(feature_name):
                    if config.bert.use_bert and feature_name in ['title_char', 'content_char']:
                        bert_config = modeling.BertConfig.from_json_file(config.bert.bert_config)
                        bert_model = modeling.BertModel(
                                config=bert_config,
                                is_training=(mode == tf.estimator.ModeKeys.TRAIN),
                                input_ids=features["fixed_len_" + feature_name],
                                input_mask=features["fixed_len_mask_" + feature_name],
                                token_type_ids=None,
                                use_one_hot_embeddings=False, 
                                scope="bert")

                        init_checkpoint = config.bert.checkpoint
                        tvars = tf.trainable_variables()
                        tvars = [v for v in tvars if 'bert' in v.name and feature_name in v.name]
                        (assignment_map, initialized_variable_names) = \
                                modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint, var_scope=feature_name)
                        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

                        text_hidden_layer = bert_model.get_pooled_output()
                        text_hidden_layer = tf.contrib.layers.fully_connected(
                                inputs=text_hidden_layer, 
                                num_outputs=dense_dim, 
                                activation_fn=None, 
                                weights_initializer=\
                                        tf.contrib.layers.xavier_initializer(seed=self.seed))
                        text_hidden_layer = tf.nn.relu(text_hidden_layer)
                    else:
                        index = data_processor.dict_names.index(feature_name)
                        padding_value = data_processor.dict_list[index][data_processor.VOCAB_PADDING]
                        text_embedding = embedding_layer.get_vocab_embedding(
                                feature_name, 
                                features["fixed_len_" + feature_name],
                                len(data_processor.dict_list[index]), 
                                config.embedding_layer.text_feature_embedding_dimension,
                                pretrained_embedding_file = data_processor.pretrained_embedding_files[index],
                                mode = mode, dict_map = data_processor.dict_list[index],
                                begin_padding_size = padding_map[feature_name][0], 
                                end_padding_size = padding_map[feature_name][1],
                                padding_id = padding_value)

                        last_hidden_size = text_embedding.get_shape().as_list()[-1]
                        text_hidden_layer = model_helper.attention(feature_name + '_att', text_embedding, last_hidden_size, config)
                        text_hidden_layer = tf.contrib.layers.fully_connected(
                                inputs=text_hidden_layer, 
                                num_outputs=dense_dim, 
                                activation_fn=None, 
                                weights_initializer=\
                                tf.contrib.layers.xavier_initializer(seed=self.seed))
                        text_hidden_layer = tf.nn.elu(text_hidden_layer)
                        text_hidden_layer = tf.layers.batch_normalization(text_hidden_layer, training=(mode == tf.estimator.ModeKeys.TRAIN), name="W_bn-%s" % (feature_name))
                    hidden_layer_list.append(text_hidden_layer)
            
            if len(hidden_layer_list) > 1:
                hidden_layer_list = map(lambda x: tf.expand_dims(x, 1), hidden_layer_list)
                text_concat_embedding = tf.concat(values=hidden_layer_list, axis=1)
                text_mean = model_helper.attention("text_att_merge", text_concat_embedding, dense_dim, config)
                hidden_layer_list = [text_mean]


            for feature_name in params["feature_names"]:
                if feature_name not in ['video_embedding']:
                    continue
                with tf.variable_scope(feature_name):
                    index = data_processor.dict_names.index(feature_name)
                    video_embedding = embedding_layer.get_vocab_embedding(
                                feature_name,
                                features["fixed_len_" + feature_name],
                                len(data_processor.dict_list[index]),
                                config.embedding_layer.video_embedding_dimension,
                                pretrained_embedding_file = data_processor.pretrained_embedding_files[index],
                                mode = mode, dict_map = data_processor.dict_list[index])

                    last_hidden_size = video_embedding.get_shape().as_list()[-1]
                    video_embedding = tf.reshape(video_embedding, shape=[-1, last_hidden_size])
                    video_hidden_layer = tf.contrib.layers.fully_connected(
                                inputs=video_embedding,
                                num_outputs=64,
                                activation_fn=None,
                                weights_initializer=\
                                        tf.contrib.layers.xavier_initializer(seed=config.data.shuffle_seed))
                    video_hidden_layer = tf.nn.relu(video_hidden_layer)
                hidden_layer_list.append(video_hidden_layer)


            discrete_feature_embedding_list = []
            discrete_feature_size = 0
            for feature_name in params["feature_names"]:
                if feature_name not in ['discrete_feature']:
                    continue
                with tf.variable_scope(feature_name):
                    discrete_feature_embedding_dimension = config.embedding_layer.discrete_feature_embedding_dimension
                    discrete_feature_size = config.fixed_len_feature.fixed_len_discrete_feature_length
                    index = data_processor.dict_names.index(feature_name)
                    discrete_feature_embedding = embedding_layer.get_vocab_embedding(
                            feature_name, 
                            features["fixed_len_" + feature_name],
                            len(data_processor.dict_list[index]), 
                            discrete_feature_embedding_dimension,
                            pretrained_embedding_file = data_processor.pretrained_embedding_files[index],
                            mode = mode, 
                            dict_map = data_processor.dict_list[index])
                    discrete_feature_embedding = tf.reshape(discrete_feature_embedding, shape=[-1, discrete_feature_size * discrete_feature_embedding_dimension])
                    discrete_feature_embedding_list.append(discrete_feature_embedding)


            discrete_group_feature_size = 0
            for feature_name in params["feature_names"]:
                if feature_name not in ['discrete_group_feature']:
                    continue
                discrete_feature_embedding_dimension = config.embedding_layer.discrete_feature_embedding_dimension                
                with tf.variable_scope(feature_name):                    
                    discrete_group_feature_size = config.fixed_len_feature.fixed_len_group_discrete_feature_length
                    index = data_processor.dict_names.index(feature_name)
                    for i in range(discrete_group_feature_size):
                        discrete_group_feature_embedding = embedding_layer.get_vocab_embedding(
                                feature_name + str(i), 
                                features["fixed_len_" + feature_name + str(i)],
                                len(data_processor.dict_list[index]), 
                                discrete_feature_embedding_dimension,
                                pretrained_embedding_file = data_processor.pretrained_embedding_files[index],
                                mode = mode, 
                                dict_map = data_processor.dict_list[index])
                        #discrete_group_feature_embedding = tf.reduce_mean(discrete_group_feature_embedding, axis=1)
                        discrete_group_feature_embedding = tf.reduce_max(discrete_group_feature_embedding, axis=1)
                        discrete_feature_embedding_list.append(discrete_group_feature_embedding)


            float_feature_size = 0
            for feature_name in params["feature_names"]:
                if feature_name not in ['float_feature']:
                    continue
                float_feature_embedding_dimension = config.embedding_layer.float_feature_embedding_dimension
                float_feature_size = config.fixed_len_feature.fixed_len_float_feature_length
                float_feature_score_W = tf.get_variable(
                        name="fixed_len_" + feature_name, 
                        shape=[float_feature_size, float_feature_embedding_dimension], 
                        initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
                float_feature_score_emb = tf.multiply(tf.expand_dims(features["fixed_len_" + feature_name], -1), float_feature_score_W)
                #### 如果float特征放在xdfm中的话，需要discrete_feature_embedding = float_feature_embedding_dimension
                float_feature_score_emb = tf.reshape(float_feature_score_emb, shape=[-1, float_feature_size * float_feature_embedding_dimension])
                discrete_feature_embedding_list.append(float_feature_score_emb)

            if len(discrete_feature_embedding_list) > 0:
                feature_name = 'discrete_feature'
                discrete_feature_embedding = tf.concat(discrete_feature_embedding_list, axis=1)
                ####xdfm
                discrete_feature_embedding_dimension = config.embedding_layer.discrete_feature_embedding_dimension 
                logit = model_helper._build_linear(
                        feature_name + '_linear', 
                        discrete_feature_embedding, 
                        config, 
                        dense_dim)
                logit = tf.add(logit, 
                        model_helper._build_extreme_FM_quick(
                            feature_name + '_CIN', 
                            discrete_feature_embedding, 
                            config, 
                            discrete_feature_size + discrete_group_feature_size+float_feature_size, 
                            discrete_feature_embedding_dimension, 
                            dense_dim,
                            [discrete_feature_embedding_dimension * 4, discrete_feature_embedding_dimension * 4, discrete_feature_embedding_dimension * 2]))
                logit = tf.add(logit, 
                        model_helper._build_dnn(
                            feature_name + '_dnn', 
                            discrete_feature_embedding, 
                            discrete_feature_embedding_dimension*(discrete_feature_size+discrete_group_feature_size+float_feature_size), 
                            config,
                            dense_dim,
                            mode,
                            [discrete_feature_embedding_dimension * 8, discrete_feature_embedding_dimension * 8, discrete_feature_embedding_dimension * 4],
                            tf.nn.elu
                            ))
                concat_input = logit
                hidden_layer_list.append(concat_input)

            for feature_name in params["feature_names"]:
                if feature_name not in ['sequence_feature']:
                    continue
                sequence_feature_embedding_dimension = config.embedding_layer.sequence_feature_embedding_dimension
                sequence_attr_feature_length = config.fixed_len_feature.fixed_len_sequence_attr_feature_length / 2
                sequence_id_feature_length = config.fixed_len_feature.fixed_len_sequence_id_feature_length / 2
                sequence_attr_window = sequence_id_feature_length
                sequence_id_window = sequence_id_feature_length
                sequence_feature_length = config.fixed_len_feature.fixed_len_sequence_feature_length / 2


                #### miss flag emb
                fixed_len_is_miss_seqence = tf.gather(features["fixed_len_" + feature_name], range(0, sequence_feature_length * 2, 2), axis=1)
                fixed_len_is_miss_seqence = tf.cast(fixed_len_is_miss_seqence, tf.int32)
                sequence_attrs_miss_flag, sequence_id_miss_flag = tf.split(fixed_len_is_miss_seqence, [sequence_attr_feature_length, sequence_attr_window], axis=1)
                fixed_len_is_miss_seqence_lookup_table = tf.get_variable(
                    name="fixed_len_is_miss_seqence_lookup_table", 
                    shape=[2, sequence_feature_embedding_dimension], ### flag只有0,1
                    initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
                #### [batch, sequence_feature_length, emb_size]
                sequence_id_miss_flag_emb = tf.nn.embedding_lookup(fixed_len_is_miss_seqence_lookup_table, sequence_id_miss_flag)
                sequence_id_miss_flag_emb = tf.reshape(sequence_id_miss_flag_emb, shape=[-1, 1, sequence_id_window, sequence_feature_embedding_dimension])
                sequence_id_miss_flag_emb_normal =sequence_id_miss_flag_emb 
                
                fixed_len_seqence = tf.gather(features["fixed_len_" + feature_name], range(1, sequence_feature_length * 2, 2), axis=1)
                fixed_len_sequence_attrs, fixed_len_sequence_id = tf.split(fixed_len_seqence, [sequence_attr_feature_length, sequence_id_feature_length], axis=1)
                sequence_feature_attrs_W = tf.get_variable(
                        name="fixed_len_sequence_attrs",
                        shape=[sequence_attr_feature_length,
                                sequence_feature_embedding_dimension],
                        initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
                sequence_feature_id_W = tf.get_variable(
                        name="sequence_feature_id_W",
                        shape=[sequence_id_feature_length,
                                sequence_feature_embedding_dimension],
                        initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))

                sequence_feature_attrs_emb = tf.multiply(tf.expand_dims(fixed_len_sequence_attrs, -1), sequence_feature_attrs_W)
                sequence_feature_id_emb = tf.multiply(tf.expand_dims(fixed_len_sequence_id, -1), sequence_feature_id_W)
            
                sequence_feature_attrs_emb = tf.reshape(sequence_feature_attrs_emb, shape=[-1, sequence_attr_feature_length / sequence_attr_window, sequence_attr_window, sequence_feature_embedding_dimension])
                sequence_feature_id_emb = tf.reshape(sequence_feature_id_emb, shape=[-1, 1, sequence_id_window, sequence_feature_embedding_dimension])
                sequence_feature_id_emb_normal = sequence_feature_id_emb
                tmp_batch = tf.shape(sequence_feature_id_emb_normal)[0]
                if mode == tf.estimator.ModeKeys.TRAIN:
                    thres = 0.5
                    fea_id = tf.ones(shape=[tmp_batch*sequence_id_window])
                    condition = tf.random.uniform(shape=[tmp_batch*sequence_id_window], minval=0, maxval=1, seed=self.seed,dtype=tf.float32)
                    miss_id_one = tf.ones(shape=[tmp_batch*sequence_id_window])
                    miss_id_zero = tf.zeros(shape=[tmp_batch*sequence_id_window])
                    miss_id = tf.where(condition>thres,miss_id_zero,miss_id_one)
                    miss_id = tf.reshape(miss_id, shape=[-1, sequence_id_window])
                    fixed_len_att_miss, fixed_len_id_miss = tf.split(fixed_len_is_miss_seqence,[sequence_attr_feature_length,sequence_attr_window], axis=1)
                    fixed_len_id_miss = tf.cast(fixed_len_id_miss, tf.float32)
                    miss_id = tf.cast(miss_id, tf.float32)
                    miss_id = tf.add(miss_id,fixed_len_id_miss)
                    miss_id = tf.clip_by_value(miss_id, 0, 1)
                    sequence_id_miss_flag_one_random_emb = tf.nn.embedding_lookup(fixed_len_is_miss_seqence_lookup_table, tf.cast(miss_id, tf.int32))
                    sequence_id_miss_flag_emb = tf.reshape(sequence_id_miss_flag_one_random_emb,
                                                               shape=[-1, 1, sequence_id_window,
                                                                      sequence_feature_embedding_dimension])

                    fea_id = tf.reshape(fea_id, shape=[-1, sequence_id_window])
                    fea_id = tf.cast(fea_id, tf.float32)
                    fea_id = tf.subtract(fea_id, miss_id)
                    fea_id = fea_id * fixed_len_sequence_id
                    sequence_feature_id_random_emb = tf.multiply(tf.expand_dims(fea_id, -1),
                                                                sequence_feature_id_W)
                    sequence_feature_id_emb = tf.reshape(sequence_feature_id_random_emb,
                               shape=[-1, 1, sequence_id_window, sequence_feature_embedding_dimension])		    


                    condition = tf.random.uniform(shape=[tmp_batch], minval=0, maxval=1, seed=self.seed)

                    fixed_len_sequence_id_zeros = tf.zeros(shape=[tmp_batch, sequence_id_feature_length])
                    sequence_feature_id_zeros_emb = tf.multiply(tf.expand_dims(fixed_len_sequence_id_zeros, -1), sequence_feature_id_W)
                    sequence_id_zeros = tf.reshape(sequence_feature_id_zeros_emb, shape=[-1, 1, sequence_id_window, sequence_feature_embedding_dimension])                    
                    sequence_feature_id_emb = tf.where(condition < config.model_common.dr_ratio, sequence_id_zeros, sequence_feature_id_emb)
                    sequence_feature_id_emb = tf.where((condition < config.model_common.dr_ratio2) & (condition >= config.model_common.dr_ratio),  sequence_feature_id_emb_normal, sequence_feature_id_emb)
                    sequence_id_miss_flag_one = tf.ones(shape=[tmp_batch, sequence_id_feature_length], dtype=tf.int32)
                    sequence_id_miss_flag_one_emb = tf.nn.embedding_lookup(fixed_len_is_miss_seqence_lookup_table, sequence_id_miss_flag_one)
                    sequence_id_miss_flag_one_emb = tf.reshape(sequence_id_miss_flag_one_emb, shape=[-1, 1, sequence_id_window, sequence_feature_embedding_dimension])
                    sequence_id_miss_flag_emb = tf.where(condition < config.model_common.dr_ratio, sequence_id_miss_flag_one_emb, sequence_id_miss_flag_emb)
                    sequence_id_miss_flag_emb = tf.where((condition < config.model_common.dr_ratio2) & (condition >= config.model_common.dr_ratio), sequence_id_miss_flag_emb_normal, sequence_id_miss_flag_emb)
                    

                sequence_attr_id = tf.add(sequence_id_miss_flag_emb,sequence_feature_id_emb)
                sequence_attr_id =tf.concat([sequence_attr_id,sequence_feature_attrs_emb],axis=1)
                sequence_attr_id = tf.reshape(sequence_attr_id, 
                    shape=[
                        -1, 
                        sequence_attr_feature_length / sequence_id_feature_length + 1,
                        sequence_id_feature_length,
                        sequence_feature_embedding_dimension
                    ])
                sequence_attr_id = tf.transpose(sequence_attr_id, [0,2,1,3])

                sequence_attr_id = model_helper.attention(feature_name + '_attr_id_att', sequence_attr_id, sequence_feature_embedding_dimension, config)

                channel_sizes = config.model_common.TCN_channels
                if channel_sizes[-1] != dense_dim:
                    channel_sizes.append(dense_dim)
                kernel_size=config.model_common.TCN_kernel_size
                dropout=tf.constant(0.0, dtype=tf.float32)
                sequence_len = sequence_id_feature_length
                
                sequence_attr_id= TemporalConvNet(input_layer=sequence_attr_id, num_channels=channel_sizes, sequence_length=sequence_len,
                    kernel_size=kernel_size, dropout=dropout, init=False)

                sequence_feature_emb = model_helper.attention(feature_name  + '_tcn_att', sequence_attr_id, dense_dim, config)
                hidden_layer_list.append(sequence_feature_emb)

            hidden_layer = hidden_layer_list[0]
            if len(hidden_layer_list) > 1:
                hidden_layer_list = map(lambda x: tf.expand_dims(x, 1), hidden_layer_list)
                hidden_layer_list = tf.concat(values=hidden_layer_list, axis=1)
                hidden_layer = model_helper.attention("all_channel_merge", hidden_layer_list, dense_dim, config)
                
            if config.task.task_type == "regression":
                regression_minimum_value = None
                regression_maximum_value = None
                if hasattr(config.task, 'regression_minimum_value'):
                    regression_minimum_value = config.task.regression_minimum_value
                if hasattr(config.task, 'regression_maximum_value'):
                    regression_maximum_value = config.task.regression_maximum_value
                estimator = model_helper.get_regression_estimator_spec(hidden_layer, mode, labels, params, regression_minimum_value, regression_maximum_value)
            return estimator

        super(MMNetEstimator, self).__init__(
            model_fn=_model_fn, model_dir=config.model_common.checkpoint_dir,
            config=model_helper.get_run_config(), params=model_params)
