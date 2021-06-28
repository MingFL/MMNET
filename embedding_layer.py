#!/usr/bin/env python
# coding:utf8

import sys
import codecs
import numpy as np
import tensorflow as tf
import util
from model import model_layer

class EmbeddingLayer(object):

    def __init__(self, config):
        self.config = config
        self.logger = util.Logger(self.config)

    def get_lookup_table(self, name, vocab_size, dimension, dict_map=None, pretrained_embedding_file="", mode=tf.estimator.ModeKeys.TRAIN, is_trainable=True):
        if dict_map and pretrained_embedding_file and mode == tf.estimator.ModeKeys.TRAIN:
            if not is_trainable:
                word_embedding = np.zeros([vocab_size, dimension], dtype=float)
            else:
                word_embedding = np.random.normal(0, self.config.embedding_layer.embedding_random_stddev, [vocab_size, dimension])
            embedding_file = codecs.open(pretrained_embedding_file, "r", encoding=util.CHARSET)
            pretrained_embedding_dim = int(embedding_file.readline().split(" ")[1])
            assert (pretrained_embedding_dim == dimension)
            for line in embedding_file:
                line = line.strip("\r\n")
                embeddings = line.split(" ")
                if len(embeddings) != dimension + 1:
                    self.logger.error("Wrong embedding line: %s" % line)
                    continue
                word = embeddings[0]
                if word not in dict_map:
                    continue
                index = dict_map[word]
                vector = []
                for i in range(1, len(embeddings)):
                    vector.append(float(embeddings[i]))
                word_embedding[index] = np.array(vector)
            embedding_file.close()
            embedding_lookup_table = tf.get_variable(name=name + "_embedding_lookup_table", initial_value = tf.convert_to_tensor(word_embedding, dtype=tf.float32), trainable = is_trainable)
            self.logger.info("Load %s embedding from %s" % (name, pretrained_embedding_file))
        else:
            tf_initializer = tf.contrib.layers.xavier_initializer(seed=self.config.data.shuffle_seed)
            embedding_lookup_table = tf.get_variable(name=name + "_embedding_lookup_table",shape=[vocab_size, dimension], initializer=tf_initializer)
        return embedding_lookup_table

    def get_lookup_vid_emb_table(self, name, vocab_size,dimension, dict_map=None, pretrained_embedding_file="", mode=tf.estimator.ModeKeys.TRAIN, is_trainable=True):
        if dict_map and pretrained_embedding_file and mode == tf.estimator.ModeKeys.TRAIN:
            if not is_trainable:
                video_embedding = np.zeros([vocab_size, dimension], dtype=float)
            else:
                video_embedding = np.random.normal(0, self.config.embedding_layer.embedding_random_stddev, [vocab_size, dimension])
            embedding_file = codecs.open(pretrained_embedding_file, "r", encoding=util.CHARSET)
            pretrained_embedding_dim = len(embedding_file.readline().split("\t")[1:])
            assert (pretrained_embedding_dim == dimension)
            for line in embedding_file:
                line = line.strip("\r\n")
                embeddings = line.split("\t")
                if len(embeddings) != dimension + 1:
                    self.logger.error("Wrong embedding line: %s" % line)
                    continue
                docid_v = embeddings[0]
                if docid_v not in dict_map:
                    continue
                index = dict_map[docid_v]
                vector = embeddings[1:]
                video_embedding[index] = np.array(vector)
            embedding_file.close()
            embedding_lookup_table = tf.Variable(name=name + "_embedding_lookup_table", initial_value = tf.convert_to_tensor(video_embedding, dtype=tf.float32), trainable = is_trainable)
            self.logger.info("Load %s embedding from %s" % (name, pretrained_embedding_file))
        else:
            tf_initializer = tf.contrib.layers.xavier_initializer(seed=self.config.data.shuffle_seed)
            embedding_lookup_table = tf.get_variable(name=name + "_embedding_lookup_table",shape=[vocab_size, dimension], initializer=tf_initializer)
        return embedding_lookup_table

    def get_vocab_embedding(self, name, vocab_ids, vocab_size, embedding_dimension, mode=tf.estimator.ModeKeys.TRAIN, 
                            pretrained_embedding_file=None, dict_map=None, begin_padding_size=0, end_padding_size=0, padding_id=0, is_trainable=True):

        vocab_lookup_table = self.get_lookup_table(name, vocab_size, embedding_dimension, pretrained_embedding_file=pretrained_embedding_file, mode=mode, dict_map=dict_map, is_trainable=is_trainable)

        if begin_padding_size > 0 or end_padding_size > 0:
            shapes = vocab_ids.shape.as_list()
            if len(shapes) != 2:
                raise NotImplementedError
            padding = [[0, 0], [begin_padding_size, end_padding_size]]
            vocab_ids = tf.pad(vocab_ids, tf.constant(padding), constant_values=padding_id)


        vocab_embedding = tf.nn.embedding_lookup(vocab_lookup_table, vocab_ids)
        return vocab_embedding

    def get_video_embedding(self, name,vocab_ids,vocab_size,embedding_dimension, mode=tf.estimator.ModeKeys.TRAIN, 
                            pretrained_embedding_file=None, dict_map=None, is_trainable=True):
        video_lookup_table = self.get_lookup_vid_emb_table(name,vocab_size, embedding_dimension, pretrained_embedding_file=pretrained_embedding_file, mode=mode, dict_map=dict_map, is_trainable=is_trainable)
        video_embedding = tf.nn.embedding_lookup(video_lookup_table, vocab_ids)
        return video_embedding


    def get_context_lookup_table(self, name, dimension, shape,
                                 is_init, mode=tf.estimator.ModeKeys.TRAIN,
                                 initializer=None):

        if is_init and mode == tf.estimator.ModeKeys.TRAIN:
            self.logger.warn("Initialize %s context embedding randomly" % name)
        if not initializer:
            initializer = tf.random_uniform_initializer(- 1.0 / pow(dimension, 0.5), 1.0 / pow(dimension, 0.5), seed=self.config.data.shuffle_seed)
        context_embedding_table = tf.get_variable(name + 'ContextEmbedLookupTable', shape=shape, initializer=initializer)
        return context_embedding_table

    def _get_alignment_embedding(self, vocab_ids, region_size,
                                 sequence_length, lookup_table,
                                 unit_id_bias=None):

        region_radius = int(region_size / 2)
        aligned_seq = map(lambda i:
                          tf.slice(vocab_ids, [0, i - region_radius],
                                   [-1, region_size]),
                          range(region_radius, sequence_length - region_radius))
        aligned_seq = tf.reshape(tf.concat(list(aligned_seq), 1),
                                 [-1, sequence_length - region_radius * 2,
                                  region_size])
        if unit_id_bias is not None:
            aligned_seq = aligned_seq + unit_id_bias
        return tf.nn.embedding_lookup(lookup_table, aligned_seq)

    def get_region_embedding(self, name, vocab_ids, vocab_size, is_init,
                             sequence_length, region_size,
                             region_embedding_mode="WC",
                             mode=tf.estimator.ModeKeys.TRAIN,
                             pretrained_embedding_file="",
                             initializer=None,
                             dict_map=None):


        region_radius = int(region_size / 2)
        if region_embedding_mode == "WC":
            # get word aligned embedding
            vocab_lookup_table = self.get_lookup_table(
                name, vocab_size,
                self.config.embedding_layer.embedding_dimension, is_init,
                pretrained_embedding_file=pretrained_embedding_file,
                mode=mode, dict_map=dict_map)
            word_aligned_emb = self._get_alignment_embedding(
                vocab_ids, region_size, sequence_length, vocab_lookup_table)
            # get context embedding
            context_lookup_table = self.get_context_lookup_table(
                name, self.config.embedding_layer.embedding_dimension,
                [vocab_size, region_size,
                 self.config.embedding_layer.embedding_dimension],
                is_init, mode, initializer)
            trimmed_seq = \
                vocab_ids[:, region_radius:sequence_length - region_radius]
            context_emb = \
                tf.nn.embedding_lookup(context_lookup_table, trimmed_seq)

            projected_emb = word_aligned_emb * context_emb

            region_emb = tf.reduce_max(projected_emb, axis=2)
        elif region_embedding_mode == "CW":

            word_lookup_table = self.get_lookup_table(
                name, vocab_size,
                self.config.embedding_layer.embedding_dimension, is_init,
                pretrained_embedding_file=pretrained_embedding_file,
                mode=mode, dict_map=dict_map)
            word_emb = tf.nn.embedding_lookup(
                word_lookup_table,
                tf.slice(vocab_ids, [0, region_radius], [-1, tf.cast(
                    sequence_length - 2 * region_radius, tf.int32)]))
            word_emb = tf.expand_dims(word_emb, 2)
            # get context aligned embedding
            context_lookup_table = self.get_context_lookup_table(
                name, self.config.embedding_layer.embedding_dimension,
                [vocab_size * region_size,
                 self.config.embedding_layer.embedding_dimension],
                is_init, mode, initializer)
            unit_id_bias = \
                np.array([i * vocab_size for i in range(region_size)])
            context_aligned_emb = self._get_alignment_embedding(
                vocab_ids, region_size, sequence_length,
                context_lookup_table, unit_id_bias)
            # compute projected embedding
            projected_emb = context_aligned_emb * word_emb
            # max pooling
            region_emb = tf.reduce_max(projected_emb, axis=2)
        else:
            raise TypeError("Invalid region embedding mode: %s" %
                            region_embedding_mode)
        return region_emb

    def char_embedding_to_token(self, char_embedding, generate_type="cnn",
                                cnn_filter_size=None, cnn_num_filters=None,
                                rnn_cell_type="gru",
                                rnn_sequence_length=None,
                                rnn_cell_dimension=None,
                                rnn_cell_hidden_keep_prob=1.):

        if generate_type == "sum":
            return tf.reduce_sum(char_embedding, axis=1)
        elif generate_type == "avg":
            return tf.reduce_mean(char_embedding, axis=1)
        elif generate_type == "max":
            return tf.reduce_max(char_embedding, axis=1)
        elif generate_type == "cnn":
            char_embedding = tf.expand_dims(char_embedding, axis=-1)
            filter_shape = \
                [cnn_filter_size, char_embedding.shape[-2], 1,
                 cnn_num_filters]
            char_embedding_cnn = model_layer.convolution(
                char_embedding, filter_shape, use_bias=True,
                activation=tf.nn.relu, name="convolution")
            return tf.reduce_max(char_embedding_cnn, axis=1)
        elif generate_type == "rnn":
            _, output_states = model_layer.recurrent(
                char_embedding, rnn_cell_dimension, rnn_sequence_length,
                cell_type=rnn_cell_type,
                cell_hidden_keep_prob=rnn_cell_hidden_keep_prob,
                name="char_embedding_to_token_rnn",
                use_bidirectional=False)
            return output_states
        else:
            raise TypeError("Wrong generate type in char_embedding_to_token: " +
                            generate_type)
