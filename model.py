from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

try:
  xrange
except:
  xrange = range

class SRCNN(object):

  def __init__(self, 
               sess, 
               image_size=33,
               label_size=21, 
               batch_size=128,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    with tf.name_scope('Inputs'):
      self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
    with tf.name_scope('Labels'):
      self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
  
    with tf.name_scope('Weights'):
      self.weights = {
        'w1': tf.Variable(tf.random_normal([9, 9, self.c_dim, 64], stddev=1e-3), name='w1'),
        'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
        'w3': tf.Variable(tf.random_normal([5, 5, 32, self.c_dim], stddev=1e-3), name='w3')
      }
    with tf.name_scope('Bias'):
      self.biases = {
        'b1': tf.Variable(tf.zeros([64]), name='b1'),
        'b2': tf.Variable(tf.zeros([32]), name='b2'),
        'b3': tf.Variable(tf.zeros([1]), name='b3')
      }

    #predivtion
    self.pred = self.model()

    # Loss function (MSE)
    self.loss = tf.reduce_mean(tf.abs(self.labels - self.pred))
    tf.summary.scalar('loss',self.loss)

    self.saver = tf.train.Saver()

  def train(self, config):

    merged = tf.summary.merge_all()


    if config.is_train:
      input_setup(self.sess, config)
    else:
      nx, ny = input_setup(self.sess, config)

    if config.is_train:     
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    else:
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

    train_data, train_label = read_data(data_dir)

    with tf.name_scope('Train'):
      # Stochastic gradient descent with the standard backpropagation
      #self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)
      self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
      

    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if config.is_train:
      print("Training...")

      train_writer = tf.summary.FileWriter('logs/train',self.sess.graph)


      for ep in xrange(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data) // config.batch_size
        for idx in xrange(0, batch_idxs):
          batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err))

          if counter % 500 == 0:
            summary = self.sess.run(merged,feed_dict={self.images: batch_images, self.labels: batch_labels})
            train_writer.add_summary(summary,counter)
            self.save(config.checkpoint_dir, counter)

    else:
      print("Testing...")

      test_writer = tf.summary.FileWriter('logs/test',self.sess.graph)

      result = self.pred.eval({self.images: train_data, self.labels: train_label})

      result = merge(result, [nx, ny])
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "srcnn.png")
      imsave(result, image_path)

  def model(self):
    with tf.name_scope('Conv1'):
      conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'])
    with tf.name_scope('Conv2'):
      conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
    with tf.name_scope('Conv3'):
      conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID') + self.biases['b3']
    return conv3

  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False











