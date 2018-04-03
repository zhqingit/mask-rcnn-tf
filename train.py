#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os
#import resnet_model as resnet
import models.resnet as resnet
#from object_detection.data_decoders import tf_example_decoder


flags = tf.app.flags
flags.DEFINE_string('input_dir', '', 'input folder')
FLAGS = flags.FLAGS
_NUM_CLASS = 5

'''
def read_dataset(input_dir):
  input_files = glob.glob(input_dir+"/*train*")
  print(input_files)
  filenames = tf.concat([tf.matching_files(pattern) for pattern in input_files],0)
  filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)

  records_dataset =  filename_dataset.apply(
      tf.contrib.data.parallel_interleave(
          lambda filename: tf.data.TFRecordDataset(filename), cycle_length=4, sloppy=True))

  decoder = tf_example_decoder.TfExampleDecoder()
  #tensor_dataset = records_dataset.map(decoder)
  #
  #
'''
def read_dataset(input_dir):
  train_files = tf.gfile.Glob(os.path.join(input_dir,"*train*"))
  #val_files = tf.gfile.Glob(os.path.join(input_dir,"*val*"))
  dataset = tf.data.TFRecordDataset(train_files)

  def _parse_function(example_proto):
    
    features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
              "image/filename": tf.FixedLenFeature((), tf.string, default_value=""),
              "image/label": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.image.decode_jpeg(parsed_features["image/encoded"])
    image = tf.image.per_image_standardization(image)
    image_resized = tf.image.resize_images(image, [32, 32])
    label = tf.one_hot(parsed_features["image/label"], _NUM_CLASS+1)

    #return parsed_features["image/encoded"], parsed_features["image/filename"]
    return image_resized,label,parsed_features["image/filename"]

  dataset = dataset.map(_parse_function)
  dataset = dataset.repeat()
  dataset = dataset.batch(16)

  iterator = dataset.make_initializable_iterator()

  #example = tf.train.Example()
  #example.ParseFromString(next_item)
  #height = int(example.features.feature['height'].int64_list.value[0])
  #width = int(example.features.feature['width'].int64_list.value[0])
  #file_name = (example.features.feature['filename'].bytes_list.value[0])
  #img = (example.features.feature['encoded'].bytes_list.value[0])
  return(iterator)


  #print(filename_dataset.get_next())
 
#def 

def main(_):
  #assert FLAGS.config_file, 'Config file is missing'
  #assert FLAGS.mode in ['train','inference'], 'mode is not correct'
  #assert FLAGS.image_file, 'image file is missing'
  assert FLAGS.input_dir, 'input folder'

  #Get the input data
  iterator = read_dataset(FLAGS.input_dir)
  next_item = iterator.get_next()


  #Create model in training mode
  resnet_size = 32
  if resnet_size % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size)
  num_blocks = (resnet_size - 2) // 6

  inputs = tf.placeholder(tf.float32,[16,32,32,3])
  y_true_batch = tf.placeholder(tf.float32,[16,6])
  (nets,end_points) = resnet.resnet(inputs,"resnet50",stage5=True,num_classes=6,training=True)
  logits = end_points["logits"]
  print(logits,"----",y_true_batch)
  #logits = model.__call__(inputs,False)
  #tf.losses.softmax_cross_entropy(y_true_batch, logits, weights=1.0, label_smoothing=0.1)
  cross_entropy_softmax = tf.losses.softmax_cross_entropy(onehot_labels = y_true_batch,
                                logits = logits, weights=1.0, label_smoothing=0)
  total_loss = tf.reduce_sum(cross_entropy_softmax, name='xentropy_mean')
  #losses = tf.losses.get_losses()
  #total_loss = tf.reduce_sum(losses, name='xentropy_mean')
  #total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
  #loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  #loss_averages_op = loss_averages.apply(losses + [total_loss])

  optimizer = tf.train.AdamOptimizer(0.00001)
  batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)

  tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  with tf.control_dependencies(batchnorm_updates):
    train_op = optimizer.minimize(total_loss, global_step=global_step)


  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(iterator.initializer)
  sess.run(init)
  for i in range(100):
    (value,label,name) = sess.run(next_item)
    print(value.shape,"--")
    ls = sess.run(total_loss,feed_dict={inputs:value,y_true_batch:label})
    print(ls,"++")


if __name__ == '__main__':
  tf.app.run()
