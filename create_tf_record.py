#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os
import logging
import random
import io
import PIL.Image
import time



flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'input folder')
flags.DEFINE_string('log_file', 'mylog.log', 'input folder')
flags.DEFINE_string('dataset_output_dir', '', 'folder saving dataset file')
flags.DEFINE_boolean('ifVal', 'True', 'if generate validation dataset')
FLAGS = flags.FLAGS

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=FLAGS.log_file,
                filemode='w')


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def dict_to_tf_example(img,file_name):
  with tf.gfile.GFile(img,'rb') as fid:
    encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    image_array = np.asarray(image)

    if image.format != 'JPEG':
      raise ValueError('Image format not JPEG')

    height,width,depth = image_array.shape
  feature_dict = {
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/filename': bytes_feature(
          file_name.encode('utf8')),
      'image/encoded': bytes_feature(encoded_jpg),
      'image/format': bytes_feature('jpeg'.encode('utf8'))    
      }
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return(example)
  
def create_tf_record(output_filename,examples):
  writer = tf.python_io.TFRecordWriter(output_filename)
  #start_t = time.time()
  #print(start_t,'----start')
  for idx,example in enumerate(examples):
    example_name = os.path.basename(example).replace(".jpg","")
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples))

   
    try:
      tf_example = dict_to_tf_example(
          img=example,
          file_name=example_name)
      writer.write(tf_example.SerializeToString())
    except ValueError:
      logging.warning('Invalid example: %s, ignoring.', example)

  #end_t = time.time()
  #print(end_t,'---end',end_t-start_t,"\n")
  writer.close()


def main(_):
  assert FLAGS.data_dir, 'data folder is missing'
  assert FLAGS.dataset_output_dir, 'dataset file folder is missing'

  data_dir = FLAGS.data_dir
  logging.info('Reading from folder %s' % data_dir)

  example_files = tf.gfile.Glob(os.path.join(data_dir,"*.jpg"))

  random.seed(45)
  random.shuffle(example_files)

  ifVal = FLAGS.ifVal

  train_examples = example_files

  if ifVal:
    num_example = len(example_files)
    num_train = int(0.7*num_example)
    train_examples = example_files[:num_train]
    val_examples = example_files[num_train:]
    logging.info('%d training and %d validation examples.', 
                len(train_examples), len(val_examples))
 
  train_out_path = os.path.join(FLAGS.dataset_output_dir,
                                     'train.record')
  create_tf_record(train_out_path,
                  train_examples)
  
  if ifVal:
    val_out_path = os.path.join(FLAGS.dataset_output_dir,
                                     'val.record')
    create_tf_record(val_out_path,
                    val_examples)


if __name__ == '__main__':
  tf.app.run()
