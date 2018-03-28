#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os
from object_detection.data_decoders import tf_example_decoder


flags = tf.app.flags
flags.DEFINE_string('input_dir', '', 'input folder')
FLAGS = flags.FLAGS

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
'''
def read_dataset(input_dir):
  input_files = tf.gfile.Glob(os.path.join(input_dir,"*.jpg"))
  print(np.arange(len(input_files)))
  #filename_dataset = tf.data.TFRecordDataset({'name':input_files,'idx':range(len(input_files))})
  filename_dataset = tf.data.Dataset.from_tensor_slices({'name':input_files,'idx':np.arange(len(input_files))})
  print(filename_dataset.output_types)
  print(filename_dataset.output_shapes)
  iterator = filename_dataset.make_one_shot_iterator()

  sess = tf.Session()
  for i in range(100):
    value = sess.run(iterator.get_next())
    print(value,"--")

  #print(filename_dataset.get_next())
  

def main(_):
  #assert FLAGS.config_file, 'Config file is missing'
  #assert FLAGS.mode in ['train','inference'], 'mode is not correct'
  #assert FLAGS.image_file, 'image file is missing'
  assert FLAGS.input_dir, 'input folder'

  #Get the input data
  read_dataset(FLAGS.input_dir)
  #Get the configuration
  #config = get_config(FLAGS.config_file)
  #Create model in training mode
  #model = modellib.MaskRCNN(mode='training',config=config,model_dir=MODEL_DIR)


if __name__ == '__main__':
  tf.app.run()
