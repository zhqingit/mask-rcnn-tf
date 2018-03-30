#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os
import resnet_model as resnet
#from object_detection.data_decoders import tf_example_decoder


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
  #train_files = tf.gfile.Glob(os.path.join(input_dir,"*train*"))
  val_files = tf.gfile.Glob(os.path.join(input_dir,"*val*"))
  dataset = tf.data.TFRecordDataset(val_files)

  def _parse_function(example_proto):
    
    features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
              "image/filename": tf.FixedLenFeature((), tf.string, default_value="")}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.image.decode_jpeg(parsed_features["image/encoded"])
    image_resized = tf.image.resize_images(image, [32, 32])

    #return parsed_features["image/encoded"], parsed_features["image/filename"]
    return image_resized,parsed_features["image/filename"]

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
  model = resnet.Model(resnet_size=resnet_size,
        bottleneck=False,
        num_classes=10,
        num_filters=16,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        second_pool_size=8,
        second_pool_stride=1,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        final_size=64,
        version=1,
        data_format='channels_last')
  inputs = tf.placeholder(tf.float32,[16,32,32,3])
  res = model.__call__(inputs,False)

  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(iterator.initializer)
  sess.run(init)
  for i in range(100):
    (value,name) = sess.run(next_item)
    print(value.shape,"--")
    rest = sess.run(res,feed_dict={inputs:value})
    print(rest.shape,"++")


if __name__ == '__main__':
  tf.app.run()
