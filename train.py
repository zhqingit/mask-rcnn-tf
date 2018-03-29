#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os
import io
import PIL.Image
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
    encoded_jpg_io = io.BytesIO(parsed_features["image/encoded"])
    image = PIL.Image.open(encoded_jpg_io)

    #return parsed_features["image/encoded"], parsed_features["image/filename"]
    return image,parsed_features["image/filename"]

  dataset = dataset.map(_parse_function)
  dataset = dataset.batch(1)

  iterator = dataset.make_initializable_iterator()
  next_item = iterator.get_next()

  #example = tf.train.Example()
  #example.ParseFromString(next_item)
  #height = int(example.features.feature['height'].int64_list.value[0])
  #width = int(example.features.feature['width'].int64_list.value[0])
  #file_name = (example.features.feature['filename'].bytes_list.value[0])
  #img = (example.features.feature['encoded'].bytes_list.value[0])

  sess = tf.Session()
  sess.run(iterator.initializer)
  for i in range(1):
    value = sess.run(next_item)
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
