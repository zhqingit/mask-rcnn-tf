#!/usr/bin/python

import tensorflow as tf


def main(_):
  assert FLAGS.config_file, 'Config file is missing'
  assert FLAGS.mode in ['train','inference'], 'mode is not correct'
  assert FLAGS.image_file, 'image file is missing'

  #Get the input data
  #Get the configuration
  #config = get_config(FLAGS.config_file)
  #Create model in training mode
  #model = modellib.MaskRCNN(mode='training',config=config,model_dir=MODEL_DIR)


if __name__ == '__main__':
  tf.app.run()
