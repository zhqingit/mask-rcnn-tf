
import tensorflow as tf

slim = tf.contrib.slim

def conv_block(inputs, kernel_size, filters, stage, block,
               stride=(2, 2), training):
  """conv_block is the block that has a conv layer at shortcut
  # Arguments
      input_tensor: input tensor
      kernel_size: defualt 3, the kernel size of middle conv layer at main path
      filters: list of integers, the nb_filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      use_bias: Boolean. To use or not use a bias in conv layers.
      train_bn: Boolean. Train or freeze Batch Norm layres
  Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
  And the shortcut should have subsample=(2,2) as well
  """
  nb_filter1, nb_filter2, nb_filter3 = filters

  scope_name = 'conv'+str(stage)+block+'_branch'
  scope_name = 'bn'+str(stage)+block+'_branch'

  x = slim.conv2d(inputs,nb_filter1,[1,1],stride=stride,padding='VALID',activation_fn=None,
                  trainable=training,scope=scope_name+'2a')
  x = slim.batch_norm(x,scope=scope_name+'2a',is_training=training)
  x = tf.nn.relu(x)


  x = slim.conv2d(x,nb_filter2,[kernel_size,kernel_size],stride=[1,1],padding='SAME',
                  activation_fn=None,trainable=training,scope=scope_name+'2b')
  x = slim.batch_norm(x,scope=scope_name+'2b',is_training=training)
  x = tf.nn.relu(x)


  x = slim.conv2d(x,nb_filter3,[1,1],stride=[1,1],padding='VALID',
                  activation_fn=None,trainable=training,scope=scope_name+'2c')
  x = slim.batch_norm(x,scope=scope_name+'2c',is_training=training)



  shortcut = slim.conv2d(inputs,nb_filter3,[1,1],stride=stride,padding='VALID',activation_fn=None,
                  trainable=training,scope=scope_name+'1')
  x = slim.batch_norm(shortcut,scope=scope_name+'1',is_training=training)

  
  x = tf.add(x,shortcut)
  x = tf.nn.relu(x,name='res'+str(stage)+block+"_out")
  return x


class Model():
  """Base class for the resnet"""

  def __init__(self):
    pass

  def build(self,inputs,training):

    #stage1
    x = slim.conv2d(inputs,64,[7,7],stride=[2,2],activation_fn=None,trainable=training,scope='conv1')
    x = slim.batch_norm(x,scope='conv1_bn',is_training=training)
    x = tf.nn.relu(x)
    C1 = x = slim.max_pool2d(x,(3, 3), stride=(2, 2), padding="same",scope='pool1')

    #stage2


    
