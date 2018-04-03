
import tensorflow as tf

slim = tf.contrib.slim

def conv_block(inputs, kernel_size, filters, stage, block,
               stride=(2, 2), training=True):
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


def identity_block(inputs, kernel_size, filters, stage, block,
                training):
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

  x = slim.conv2d(inputs,nb_filter1,[1,1],stride=[1,1],padding='VALID',activation_fn=None,
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

  
  x = tf.add(x,inputs)
  x = tf.nn.relu(x,name='res'+str(stage)+block+"_out")
  return x


def resnet(inputs,architecture,stage5=True,num_classes=None,training=False):

  assert architecture in ["resnet50", "resnet101"]

  end_points = {}
  #stage1
  x = tf.keras.layers.ZeroPadding2D((3,3))(inputs)
  x = slim.conv2d(x,64,[7,7],stride=[2,2],activation_fn=None,trainable=training,scope='conv1')
  x = slim.batch_norm(x,scope='conv1_bn',is_training=training)
  x = tf.nn.relu(x)
  C1 = x = slim.max_pool2d(x,(3, 3), stride=[2, 2], padding="same",scope='pool1')

  #stage2
  x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', stride=[1, 1], training=training)
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', training=training)
  C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', training=training)

  # Stage 3
  x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', training=training)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', training=training)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', training=training)
  C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', training=training)

  # Stage 4
  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', training=training)
  block_count = {"resnet50": 5, "resnet101": 22}[architecture]
  for i in range(block_count):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), training=training)
  C4 = x
  # Stage 5
  if stage5:
      x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', training=training)
      x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', training=training)
      C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', training=training)
  else:
      C5 = None

  #only for resnet
  x = tf.reduce_mean(x, [1, 2], name='pool5', keep_dims=True)
  end_points['global_pool'] = x 

  if num_classes:
    x = slim.conv2d(x,num_classes,[1,1],[1,1],activation_fn=None,
                    normalizer_fn=None,scope='logits')
    x = tf.squeeze(x, [1, 2], name='SpatialSqueeze')
    end_points['spatial_squeeze'] = x
    end_points['logits'] = x
    end_points['predictions'] = slim.softmax(x, scope='predictions')

  return [C1, C2, C3, C4, C5],end_points
    
