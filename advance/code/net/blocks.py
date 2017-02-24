'''
building blocks of network
#http://programtalk.com/vs2/python/3069/image_captioning/utils/nn.py/
'''
from net.common import *
from tensorflow.python.training import moving_averages
from tensorflow.contrib.framework import add_model_variable


##  global varaiables ##
IS_TRAIN_PHASE = tf.placeholder(dtype=tf.bool, name='is_train_phase')



## tools ## -------------------------------
# http://stackoverflow.com/questions/36883949/in-tensorflow-get-the-names-of-all-the-tensors-in-a-graph
# https://www.tensorflow.org/how_tos/tool_developers/
# https://medium.com/@Arafat./ruby-tensorflow-for-developers-2ec56b8668c5#.c46vp4yle
# http://stackoverflow.com/questions/35351760/tf-save-restore-graph-fails-at-tf-graphdef-parsefromstring
# https://github.com/tensorflow/tensorflow/issues/616

#compute mac = multiply accumulation counts
def print_macs_to_file(log=None):

    #nodes = tf.get_default_graph().as_graph_def().node
    #variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    if log is not None:
        log.write( 'MAC for conv layers : \n')
        log.write( 'MAC  param_size  :   name           (op)    params   out    in \n')
        log.write( '----------------------------------------------------------------\n')


    all =0
    all_param_size=0
    all_mac=0

    ops = tf.Graph.get_operations(tf.get_default_graph())
    for op in ops:
        if hasattr(op.op_def, 'name'):
            op_name = op.op_def.name
            if op_name =='Conv2D':

                #print(op.name)
                #print(op.inputs)
                #print(op.outputs)
                #print(op.op_def)
                # assert(op.inputs[1].name == op.name + '_weight/read:0')
                # input_shape  = op.inputs[0].get_shape().as_list()
                # output_shape = op.outputs[0].get_shape().as_list()
                # kernel_shape = op.inputs[1].get_shape().as_list()
                # print(input_shape)
                # print(output_shape)
                # print(kernel_shape)

                g=1 # how do we handle group (e.g used in caffe, mxnet) here ???
                assert(op.inputs[1].name == op.name + '_weight/read:0')
                inum, ih, iw, ic  = op.inputs[0].get_shape().as_list()
                onum, oh, ow, oc  = op.outputs[0].get_shape().as_list()
                h, w, ki, ko    = op.inputs[1].get_shape().as_list()
                assert(ic==ki)
                assert(oc==ko)


                name=op.name
                input_name =op.inputs [0].name
                output_name=op.outputs[0].name

                mac = w*h*ic *oc* oh*ow /1000000./g   #10^6 "multiply-accumulate count"
                param_size = oc*h*w*ic/1000000.

                all_param_size += param_size
                all_mac += mac
                all += 1

                if log is not None:
                    log.write('%10.1f  %5.2f  :  %-26s (%s)   %4d  %dx%dx%4d   %-30s %3d, %3d, %4d,   %-30s %3d, %3d, %5d\n'%
                               (mac, param_size , name,'Conv2D', oc,h,w,ic,  output_name, oh, ow, oc, input_name, ih, iw, ic ))

            if op_name == 'MatMul':
                #raise Exception('xxx')
                # print(op.name)
                # print(op.inputs)
                # print(op.outputs)
                # print(op.op_def)

                #assert (op.inputs[1].name == op.name + ':0')
                inum, ic  = op.inputs[0].get_shape().as_list()
                onum, oc  = op.outputs[0].get_shape().as_list()

                name = op.name
                input_name = op.inputs[0].name
                output_name = op.outputs[0].name

                mac =  ic * oc  / 1000000. / g  # 10^6 "multiply-accumulate count"
                param_size = oc * ic / 1000000.

                all_param_size += param_size
                all_mac += mac
                all += 1

                if log is not None:
                    log.write('%10.1f  %5.2f  :  %-26s (%s)   %4d  %dx%dx%3d   %-30s %3d, %3d, %4d,   %-30s %3d, %3d, %5d\n' %
                        (mac, param_size, name, 'Conv2D', oc, 1, 1, ic, output_name, 1, 1, oc, input_name, 1, 1, ic))
    if log is not None:
        log.write( '\n')
        log.write('summary : \n')
        log.write( 'num of conv     = %d\n'%all)
        log.write( 'all mac         = %.1f (M)\n'%all_mac)
        log.write( 'all param_size  = %.1f (M)\n'%all_param_size)

    return all, all_mac, all_param_size


## loss and metric ## -------------------------------

def l2_regulariser(decay):

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in variables:
        name = v.name
        if 'weight' in name:  #this is weight
            l2 = decay * tf.nn.l2_loss(v)
            tf.add_to_collection('losses', l2)
        elif 'bias' in name:  #this is bias
            pass
        elif 'beta' in name:
            pass
        elif 'gamma' in name:
            pass
        elif 'moving_mean' in name:
            pass
        elif 'moving_variance' in name:
            pass
        elif 'moments' in name:
            pass

        else:
            #pass
            #raise Exception('unknown variable type: %s ?'%name)
            pass

    l2_loss = tf.add_n(tf.get_collection('losses'))
    return l2_loss



def l1_regulariser(decay):

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in variables:
        name = v.name
        if 'weight' in name:  #this is weight
            l1 = decay * tf.reduce_sum(tf.abs(v))
            tf.add_to_collection('losses', l1)
        elif 'bias' in name:  #this is bias
            pass
        elif 'beta' in name:
            pass
        elif 'gamma' in name:
            pass
        elif 'moving_mean' in name:
            pass
        elif 'moving_variance' in name:
            pass
        elif 'moments' in name:
            pass

        else:
            #pass
            raise Exception('unknown variable type: %s ?'%name)

    l1_loss = tf.add_n(tf.get_collection('losses'))
    return l1_loss


def cross_entropy(logit, label, name='cross_entropy'):
    label = tf.cast(label, tf.int64)
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit, label), name=name)
    return cross_entropy


def accuracy(prob, label, name='accuracy'):
    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.cast(label, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name=name)
    return accuracy





## op layers ## -------------------------------

# http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
def conv2d(input, num_kernels=1, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', has_bias=True, name='conv'):

    input_shape = input.get_shape().as_list()
    assert len(input_shape)==4
    C = input_shape[3]
    H = kernel_size[0]
    W = kernel_size[1]
    K = num_kernels

    ##[filter_height, filter_width, in_channels, out_channels]
    w    = tf.get_variable(name=name+'_weight', shape=[H, W, C, K], initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv = tf.nn.conv2d(input, w, strides=stride, padding=padding, name=name)
    if has_bias:
        b = tf.get_variable(name=name + '_bias', shape=[K], initializer=tf.constant_initializer(0.0))
        conv = conv+b

    return conv



def bias(input, name='bias'):

    input_shape = input.get_shape().as_list()
    assert len(input_shape) in [2,4]
    C = input_shape[-1]

    b = tf.get_variable(name=name, shape=[C], initializer=tf.constant_initializer(0.0))
    bias = input+b

    return bias



def relu(input, name='relu'):
    act = tf.nn.relu(input, name=name)
    return act


def prelu(input, name='prelu'):
  alpha = tf.get_variable(name=name+'_alpha', shape=input.get_shape()[-1],
                       #initializer=tf.constant_initializer(0.25),
                        initializer=tf.random_uniform_initializer(minval=0.1, maxval=0.3),
                        dtype=tf.float32)
  pos = tf.nn.relu(input)
  neg = alpha * (input - abs(input)) * 0.5

  return pos + neg


# very leaky relu
def vlrelu(input, alpha=0.25, name='vlrelu'): #  alpha between 0.1 to 0.5
    act =tf.maximum(alpha*input,input)
    return act

def mixpool(input, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', has_bias=True, name='mix' ):
    H = kernel_size[0]
    W = kernel_size[1]
    pool1 = tf.nn.max_pool(input, ksize=[1, H, W, 1], strides=stride, padding=padding, name=name+'/max')
    pool2 = tf.nn.avg_pool(input, ksize=[1, H, W, 1], strides=stride, padding=padding, name=name+'/avg')
    pool =pool1+pool2
    return pool

def maxpool(input, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', has_bias=True, name='max' ):
    H = kernel_size[0]
    W = kernel_size[1]
    pool = tf.nn.max_pool(input, ksize=[1, H, W, 1], strides=stride, padding=padding, name=name)
    return pool

def avgpool(input, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', has_bias=True, is_global_pool=False, name='avg'):

    if is_global_pool==True:
        input_shape = input.get_shape().as_list()
        assert len(input_shape) == 4
        H = input_shape[1]
        W = input_shape[2]

        pool = tf.nn.avg_pool(input, ksize=[1, H, W, 1], strides=[1,H,W,1], padding='VALID', name=name)
        pool = flatten(pool)

    else:
        H = kernel_size[0]
        W = kernel_size[1]
        pool = tf.nn.avg_pool(input, ksize=[1, H, W, 1], strides=stride, padding=padding, name=name)

    return pool


def flatten(input, name='flat'):
    input_shape = input.get_shape().as_list()        # list: [None, 9, 2]
    dim   = np.prod(input_shape[1:])                 # dim = prod(9,2) = 18
    flat  = tf.reshape(input, [-1, dim], name=name)  # -1 means "all"
    return flat

def dense(input, num_hiddens=1,  has_bias=True, name='dense'):
    input_shape = input.get_shape().as_list()
    assert len(input_shape)==2

    C = input_shape[1]
    K = num_hiddens

    w = tf.get_variable(name=name + '_weight', shape=[C,K], initializer=tf.truncated_normal_initializer(stddev=0.1))
    dense = tf.matmul(input, w, name=name)
    if has_bias:
        b = tf.get_variable(name=name + '_bias', shape=[K], initializer=tf.constant_initializer(0.0))
        dense = dense + b

    return dense


def dropout(input, keep=1.0, name='drop'):
    #drop = tf.cond(IS_TRAIN_PHASE, lambda: tf.nn.dropout(input, keep), lambda: input)
    drop = tf.cond(IS_TRAIN_PHASE,
                   lambda: tf.nn.dropout(input, keep),
                   lambda: tf.nn.dropout(input, 1))
    return drop


def spatial_lrn(input, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='lrn'):
    squared = tf.square(input)
    in_channels = input.get_shape().as_list()[3]
    kernel = tf.constant(1.0, shape=[depth_radius, depth_radius, in_channels, 1])
    squared_sum = tf.nn.depthwise_conv2d(squared,
                                         kernel,
                                         [1, 1, 1, 1],
                                         padding='SAME')
    bias  = tf.constant(bias,  dtype=tf.float32)
    alpha = tf.constant(alpha, dtype=tf.float32)
    beta  = tf.constant(beta,  dtype=tf.float32)
    return input / ((bias + alpha * squared_sum) ** beta)


def lrn(input, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, name='lrn'):
    return tf.nn.lrn(input, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta, name=name)



# https://www.tensorflow.org/api_docs/python/nn/other_functions_and_classes#batch_normalization
# adopted from : http://workpiles.com/2016/06/ccb9-tensorflow-batch_norm/
#    https://www.tensorflow.org/api_docs/python/framework/core_graph_data_structures#Graph.control_dependencies
#    http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
#    https://github.com/tensorflow/models/blob/master/resnet/resnet_model.py
def bn0 (input, decay=0.9, eps=1e-5, name='bn'):
    input_shape = input.get_shape().as_list()
    assert len(input_shape) in [2, 4]

    C = input_shape[-1]
    beta  = tf.get_variable(name=name+'_beta',  shape=[C], initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(name=name+'_gamma', shape=[C], initializer=tf.constant_initializer(1.0, tf.float32))

    if len(input_shape) == 2:
        batch_mean, batch_var = tf.nn.moments(input, [0], name=name + '_moments')
    else:
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name=name + '_moments')

    ema = tf.train.ExponentialMovingAverage(decay=decay)
    def train_mode():
        update_moving = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([update_moving]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    def test_mode():
        moving_mean = ema.average(batch_mean)
        moving_var  = ema.average(batch_var)
        return moving_mean, moving_var

    mean, var = tf.cond(IS_TRAIN_PHASE, lambda: train_mode(), lambda: test_mode())
    bn = tf.nn.batch_normalization(input, mean, var, beta, gamma, eps)
    return bn


#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard4/tf.contrib.layers.batch_norm.md
#http://www.bubufx.com/detail-1792794.html
def bn (input, decay=0.9, eps=1e-5, name='bn'):
    with tf.variable_scope(name) as scope:
        bn = tf.cond(IS_TRAIN_PHASE,
            lambda: tf.contrib.layers.batch_norm(input,  decay=decay, epsilon=eps, center=True, scale=True,
                              is_training=1,reuse=None,
                              updates_collections=None, scope=scope),
            lambda: tf.contrib.layers.batch_norm(input, decay=decay, epsilon=eps, center=True, scale=True,
                              is_training=0, reuse=True,
                              updates_collections=None, scope=scope))

    return bn


# Batch Renormalization
# https://www.reddit.com/r/MachineLearning/comments/5tr0cd/r_batch_renormalization_towards_reducing/
# https://github.com/tensorflow/tensorflow/issues/7476
# https://arxiv.org/pdf/1702.03275.pdf
# https://github.com/ppwwyyxx/tensorpack/blob/3f238a015b941041e58ed43e63d5306dbae979fc/tensorpack/models/batch_norm.py#L208

def bn2 (input, decay=0.9, eps=1e-5, rmax=1000, dmax=1000, name='bn' ):

    input_shape = input.get_shape().as_list()
    assert len(input_shape) in [2, 4]
    C = input_shape[-1]

    x = input
    if len(input_shape) == 2:
        x = tf.reshape(x, [-1, 1, 1, C])


    beta  = tf.get_variable(name=name + '_beta',  shape=[C], initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(name=name + '_gamma', shape=[C], initializer=tf.constant_initializer(1.0, tf.float32))
    moving_mean = tf.get_variable(name=name + '_moving_mean',     shape=[C], initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
    moving_var  = tf.get_variable (name=name + '_moving_variance',shape=[C], initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)


    def  train_mode():
        xn, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta, epsilon=eps, is_training=True)
        moving_sigma = tf.sqrt(moving_var)
        r = tf.stop_gradient(tf.clip_by_value(tf.sqrt(batch_var / moving_var), 1.0 / rmax, rmax))
        d = tf.stop_gradient(tf.clip_by_value((batch_mean - moving_mean) / moving_sigma,-dmax, dmax))
        xn = xn * r + d

        #update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay)
        update_op1 = moving_averages.assign_moving_average( moving_mean, batch_mean, decay, zero_debias=False, name='mean_ema_op')
        update_op2 = moving_averages.assign_moving_average( moving_var,  batch_var, decay,  zero_debias=False, name='var_ema_op' )
        add_model_variable(moving_mean)
        add_model_variable(moving_var)
        with tf.control_dependencies([update_op1, update_op2]):
            return tf.identity(xn, name='bn')

    def test_mode():
        xn = tf.nn.batch_normalization( x, moving_mean, moving_var, beta, gamma, variance_epsilon=eps)
        return tf.identity(xn, name='bn')

    bn = tf.cond(IS_TRAIN_PHASE, lambda: train_mode(), lambda: test_mode())
    if len(input_shape) == 2:
        bn = tf.squeeze(xn, [1, 2])

    return bn


def concat(input, name='cat'):
    cat = tf.concat(concat_dim=3, values=input, name=name)
    return cat

## basic blocks ## -------------------------------

def conv2d_bn_relu(input, num_kernels=1, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='conv'):
    with tf.variable_scope(name) as scope:
        block = conv2d(input, num_kernels=num_kernels, kernel_size=kernel_size, stride=stride, padding=padding, has_bias=False)
        block = bn(block)
        block = relu(block)
    return block


def bn_relu_conv2d (input, num_kernels=1, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', name='conv'):
    with tf.variable_scope(name) as scope:
        block = bn(input)
        block = relu(block)
        block = conv2d(block, num_kernels=num_kernels, kernel_size=kernel_size, stride=stride, padding=padding, has_bias=False)
    return block


def bn_conv2d (input, num_kernels=1, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', name='conv'):
    with tf.variable_scope(name) as scope:
        block = bn(input)
        block = conv2d(block, num_kernels=num_kernels, kernel_size=kernel_size, stride=stride, padding=padding, has_bias=False)
    return block

def dense_bn_relu(input, num_hiddens=1, name='dense'):
    with tf.variable_scope(name) as scope:
        block = dense(input, num_hiddens=num_hiddens, has_bias=False)
        block = bn(block)
        block = relu(block)

    return block


# dense
# Densely Connected Convolutional Networks (DenseNets)
# https://github.com/liuzhuang13/DenseNet
def dense_block_brc (input, num=1, num_kernels=1, kernel_size=(1, 1), drop=None, name='DENSE'):

    width=0
    block = input
    for n in  range(num):
        with tf.variable_scope(name+'_%d'%n) as scope:

            conv = bn(block)
            conv = relu(conv)
            conv = conv2d(conv, num_kernels=num_kernels, kernel_size=kernel_size, stride=[1, 1, 1, 1], padding='SAME', has_bias=False)

            if drop is not None:
                keep  = (1-drop) ** (1. / num)
                conv  = dropout(conv, keep=keep)

            block = concat((block, conv))

    return block




def dense_block_cbr (input, num=1, num_kernels=1, kernel_size=(1, 1), drop=None, name='DENSE'):

    #width=0
    block = input
    for n in  range(num):
        with tf.variable_scope(name+'_%d'%n) as scope:
            conv = conv2d(block, num_kernels=num_kernels, kernel_size=kernel_size, stride=[1,1,1,1], padding='SAME', has_bias=False)
            conv = bn(conv)
            conv = relu(conv)

            if drop is not None:
                keep = (1 - drop) ** (1. / num)
                conv = dropout(conv, keep=keep)

            block = concat((block, conv))
    return block


# these code are not tested yet !!!!!!
# residual
# def res_block_A(input, name='res',  num_kernels=(1,1), project=None, stride=[1,1,1,1]):
#
#     block = bn_relu_conv2d(input, name='conv_1', num_kernels=num_kernels[0], kernel=(3,3), stride=stride,    pad='SAME')
#     block = bn_relu_conv2d(block, name='conv_2', num_kernels=num_kernels[1], kernel=(3,3), stride=[1,1,1,1], pad='SAME')
#
#     if project is None:
#         shortcut = input
#     else:
#         shortcut = conv2d(input, name='shortcut', num_kernels=project, kernel_size=(1,1), stride=stride, padding='SAME', has_bias=False)
#
#     add = block + shortcut
#     return add
#
#
# def res_block_B(input, name='res', num_kernels=(1, 1), project=None, stride=[1, 1, 1, 1]):
#     block = bn_relu_conv2d(input, name='conv_1', num_kernels=num_kernels[0], kernel=(3, 3), stride=stride, pad='SAME')
#     block = bn_relu_conv2d(block, name='conv_2', num_kernels=num_kernels[1], kernel=(3, 3), stride=[1, 1, 1, 1],
#                            pad='SAME')
#
#     if project is None:
#         shortcut = input
#     else:
#         shortcut = conv2d(input, name='shortcut', num_kernels=project, kernel_size=(1, 1), stride=stride,
#                           padding='SAME', has_bias=False)
#
#     add = block + shortcut
#     return add

#<todo>
# inception
