from net.blocks import *
from net.file import *


def LeNet( input_shape=(1,1,1), output_shape = (1)):
    #set_description('LeNet_0: standard LeNet')

    H, W, C   = input_shape
    num_class = output_shape
    input = tf.placeholder(shape=[None, H, W, C], dtype=tf.float32, name='input')

    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d(input, num_kernels=108, kernel_size=(5,5), stride=[1,1,1,1], padding='SAME', has_bias=True)
        relu1 = relu(conv1)

    with tf.variable_scope('pool1') as scope:
        pool1 = maxpool(relu1,  kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')


    with tf.variable_scope('conv2') as scope:
        conv2 = conv2d(pool1, num_kernels=108, kernel_size=(5,5), stride=[1,1,1,1], padding='SAME', has_bias=True)
        relu2 = relu(conv2)

    with tf.variable_scope('pool2') as scope:
        pool2 = maxpool(relu2, kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('dense3') as scope:
        flat   = flatten(pool2)
        dense3 = dense(flat, num_hiddens=100, has_bias=True)
        relu3  = relu(dense3)

    with tf.variable_scope('dense4') as scope:
        dense4 = dense(relu3, num_hiddens=100, has_bias=True)
        relu4  = relu(dense4)

    with tf.variable_scope('dense5') as scope:
        dense5 = dense(relu4, num_hiddens=num_class, has_bias=True)

    logit = dense5
    return logit




def LeNet_1( input_shape=(1,1,1), output_shape = (1)):
    #set_description('LeNet_1: LeNet  + dropout')

    H, W, C   = input_shape
    num_class = output_shape
    input = tf.placeholder(shape=[None, H, W, C], dtype=tf.float32, name='input')

    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d(input, num_kernels=108, kernel_size=(5,5), stride=[1,1,1,1], padding='SAME', has_bias=True)
        relu1 = relu(conv1)

    with tf.variable_scope('pool1') as scope:
        pool1 = maxpool(relu1,  kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')


    with tf.variable_scope('conv2') as scope:
        conv2 = conv2d(pool1, num_kernels=108, kernel_size=(5,5), stride=[1,1,1,1], padding='SAME', has_bias=True)
        relu2 = relu(conv2)

    with tf.variable_scope('pool2') as scope:
        pool2 = maxpool(relu2, kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')


    with tf.variable_scope('dense3') as scope:
        flat   = flatten(pool2)
        dense3 = dense(flat, num_hiddens=100, has_bias=True)
        relu3  = relu(dense3)
        relu3  = dropout(relu3, keep=0.5)


    with tf.variable_scope('dense4') as scope:
        dense4 = dense(relu3, num_hiddens=100, has_bias=True)
        relu4  = relu(dense4)
        relu4  = dropout(relu4, keep=0.5)


    with tf.variable_scope('dense5') as scope:
        dense5 = dense(relu4, num_hiddens=num_class, has_bias=True)

    logit = dense5
    return logit




def LeNet_2( input_shape=(1,1,1), output_shape = (1)):
    #set_description('LeNet_4: LeNet bn + dropout')

    H, W, C   = input_shape
    num_class = output_shape
    input = tf.placeholder(shape=[None, H, W, C], dtype=tf.float32, name='input')
    bn0   = bn(input, name='bn0')

    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d(bn0, num_kernels=108, kernel_size=(5,5), stride=[1,1,1,1], padding='SAME', has_bias=True)
        conv1 = bn(conv1)
        relu1 = relu(conv1)

    with tf.variable_scope('pool1') as scope:
        pool1 = maxpool(relu1,  kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')


    with tf.variable_scope('conv2') as scope:
        conv2 = conv2d(pool1, num_kernels=108, kernel_size=(5,5), stride=[1,1,1,1], padding='SAME', has_bias=True)
        conv2 = bn(conv2)
        relu2 = relu(conv2)

    with tf.variable_scope('pool2') as scope:
        pool2 = maxpool(relu2, kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')


    with tf.variable_scope('dense3') as scope:
        flat   = flatten(pool2)
        dense3 = dense(flat, num_hiddens=100, has_bias=True)
        dense3 = bn(dense3)
        relu3  = relu(dense3)
        relu3  = dropout(relu3, keep=0.5)


    with tf.variable_scope('dense4') as scope:
        dense4 = dense(relu3, num_hiddens=100, has_bias=True)
        dense4 = bn(dense4)
        relu4  = relu(dense4)
        relu4  = dropout(relu4, keep=0.5)


    with tf.variable_scope('dense5') as scope:
        dense5 = dense(relu4, num_hiddens=num_class, has_bias=True)

    logit = dense5
    return logit





def LeNet_3( input_shape=(1,1,1), output_shape = (1)):
    #set_description('LeNet_3: LeNet(deeper) + bn + dropout')

    H, W, C   = input_shape
    num_class = output_shape
    input = tf.placeholder(shape=[None, H, W, C], dtype=tf.float32, name='input')
    bn0   = bn(input, name='bn0')


    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d(bn0, num_kernels=96, kernel_size=(5,5), stride=[1,1,1,1], padding='SAME', has_bias=False)
        conv1 = bn(conv1)
        relu1 = relu(conv1)
    with tf.variable_scope('pool1') as scope:
        pool1 = maxpool(relu1,  kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')


    with tf.variable_scope('conv2') as scope:
        conv2 = conv2d(pool1, num_kernels=96, kernel_size=(5,5), stride=[1,1,1,1], padding='SAME', has_bias=False)
        conv2 = bn(conv2)
        relu2 = relu(conv2)
    with tf.variable_scope('pool2') as scope:
        pool2 = maxpool(relu2, kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')


    with tf.variable_scope('conv3') as scope:
        conv3 = conv2d(pool2, num_kernels=96, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', has_bias=False)
        conv3 = bn(conv3)
        relu3 = relu(conv3)

    with tf.variable_scope('conv4') as scope:
        conv4 = conv2d(conv3, num_kernels=96, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', has_bias=False)
        conv4 = bn(conv4)
        relu4 = relu(conv4)


    with tf.variable_scope('dense5') as scope:
        flat   = flatten(relu4)
        dense5 = dense(flat, num_hiddens=100, has_bias=False)
        dense5 = bn(dense5)
        relu5  = relu(dense5)
        relu5  = dropout(relu5, keep=0.5)


    with tf.variable_scope('dense6') as scope:
        dense6 = dense(relu5, num_hiddens=100, has_bias=False)
        dense6 = bn(dense6)
        relu6  = relu(dense6)
        relu6  = dropout(relu6, keep=0.5)


    with tf.variable_scope('dense7') as scope:
        dense7 = dense(relu6, num_hiddens=num_class, has_bias=False)

    logit = dense7
    return logit



def LeNet_4( input_shape=(1,1,1), output_shape = (1)):
    #set_description('LeNet_4: LeNet(fully convolutional + global ave) + bn + dropout')

    H, W, C   = input_shape
    num_class = output_shape
    input = tf.placeholder(shape=[None, H, W, C], dtype=tf.float32, name='input')


    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d_bn_relu(input, num_kernels=64, kernel_size=(5,5), stride=[1,1,1,1], padding='SAME')
        pool1 = maxpool(conv1,  kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')


    with tf.variable_scope('conv2') as scope:
        conv2_1 = conv2d_bn_relu(pool1,   name='1', num_kernels=32,  kernel_size=(1,1), stride=[1,1,1,1], padding='SAME')
        conv2_2 = conv2d_bn_relu(conv2_1, name='2', num_kernels=32,  kernel_size=(3,3), stride=[1,1,1,1], padding='SAME')
        conv2_3 = conv2d_bn_relu(conv2_2, name='3', num_kernels=64,  kernel_size=(1,1), stride=[1,1,1,1], padding='SAME')
        pool2 = maxpool(conv2_3, kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')


    with tf.variable_scope('conv3') as scope:
        conv3_1 = conv2d_bn_relu(pool2,   name='1', num_kernels=32,  kernel_size=(1,1), stride=[1,1,1,1], padding='SAME')
        conv3_2 = conv2d_bn_relu(conv3_1, name='2', num_kernels=32,  kernel_size=(3,3), stride=[1,1,1,1], padding='SAME')
        conv3_3 = conv2d_bn_relu(conv3_2, name='3', num_kernels=64,  kernel_size=(1,1), stride=[1,1,1,1], padding='SAME')


    with tf.variable_scope('conv4') as scope:
        conv4_1 = conv2d_bn_relu(conv3_3, name='1', num_kernels=64,  kernel_size=(1,1), stride=[1,1,1,1], padding='SAME')
        conv4_2 = conv2d_bn_relu(conv4_1, name='2', num_kernels=64,  kernel_size=(3,3), stride=[1,1,1,1], padding='SAME')
        conv4_3 = conv2d_bn_relu(conv4_2, name='3', num_kernels=128, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME')


    with tf.variable_scope('conv5') as scope:
        conv5 = conv2d_bn_relu(conv4_3, num_kernels=512,  kernel_size=(1,1), stride=[1,1,1,1], padding='SAME')
        conv5 = dropout(conv5, keep=0.5)

    with tf.variable_scope('conv6') as scope:
        conv6 = conv2d_bn_relu(conv5, num_kernels=num_class,  kernel_size=(1,1), stride=[1,1,1,1], padding='SAME')
        conv6 = dropout(conv6, keep=0.5)

    with tf.variable_scope('pool') as scope:
        pool = avgpool(conv6, is_global_pool=True)
        pool = flatten(pool)

    logit = pool
    return logit

#http://navoshta.com/traffic-signs-classification/
def LeNet_5( input_shape=(1,1,1), output_shape = (1)):
    #set_description('LeNet_4: LeNet(fully convolutional + global ave) + bn + dropout')

    H, W, C   = input_shape
    num_class = output_shape
    input = tf.placeholder(shape=[None, H, W, C], dtype=tf.float32, name='input')


    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d_bn_relu(input, num_kernels=32, kernel_size=(5,5), stride=[1,1,1,1], padding='SAME')
        pool1 = maxpool(conv1,  kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')


    with tf.variable_scope('conv2') as scope:
        conv2 = conv2d_bn_relu(pool1, num_kernels=64,  kernel_size=(5,5), stride=[1,1,1,1], padding='SAME')
        pool2 = maxpool(conv2, kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')


    with tf.variable_scope('conv3') as scope:
        conv3 = conv2d_bn_relu(pool2,  num_kernels=128,  kernel_size=(5,5), stride=[1,1,1,1], padding='SAME')
        pool3 = maxpool(conv3, kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')



    with tf.variable_scope('dense4') as scope:
        i1 = maxpool(pool1,  name='1', kernel_size=(4,4), stride=[1, 4, 4, 1], padding='SAME')
        i2 = maxpool(pool2,  name='2', kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')
        cat = concat((i1, i2, pool3))
        flat = flatten(cat)
        dense4 = dense_bn_relu(flat, num_hiddens=1024)


    with tf.variable_scope('dense5') as scope:
        dense5 = dense(dense4, num_hiddens=num_class, has_bias=False)

    logit = dense5
    return logit


#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print  ( 'running main function ...' )

    log = Logger()  # log file
    log.open('/root/share/out/udacity/00/xxx_log.txt', mode='a')

    out_dir = '/root/share/out/udacity/00/tf'
    empty(out_dir)

    num_class=43
    H, W, C = (32, 32, 3)
    #logit = LeNet(input_shape=(H,W,C), output_shape = (num_class))
    logit = LeNet_5(input_shape=(H,W,C), output_shape = (num_class))

    #input   = tf.get_default_graph().get_tensor_by_name('input:0')
    #print(input)

    # draw graph to check connections
    with tf.Session()  as sess:
        tf.global_variables_initializer().run(feed_dict={IS_TRAIN_PHASE:True})
        summary_writer = tf.summary.FileWriter(out_dir, sess.graph)

        print_macs_to_file(log)
    print ('sucess!')


