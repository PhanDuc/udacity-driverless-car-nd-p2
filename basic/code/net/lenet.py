from net.blocks import *
from net.file import *

#LeNet
def LeNet_0( input_shape=(1,1,1), output_shape = (1)):

    H, W, C   = input_shape
    num_class = output_shape
    input = tf.placeholder(shape=[None, H, W, C], dtype=tf.float32, name='input')

    with tf.variable_scope('block1') as scope:
        block1 = conv2d(input, num_kernels=108, kernel_size=(5,5), stride=[1,1,1,1], padding='SAME', has_bias=True)
        block1 = relu(block1)
        block1 = maxpool(block1,  kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('block2') as scope:
        block2 = conv2d(block1, num_kernels=108, kernel_size=(5,5), stride=[1,1,1,1], padding='SAME', has_bias=True)
        block2 = relu(block2)
        block2 = maxpool(block2, kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('block3') as scope:
        block3  = flatten(block2)
        block3  = dense(block3, num_hiddens=100, has_bias=True)
        block3  = relu(block3)

    with tf.variable_scope('block4') as scope:
        block4 = dense(block3, num_hiddens=100, has_bias=True)
        block4 = relu(block4)

    with tf.variable_scope('block5') as scope:
        block5 = dense(block4, num_hiddens=num_class, has_bias=True)

    logit = block5
    return logit



#LeNet + dropout
def LeNet_1( input_shape=(1,1,1), output_shape = (1)):
    H, W, C = input_shape
    num_class = output_shape
    input = tf.placeholder(shape=[None, H, W, C], dtype=tf.float32, name='input')

    with tf.variable_scope('block1') as scope:
        block1 = conv2d(input, num_kernels=108, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME', has_bias=True)
        block1 = relu(block1)
        block1 = maxpool(block1, kernel_size=(2, 2), stride=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('block2') as scope:
        block2 = conv2d(block1, num_kernels=108, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME', has_bias=True)
        block2 = relu(block2)
        block2 = maxpool(block2, kernel_size=(2, 2), stride=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('block3') as scope:
        block3 = flatten(block2)
        block3 = dense(block3, num_hiddens=100, has_bias=True)
        block3 = relu(block3)
        block3 = dropout(block3,keep=0.5)

    with tf.variable_scope('block4') as scope:
        block4 = dense(block3, num_hiddens=100, has_bias=True)
        block4 = relu(block4)
        block4 = dropout(block4,keep=0.5)

    with tf.variable_scope('block5') as scope:
        block5 = dense(block4, num_hiddens=num_class, has_bias=True)

    logit = block5
    return logit



#LeNet + dropout + bn + whiten
def LeNet_3( input_shape=(1,1,1), output_shape = (1)):

    H, W, C = input_shape
    num_class = output_shape
    input = tf.placeholder(shape=[None, H, W, C], dtype=tf.float32, name='input')

    input = tf.map_fn(lambda img: tf.image.per_image_standardization(img), input)
    with tf.variable_scope('block1') as scope:
        block1 = conv2d(input, num_kernels=108, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME', has_bias=False)
        block1 = bn(block1)
        block1 = relu(block1)
        block1 = maxpool(block1, kernel_size=(2, 2), stride=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('block2') as scope:
        block2 = conv2d(block1, num_kernels=108, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME', has_bias=False)
        block2 = bn(block2)
        block2 = relu(block2)
        block2 = maxpool(block2, kernel_size=(2, 2), stride=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('block3') as scope:
        block3 = flatten(block2)
        block3 = dense(block3, num_hiddens=100, has_bias=False)
        block3 = bn(block3)
        block3 = relu(block3)
        block3 = dropout(block3,keep=0.5)

    with tf.variable_scope('block4') as scope:
        block4 = dense(block3, num_hiddens=100, has_bias=False)
        block4 = bn(block4)
        block4 = relu(block4)
        block4 = dropout(block4,keep=0.5)

    with tf.variable_scope('block5') as scope:
        block5 = dense(block4, num_hiddens=num_class, has_bias=True)

    logit = block5
    return logit

# LeNet + dropout + bn
def LeNet_2(input_shape=(1, 1, 1), output_shape=(1)):
    H, W, C = input_shape
    num_class = output_shape
    input = tf.placeholder(shape=[None, H, W, C], dtype=tf.float32, name='input')

    with tf.variable_scope('block1') as scope:
        block1 = conv2d(input, num_kernels=108, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME',
                        has_bias=False)
        block1 = bn(block1)
        block1 = relu(block1)
        block1 = maxpool(block1, kernel_size=(2, 2), stride=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('block2') as scope:
        block2 = conv2d(block1, num_kernels=108, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME',
                        has_bias=False)
        block2 = bn(block2)
        block2 = relu(block2)
        block2 = maxpool(block2, kernel_size=(2, 2), stride=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('block3') as scope:
        block3 = flatten(block2)
        block3 = dense(block3, num_hiddens=100, has_bias=False)
        block3 = bn(block3)
        block3 = relu(block3)
        block3 = dropout(block3, keep=0.5)

    with tf.variable_scope('block4') as scope:
        block4 = dense(block3, num_hiddens=100, has_bias=False)
        block4 = bn(block4)
        block4 = relu(block4)
        block4 = dropout(block4, keep=0.5)

    with tf.variable_scope('block5') as scope:
        block5 = dense(block4, num_hiddens=num_class, has_bias=True)

    logit = block5
    return logit




#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print  ( 'running main function ...' )

    log = Logger()  # log file
    log.open('/root/share/out/udacity/11/mac.txt', mode='a')

    out_dir = '/root/share/out/udacity/11/tf'
    empty(out_dir)

    num_class=43
    H, W, C = (32, 32, 3)
    logit = LeNet_4(input_shape=(H,W,C), output_shape = (num_class))

    #input   = tf.get_default_graph().get_tensor_by_name('input:0')
    #print(input)

    # draw graph to check connections
    with tf.Session()  as sess:
        tf.global_variables_initializer().run(feed_dict={IS_TRAIN_PHASE:True})
        summary_writer = tf.summary.FileWriter(out_dir, sess.graph)

        print_macs_to_file(log)
    print ('sucess!')

