from net.blocks import *
from net.file import *


def DenseNet_3( input_shape=(1,1,1), output_shape = (1)):

    H, W, C   = input_shape
    num_class = output_shape
    input     = tf.placeholder(shape=[None, H, W, C], dtype=tf.float32, name='input')

    #color preprocessing using conv net:
    #see "Systematic evaluation of CNN advances on the ImageNet"-Dmytro Mishkin, Nikolay Sergievskiy, Jiri Matas, ARXIV 2016
    # https://arxiv.org/abs/1606.02228
    # we use learnable prelu (different from paper) and 3x3 onv
    with tf.variable_scope('preprocess') as scope:
        input = bn(input, name='b1')
        input = conv2d(input, num_kernels=8, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME', has_bias=True, name='c1')
        input = prelu(input, name='r1')
        input = conv2d(input, num_kernels=8, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', has_bias=True, name='c2')
        input = prelu(input, name='r2')

    with tf.variable_scope('block1') as scope:
        block1 = conv2d_bn_relu(input, num_kernels=32, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME')
        block1 = maxpool(block1, kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')

    # we use conv-bn-relu in DENSE block (different from paper).dropout is taken out of the block
    with tf.variable_scope('block2') as scope:
        block2 = dense_block_cbr(block1, num=4, num_kernels=16, kernel_size=(3, 3), drop=None)
        block2 = maxpool(block2, kernel_size=(2, 2), stride=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('block3') as scope:
        block3 = dense_block_cbr(block2, num=4, num_kernels=24, kernel_size=(3, 3), drop=None)
        block3 = dropout(block3, keep=0.9)
        block3 = maxpool(block3,  kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('block4') as scope:
        block4 = dense_block_cbr(block3, num=4, num_kernels=32, kernel_size=(3, 3), drop=None)
        block4 = conv2d_bn_relu(block4, num_kernels=num_class, kernel_size=(1,1), stride=[1, 1, 1, 1], padding='SAME')
        block4 = dropout(block4, keep=0.8)
        block4 = avgpool(block4, is_global_pool=True)


    logit = block4
    return logit


#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print  ( 'running main function ...' )

    log = Logger()  # log file
    log.open('/root/share/out/udacity/05/mac_log.txt', mode='a')

    out_dir = '/root/share/out/udacity/05/tf'
    empty(out_dir)

    num_class=43
    H, W, C = (32, 32, 3)
    logit = DenseNet_2(input_shape=(H,W,C), output_shape = (num_class))

    #input   = tf.get_default_graph().get_tensor_by_name('input:0')
    #print(input)

    # draw graph to check connections
    with tf.Session()  as sess:
        tf.global_variables_initializer().run(feed_dict={IS_TRAIN_PHASE:True})
        summary_writer = tf.summary.FileWriter(out_dir, sess.graph)

        print_macs_to_file(log)
    print ('sucess!')

