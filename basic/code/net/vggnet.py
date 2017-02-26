from net.blocks import *
from net.file import *



#vgg  + bn
def Vgg_0( input_shape=(1,1,1), output_shape = (1)):

    H, W, C   = input_shape
    num_class = output_shape
    input = tf.placeholder(shape=[None, H, W, C], dtype=tf.float32, name='input')


    with tf.variable_scope('block1') as scope:
        block1 = conv2d_bn_relu(input, num_kernels=64, kernel_size=(5,5), stride=[1,1,1,1], padding='SAME')
        block1 = maxpool(block1,  kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('block2') as scope:
        block2 = conv2d_bn_relu(block1, name='1', num_kernels=32,  kernel_size=(1,1), stride=[1,1,1,1], padding='SAME')
        block2 = conv2d_bn_relu(block2, name='2', num_kernels=32,  kernel_size=(3,3), stride=[1,1,1,1], padding='SAME')
        block2 = conv2d_bn_relu(block2, name='3', num_kernels=64,  kernel_size=(1,1), stride=[1,1,1,1], padding='SAME')
        block2 = maxpool(block2, kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('block3') as scope:
        block3 = conv2d_bn_relu(block2, name='1', num_kernels=64,  kernel_size=(1,1), stride=[1,1,1,1], padding='SAME')
        block3 = conv2d_bn_relu(block3, name='2', num_kernels=64,  kernel_size=(3,3), stride=[1,1,1,1], padding='SAME')
        block3 = conv2d_bn_relu(block3, name='3', num_kernels=128, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME')

    with tf.variable_scope('block4') as scope:
        block4 = conv2d_bn_relu(block3, name='1', num_kernels=64,  kernel_size=(1,1), stride=[1,1,1,1], padding='SAME')
        block4 = conv2d_bn_relu(block4, name='2', num_kernels=64,  kernel_size=(3,3), stride=[1,1,1,1], padding='SAME')
        block4 = conv2d_bn_relu(block4, name='3', num_kernels=128, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME')
        block4 = maxpool(block4, kernel_size=(2,2), stride=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('block5') as scope:
        block4 = flatten(block4)
        block5 = dense_bn_relu(block4, name='1', num_hiddens=256)
        block5 = dense_bn_relu(block5, name='2', num_hiddens=256)
        block5 = dense_bn_relu(block5, name='3', num_hiddens=num_class)

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

