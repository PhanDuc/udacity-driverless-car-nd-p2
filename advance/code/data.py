'''
 all data processing
'''

from net.common import *


# data visualisation ----------------------
# e.g. for debug

def imshow(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8))
    cv2.resizeWindow(name, 320, 320)


def show_data(datas,labels,classnames, num=None, pause_time=-1 ):
    N=num if num is not None else len(datas)
    for n in range(N):
        data  =  datas[n]
        label =  labels[n]

        # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = undo_preprocess_simple(data).astype(np.uint8)
        fig, ax = plt.subplots(1, 1, figsize=[6, 6])
        fig.suptitle('label=%d : %s' % (label, classnames[label]), fontsize=14)
        ax.imshow(data)


        if(pause_time>0):
            plt.pause(pause_time)
        else:
            plt.show()


# data pre-processor ----------------------
def preprocess_simple(images):
    images = (images-128.)/128.
    return images


def undo_preprocess_simple(images):
    images = images*128. + 128.
    return images


def preprocess_whiten(images):
    '''
       y = (x - mean) / adjusted_stddev

          where
                mean =  average  of all values in image,
                adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))
    '''
    images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), images)
    return images


def pre_process_ycrcb(images):
    images=images.astype(np.uint8)

    N=len(images)
    for n in range(N):
        images[n] = cv2.cvtColor(images[n], cv2.COLOR_BGR2YCR_CB)

    images = (images.astype(np.float32) - 128.) / 128.
    return images





# data agumentation ----------------------

## http://navoshta.com/traffic-signs-classification/
def extend_data_by_flipping(images, labels):

    X=images
    y=labels

    # Classes of signs that, when flipped horizontally, should still be classified as the same class
    self_flippable_horizontally = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    # Classes of signs that, when flipped vertically, should still be classified as the same class
    self_flippable_vertically = np.array([1, 5, 12, 15, 17])
    # Classes of signs that, when flipped horizontally and then vertically, should still be classified as the same class
    self_flippable_both = np.array([32, 40])
    # Classes of signs that, when flipped horizontally, would still be meaningful, but should be classified as some other class
    cross_flippable = np.array([
        [19, 20],
        [33, 34],
        [36, 37],
        [38, 39],
        [20, 19],
        [34, 33],
        [37, 36],
        [39, 38],
    ])
    num_classes = 43

    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype=np.float32)
    y_extended = np.empty([0], dtype=np.int32)

    for c in range(num_classes):
        # First copy existing data for this class
        X_extended = np.append(X_extended, X[y == c], axis=0)
        # If we can flip images of this class horizontally and they would still belong to said class...
        if c in self_flippable_horizontally:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X[y == c][:, :, ::-1, :], axis=0)
        # If we can flip images of this class horizontally and they would belong to other class...
        if c in cross_flippable[:, 0]:
            # ...Copy flipped images of that other class to the extended array.
            flip_class = cross_flippable[cross_flippable[:, 0] == c][0][1]
            X_extended = np.append(X_extended, X[y == flip_class][:, :, ::-1, :], axis=0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=np.int32))

        # If we can flip images of this class vertically and they would still belong to said class...
        if c in self_flippable_vertically:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, :, :], axis=0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=np.int32))

        # If we can flip images of this class horizontally AND vertically and they would still belong to said class...
        if c in self_flippable_both:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, ::-1, :], axis=0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=np.int32))

    extend_datas  = X_extended
    extend_labels = y_extended
    return (extend_datas, extend_labels)



# use opencv to do data agumentation
def perturb(image, keep, angle_limit=15, scale_limit=0.1, translate_limit=3, distort_limit=3):

    u=np.random.uniform()
    if u>keep :
        (W, H, C) = image.shape
        center = np.array([W / 2., H / 2.])
        da = np.random.uniform(low=-1, high=1) * angle_limit/180. * math.pi
        scale = np.random.uniform(low=-1, high=1) * scale_limit + 1

        cc = scale*math.cos(da)
        ss = scale*math.sin(da)
        rotation    = np.array([[cc, ss],[-ss,cc]])
        translation = np.random.uniform(low=-1, high=1, size=(1,2)) * translate_limit
        distort     = np.random.standard_normal(size=(4,2)) * distort_limit

        pts1 = np.array([[0., 0.], [0., H], [W, H], [W, 0.]])
        pts2 = np.matmul(pts1-center, rotation) + center  + translation

        #add perspective noise
        pts2 = pts2 + distort


        #http://milindapro.blogspot.jp/2015/05/opencv-filters-copymakeborder.html
        matrix  = cv2.getPerspectiveTransform(pts1.astype(np.float32), pts2.astype(np.float32))
        #perturb = cv2.warpPerspective(image, matrix, (W, H),flags =cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)  #BORDER_WRAP  #BORDER_REFLECT_101
        perturb = cv2.warpPerspective(image, matrix, (W, H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)  # BORDER_WRAP  #BORDER_REFLECT_101  #cv2.BORDER_CONSTANT  BORDER_REPLICATE

        return perturb

    else:
        return image



def make_perturb_images(images, keep ):
    arguments = np.zeros(images.shape)
    for n in range(len(images)):
        arguments[n] = perturb(images[n],keep = keep)

    return arguments


# data sampler ----------------------
def generate_train_batch_random (train_datas, train_labels, batch_size):

    #here we just subsample random block
    num  = len(train_datas)
    i = np.random.randint(0, num-batch_size)   ## set 0 for debug
    batch_datas  = train_datas [i:i+batch_size]
    batch_labels = train_labels[i:i+batch_size]
    return batch_datas, batch_labels



def generate_train_batch_next(datas, labels, n, batch_size):
    i = n*batch_size
    batch_datas  = datas [i:i+batch_size]
    batch_labels = labels[i:i+batch_size]
    return batch_datas, batch_labels



def shuffle_data(datas, labels):

    num =len(datas)
    index = list(range(num))
    random.shuffle(index)
    shuffle_datas  = datas[index]
    shuffle_labels = labels[index]

    return shuffle_datas, shuffle_labels


def shuffle_data_uniform(datas, labels, num_class, num_per_class=None):

    if num_per_class is None:
        max_count = 0
        for c in range(num_class):
            idx = list(np.where(labels == c)[0])
            count = len(idx)
            max_count = max(count, max_count)
        num_per_class = max_count

    index = []
    for c in range(num_class):
        idx = list(np.where(labels == c)[0])
        index = index + list(np.random.choice(idx, num_per_class))

    random.shuffle(index)
    shuffle_datas  = datas[index]
    shuffle_labels = labels[index]

    return shuffle_datas, shuffle_labels




# data loader ----------------------

def load_data():

    # https://raw.githubusercontent.com/udacity/CarND-Traffic-Sign-Classifier-Project/master/signnames.csv

    data_dir = '/root/share/project/udacity/project2_01/data'
    training_file  = data_dir + '/train.p'
    testing_file   = data_dir + '/test.p'
    classname_file = data_dir + '/signnames.csv'

    classnames = []
    with open(classname_file) as _f:
        rows = csv.reader(_f, delimiter=',')
        next(rows, None)  # skip the headers
        for i, row in enumerate(rows):
            assert(i==int(row[0]))
            classnames.append(row[1])


    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    num  = len(train['labels'])   # 39209
    num_valid = 3000  #let's use 5000 as validation

    index = list(range(num))
    random.shuffle(index)
    train_index=index[num_valid:]
    valid_index=index[:num_valid]


    train_images = train['features'][train_index].astype(np.float32)
    train_labels = train['labels'  ][train_index].astype(np.int32)
    valid_images = train['features'][valid_index].astype(np.float32)
    valid_labels = train['labels'  ][valid_index].astype(np.int32)
    test_images  = test ['features'].astype(np.float32)
    test_labels  = test ['labels'  ].astype(np.int32)

    # print('train_images=%s'%str(train_images.shape))
    # print('train_labels=%s'%str(train_labels.shape))
    # print('valid_images=%s'%str(valid_images.shape))
    # print('valid_labels=%s'%str(valid_labels.shape))
    # print('test_images=%s' %str(test_images.shape))
    # print('test_labels=%s' %str(test_labels.shape))
    # print('')

    return  classnames, train_images, train_labels,  valid_images, valid_labels, test_images, test_labels




# main --------------------------------------------------------------------------
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
