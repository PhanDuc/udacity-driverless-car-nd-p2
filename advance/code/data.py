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


# see also: https://github.com/dmlc/mxnet/blob/master/python/mxnet/image.py
# use opencv to do data agumentation


# def perturb(image, keep, angle_limit=15, scale_limit=0.1, translate_limit=3, distort_limit=3, illumin_limit=0.5)
def perturb(image, keep, angle_limit=15, scale_limit=0.1, translate_limit=3, distort_limit=3, illumin_limit=0.7):

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
        perturb = cv2.warpPerspective(image, matrix, (W, H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)  # BORDER_WRAP  #BORDER_REFLECT_101  #cv2.BORDER_CONSTANT  BORDER_REPLICATE

        #brightness, contrast, saturation-------------
        #from mxnet code
        if 1:  #brightness
            alpha = 1.0 + illumin_limit*random.uniform(-1, 1)
            perturb *= alpha
            perturb = np.clip(perturb,0.,255.)
            pass

        if 1:  #contrast
            coef = np.array([[[0.299, 0.587, 0.114]]]) #rgb to gray (YCbCr) :  Y = 0.299R + 0.587G + 0.114B

            alpha = 1.0 + illumin_limit*random.uniform(-1, 1)
            gray = perturb * coef
            gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
            perturb *= alpha
            perturb += gray
            perturb = np.clip(perturb,0.,255.)
            pass

        if 1:  #saturation
            coef = np.array([[[0.299, 0.587, 0.114]]]) #rgb to gray (YCbCr) :  Y = 0.299R + 0.587G + 0.114B

            alpha = 1.0 + illumin_limit*random.uniform(-1, 1)
            gray = perturb * coef
            gray = np.sum(gray, axis=2, keepdims=True)
            gray *= (1.0 - alpha)
            perturb *= alpha
            perturb += gray
            perturb = np.clip(perturb,0.,255.)
            pass

        return perturb

    else:
        return image



def make_perturb_images(images, keep ):
    #images = undo_preprocess_simple(images)

    arguments = np.zeros(images.shape)
    for n in range(len(images)):
        arguments[n] = perturb(images[n],keep = keep)

    #arguments = preprocess_simple(arguments)
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



# exploration -------------------------------------------------------------------
def get_label_image(c):

    img=cv2.imread('/root/share/project/udacity/project2_01/data/signnames_all.jpg',1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    dH = H/7.
    dW = W/7.105
    y = c//7
    x = c%7

    label_image = img[round(y*dH):round(y*dH+dH), round(x*dW):round(x*dW+dW),:]
    label_image = cv2.resize(label_image, (0,0), fx=32./dW, fy=32./dH,)
    return label_image


def insert_subimage(image, sub_image, y, x):

    h, w, c = sub_image.shape
    image[y:y+h, x:x+w, :]=sub_image

    return image


#data summary
def run_data_explore_0():

    classnames, train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_data()
    num_class = 43
    _, height, width, channel = train_images.shape

    #count
    #h = np.histogram(train_labels, bins=np.arange(num_class))

    #results image
    num_sample=10
    results_image = 255.*np.ones(shape=(num_class*height,(num_sample+2+22)*width, channel),dtype=np.float32)
    for c in range(num_class):
        label_image = get_label_image(c)
        insert_subimage(results_image, label_image, c*height, 0)

        #make mean
        idx = list(np.where(train_labels== c)[0])
        mean_image = np.average(train_images[idx], axis=0)
        insert_subimage(results_image, mean_image, c*height, width)

        # imshow('mean_image',mean_image)
        # imshow('label_image',label_image)
        # cv2.waitKey(0)

        #make random sample
        for n in range(num_sample):
            sample_image = train_images[np.random.choice(idx)]
            insert_subimage(results_image, sample_image, c*height, (2+n)*width)

        #print summary
        count=len(idx)
        percentage = float(count)/float(len(train_images))
        cv2.putText(results_image, '%02d:%-6s'%(c, classnames[c]), ((2+num_sample)*width, int((c+0.7)*height)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
        cv2.putText(results_image, '[%4d]'%(count), ((2+num_sample+14)*width, int((c+0.7)*height)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        cv2.rectangle(results_image,((2+num_sample+16)*width, c*height),((2+num_sample+16)*width + round(percentage*3000), (c+1)*height),(0,0,255),-1)


    cv2.imwrite('/root/share/project/udacity/project2_01/data/data_summary.jpg',cv2.cvtColor(results_image, cv2.COLOR_BGR2RGB))
    imshow('results_image',results_image)
    cv2.waitKey(0)


#data augmentation
def run_data_explore_1():

    classnames, train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_data()
    train_images, train_labels = extend_data_by_flipping(train_images, train_labels)
    num_train_flip = len(train_images)

    num_class = 43
    _, height, width, channel = train_images.shape

    # count
    # h = np.histogram(train_labels, bins=np.arange(num_class))

    # results image
    num_sample = 30
    perturbance_per_sample = 30

    results_image = 255. * np.ones(shape=(num_sample * height, (perturbance_per_sample+1)* width+10, channel),dtype=np.float32)

    for j in range(num_sample):
        i = random.randint(0, num_train_flip - 1)

        image = train_images[i]
        insert_subimage(results_image, image, j * height, 0)

        for k in range(0, perturbance_per_sample):
            perturb_image = perturb(image, keep=0)
            insert_subimage(results_image, perturb_image, j*height, (k+1)*width+10)



    cv2.imwrite('/root/share/project/udacity/project2_01/data/data_perturb.jpg',cv2.cvtColor(results_image, cv2.COLOR_BGR2RGB))
    imshow('results_image', results_image)
    cv2.waitKey(0)


# main --------------------------------------------------------------------------
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_data_explore_1()