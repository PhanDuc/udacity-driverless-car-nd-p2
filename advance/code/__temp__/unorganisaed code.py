

# show preprocessing layer
def run_extra_test_1():

    # output dir, etc
    out_dir = '/root/share/docs/git/hengck23-udacity/udacity-driverless-car-nd-p2/submission(notebook+html)/002/out'

    # data -------------------------------------------------------------------------
    print('read data:\n')
    classnames, train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_data()

    num_class = 43
    _, height, width, channel = train_images.shape
    num_train = len(train_images)
    num_valid = len(valid_images)
    num_test  = len(test_images)

    # net  -----------------------------------------------
    logit = make_net(input_shape=(height, width, channel), output_shape=(num_class))
    data  = tf.get_default_graph().get_tensor_by_name('input:0')
    label = tf.placeholder(dtype=tf.int32, shape=[None])
    prob  = tf.nn.softmax(logit)
    loss  = cross_entropy(logit, label)
    metric = accuracy(prob, label)

    features = tf.get_default_graph().get_tensor_by_name("preprocess/add_3:0")

    # start testing here ------------------------------------------------
    batch_size=16

    sess = tf.InteractiveSession()
    with sess.as_default():
        saver  = tf.train.Saver()
        saver.restore(sess, out_dir + '/check_points/final.ckpt')

        idx = list(np.where(test_labels == 38)[0])
        start = idx[5]
        end = start + batch_size if start + batch_size <= num_test else num_test
        batch_datas  = test_images[start:end]
        batch_labels = test_labels[start:end]

        for n in range(1,batch_size):
            batch_datas[n]= perturb(batch_datas[0], keep=0, angle_limit=0, scale_limit=0, translate_limit=0, distort_limit=0,
                    illumin_limit=0.7)

        fd = {data: batch_datas, label: batch_labels, IS_TRAIN_PHASE: False}
        fs, test_probs, test_loss, test_acc = sess.run([features, prob, loss, metric], feed_dict=fd)


        results_image = 255. * np.ones(shape=(batch_size*32, 32, 3), dtype=np.float32)
        channels =np.zeros(shape=(batch_size*32,8*32))
        for n in range(batch_size):
            insert_subimage(results_image, batch_datas[n], n*32, 0)

            f = fs[n]
            for i in range(7):
                f_i = f[:,:,i]
                channels[n*32:(n+1)*32,i*32:(i+1)*32]=f_i


        #imshow('results_image', results_image, 42)
        #cv2.waitKey(0)
        cv2.imwrite(out_dir + '/preprocess_input4.jpg', cv2.cvtColor(results_image, cv2.COLOR_BGR2RGB))
        plt.matshow(channels, cmap='gray')
        plt.show()
        pass

def run_extra_test():

    # output dir, etc
    out_dir  = '/root/share/out/udacity/08'
    data_dir = '/root/share/project/udacity/project2_01/data'
    # ----------------------------------------
    # data_dir = '/root/share/project/udacity/project2_01/data'
    # classname_file = data_dir + '/signnames.csv'
    # classnames = []
    # with open(classname_file) as _f:
    #     rows = csv.reader(_f, delimiter=',')
    #     next(rows, None)  # skip the headers
    #     for i, row in enumerate(rows):
    #         assert (i == int(row[0]))
    #         classnames.append(row[1])
    #
    classnames, train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_data()
    height, width, channel = 32, 32, 3
    num_class = 43

    #prepare data ----------------------------------------------
    test_files = ['0004.jpg',  # normal
                  '0000.jpg',  # normal
                  '0007.jpg',  # occluded with snow
                  '0006.jpg',  # small
                  '0005.jpg',  # not in class
                  ]
    test_rois = [(54, 180, 125, 260), (160, 430, 207, 469), (181, 32, 321, 142), (226, 65, 242, 78),
                 (388, 408, 700, 676)]
    num = len(test_files)
    print('num=%d' % num)

    # crop roi to 32x32
    results_image  = 255. * np.ones(shape=(1 * height, num * width, channel), dtype=np.float32)
    results_image1 = 255. * np.ones(shape=(1 * 320, num * 320, channel), dtype=np.float32)
    crops = np.zeros(shape=(num, height, width, channel), dtype=np.float32)
    for n in range(num):
        img = cv2.imread(data_dir + '/extra/' + test_files[n], 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        x1, y1, x2, y2 = test_rois[n]
        crop = cv2.resize(img[y1:y2, x1:x2, :], (0, 0), fx=32. / (x2 - x1), fy=32. / (y2 - y1),
                          interpolation=cv2.INTER_CUBIC)

        crop = np.clip(crop, 0, 255)
        crops[n] = crop
        insert_subimage(results_image, crop, 0, n * width)

        # mak roi and show
        H, W, C = img.shape
        S = max(H, W)
        f = 320. / S
        norm_img = cv2.resize(img, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(norm_img, (round(f * x1), round(f * y1)), (round(f * x2), round(f * y2)), (255, 255, 0), 3)
        insert_subimage(results_image1, norm_img, 0, n * 320)
        imshow('crop', crop)
        imshow('img', img)
        cv2.waitKey(1)

    cv2.imwrite(data_dir + '/extra/' + 'crops.jpg', cv2.cvtColor(results_image, cv2.COLOR_BGR2RGB))
    # imshow('results_image', results_image)
    # imshow('results_image1', results_image1)
    # cv2.waitKey(1)



    #net  -----------------------------------------------
    logit = make_net(input_shape=(height, width, channel), output_shape=(num_class))

    # data  = tf.placeholder(dtype=tf.float32, shape=[None, height, width, channel])
    data = tf.get_default_graph().get_tensor_by_name('input:0')
    label = tf.placeholder(dtype=tf.float32, shape=[None])
    prob = tf.nn.softmax(logit)
    # top_k = tf.nn.top_k(prob, k=5)


    # start testing here ------------------------------------------------


    # sess = tf.InteractiveSession()
    # with sess.as_default():
    #     print('** test on extra **')
    #
    #     # saver = tf.train.Saver()
    #     # saver.restore(sess, out_dir + '/check_points/final.ckpt')
    #     # fd = {data: crops, IS_TRAIN_PHASE: False}
    #     # test_prob = sess.run(prob, feed_dict=fd)

    test_prob = np.random.uniform(size=(num,num_class))


    # show results ------------------

    f=8
    results_image = 255. * np.ones(shape=(5*(f*height + 30), 6*f*width, channel), dtype=np.float32)

    for n in range(num):
        print('n=%d:' % n)
        crop = crops[n]
        #crop = cv2.resize(crop, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_NN)
        crop = crop.repeat(f, axis=0).repeat(f, axis=1)
        insert_subimage(results_image, crop, n * (f*height + 30), 0)

        p = test_prob[n]
        idx = np.argsort(p)[::-1]
        for k in range(5):
            c = int(idx[k])
            label_image = get_label_image(c)
            #label_image = cv2.resize(label_image, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_NN)
            label_image = label_image.repeat(f, axis=0).repeat(f, axis=1)
            insert_subimage(results_image, label_image, n * (f*height + 30), (k + 1) * f*width)


            print('\ttop%d: %f  %02d:%s' % (k, p[c], c, classnames[c]))
            cv2.putText(results_image, 'top%d: %f' % (k, p[c]), (5+(k + 1) * f*width, (n+1) * (f*height + 30)-27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(results_image, '%02d:%s%s' % (c, classnames[c][0:20], '...' if len(classnames[c])>20 else ''), (5+(k + 1) * f*width, (n+1) * (f*height + 30)-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    imshow('results_image', results_image)
    cv2.waitKey(0)
# show which samples are wrong
def run_extra_test_0():

    # output dir, etc
    out_dir = '/root/share/docs/git/hengck23-udacity/udacity-driverless-car-nd-p2/submission(notebook+html)/002/out'

    # data -------------------------------------------------------------------------
    print('read data:\n')
    classnames, train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_data()

    num_class = 43
    _, height, width, channel = train_images.shape
    num_train = len(train_images)
    num_valid = len(valid_images)
    num_test  = len(test_images)

    # net  -----------------------------------------------
    logit = make_net(input_shape=(height, width, channel), output_shape=(num_class))
    data  = tf.get_default_graph().get_tensor_by_name('input:0')
    label = tf.placeholder(dtype=tf.int32, shape=[None])
    prob  = tf.nn.softmax(logit)
    loss  = cross_entropy(logit, label)
    metric = accuracy(prob, label)



    # start testing here ------------------------------------------------

    sess = tf.InteractiveSession()
    with sess.as_default():
        saver  = tf.train.Saver()
        saver.restore(sess, out_dir + '/check_points/final.ckpt')

        # shuffle and test using difference batch size (just make sure there is not bug!)
        print('** evaluation on test set **')
        for i in range(2): #10
            images,  labels = shuffle_data(test_images, test_labels)
            batch_size =  np.random.randint(1, 256)
            test_loss, test_acc = test_net(images, labels, batch_size, data, label, loss, metric, sess)
            print('  %d,   batch_size=%3d  : %f    (%f)' % (i,batch_size,test_loss, test_acc))
        print('')

        #---------------------------------------------------------

        num_test   = len(test_images)
        test_probs = np.zeros(shape=(num_test,num_class))

        all_loss = 0
        all_acc = 0
        all = 0
        for n in range(0, num_test, batch_size):
            # print('\r  evaluating .... %d/%d' % (n, num), end='', flush=True)
            start = n
            end = start + batch_size if start + batch_size <= num_test else num_test
            batch_datas  = test_images[start:end]
            batch_labels = test_labels[start:end]

            fd = {data: batch_datas, label: batch_labels, IS_TRAIN_PHASE: False}
            test_probs[start:end], test_loss, test_acc = sess.run([prob, loss, metric], feed_dict=fd)

            a = end - start
            all += a
            all_loss += a * test_loss
            all_acc += a * test_acc
        assert (all == num_test)
        loss = all_loss / all
        acc = all_acc / all

        print(' ** final ** : %f    (%f)' % (loss, acc))
        print('')

        idx = np.argsort(test_probs,axis=1)[:, ::-1]
        top_labels = idx[:,0:5]
        top_probs  = np.zeros(shape=(num_test,5))
        true_probs = np.zeros(shape=(num_test))
        for n in range(num_test):
            top_probs [n]  = test_probs[n,top_labels [n]]
            true_probs[n]  = test_probs[n,test_labels[n]]

        top_label = top_labels[:,0]
        correct   = top_label == test_labels
        wrong     = np.invert(correct)
        wrong_idx = np.where(wrong)[0]

        # check
        print(wrong_idx)
        print('')
        correct_rate = np.sum(correct) / float(num_test)
        print('correct=%f' % correct_rate)
        print('')

        wrong_idx = wrong_idx[np.argsort(true_probs[wrong_idx])]
        print(wrong_idx)
        print('')

        #show wrong --------------------------------------------------------------------------------------------
        label_images=get_all_label_images() #get_all_label_images()
        mean_images =get_all_train_mean_images()

        num_wrong = len(wrong_idx)
        results_image = 255. * np.ones(shape=(num_wrong * (height+20), 13* width +2, channel), dtype=np.float32)
        for n in range(num_wrong):
            m = wrong_idx[n]
            c_hat = test_labels[m]
            image = test_images[m]

            y = n*(height+20)
            y_text= y+height+15
            insert_subimage(results_image, image, y, 0)
            insert_subimage(results_image, mean_images [c_hat], y, 1+ width)
            insert_subimage(results_image, label_images[c_hat], y, 1+ width*2)
            cv2.putText(results_image, '%0.4f' % (test_probs[m, c_hat]), (1+ width, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2 )

            for k in range(5):
                c = top_labels[m,k]
                insert_subimage(results_image, mean_images [c], y, 2+(3+k*2  )*width)
                insert_subimage(results_image, label_images[c], y, 2+(3+k*2+1)*width)
                cv2.putText(results_image, '%0.4f' % (test_probs[m, c]), (2+(3+k*2  )*width, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )

        cv2.imwrite(out_dir+'/wrong_test.jpg',
                    cv2.cvtColor(results_image, cv2.COLOR_BGR2RGB))

        imshow('results_image', results_image,1)
        cv2.waitKey(1)

        #show expcted probability --------------------------------------------------------------------------------------------
        expected_probs = np.zeros(shape=(num_class, num_class))
        counts = np.zeros(shape=(num_class))

        for n in range(num_test):
            c = test_labels[n]
            expected_probs[c] += test_probs[n]
            counts[c] += 1
        expected_probs =expected_probs/ (np.array([counts,]*num_class).transpose()+1e-5)
        print (expected_probs)
        plt.imshow(expected_probs,  cmap=plt.get_cmap('gray'))
        #plt.show()

        results_image = 255. * np.ones(shape=(num_class * height, num_class * width, channel), dtype=np.float32)
        for c in range(num_class):
            for k in range(num_class):
                s = expected_probs[c,k]
                #s = (100+np.clip(17*math.log(expected_probs[c,k]+0.001),-100,0))/100.

                image = label_images[k] * s
                insert_subimage(results_image, image, c*height, k* width)


        cv2.imwrite(out_dir+'/expect_test.jpg',  cv2.cvtColor(results_image, cv2.COLOR_BGR2RGB))
        #cv2.imwrite(out_dir+'/expect_test.jpg',  expected_probs)  #log_expect_test

        imshow('results_image', results_image,1)
        cv2.waitKey(0)
        pass

#################################################################################33




