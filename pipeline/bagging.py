import numpy as np
import tensorflow as tf
import cv2
import os
import glob
import time
from math import ceil
import csv
import argparse

big_x_reconst_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, None))
big_ground_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, None))
img_placeholder = tf.placeholder(tf.float32, shape=(None, None))

big_product = tf.multiply(big_x_reconst_placeholder, big_ground_placeholder)
big_product_reduced_test = tf.reduce_sum(big_product)
big_accuracy = (tf.reduce_sum(big_product) / (
        tf.reduce_sum(tf.square(big_ground_placeholder)) + tf.reduce_sum(
    tf.square(big_x_reconst_placeholder)) - tf.reduce_sum(
    big_product)))

new_big_accuracy = (tf.reduce_sum(big_product) / (
        tf.reduce_sum(big_ground_placeholder) + tf.reduce_sum(big_x_reconst_placeholder) - tf.reduce_sum(big_product)))



data_root = '/homes/my_d/data/ISIC_dataset/Task_1/'
splitsdic = {
    'training_2017': data_root + "splits/2017_training.csv",
    'validation_2017': data_root + "splits/2017_validation.csv",
    'test_2017': data_root + "splits/2017_test.csv",
    'training_2018': data_root + "splits/2018_training.csv",
    'validation_2018': data_root + "splits/2018_validation.csv",
    'test_2018': data_root + "splits/2018_test.csv",
    'from_API': data_root + "splits/from_API.csv",
    'dermoscopic': data_root + "splits/dermoscopic.csv",
    'clinic': data_root + "splits/clinic.csv",
    'dermoscopic_wmasks': data_root + "splits/dermoscopic_with_mask.csv",
    'dermot_2017': data_root + "splits/dermoscopic_train_2017.csv",
    'mtap': data_root + "splits/dermo_MTAP.csv",

}

if not os.path.exists(FILES_PATH):
    os.makedirs(FILES_PATH)
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)


def write_on_file(epoch, training_cost, validation_acc, new_big_acc, old_big_acc, tm, filename):
    ffname = FILES_PATH + filename
    with open(ffname, 'a+') as f:
        f.write("E:" + str(epoch) + " | Time: " + str(tm) +
                " | Training_acc: " + str(1 - training_cost) +
                " | validation_acc: " + str(1 - validation_acc) +
                " | new_big_acc: " + str(new_big_acc) + " | old_big_acc: " + str(old_big_acc) + "\n")


def get_images_from_list(i_list):
    imgs = []
    grnds = []
    for i in i_list:
        imgs.append(data_root + "images/ISIC_" + str(i) + ".jpg")
        grnds.append(data_root + "ground_truth/ISIC_" + str(i) + "_segmentation.png")
    return imgs, grnds


def read_csv(csv_filename):
    split_list = []
    with open(splitsdic.get(csv_filename)) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            split_list.append(row[0])

    return split_list


def read_images(r_batch_size, r_csv_name, total_images=None):
    start_time = time.time()
    X_total_imgs = []
    Y_total_imgs = []
    y_name = []
    csv_list = read_csv(r_csv_name)
    if not total_images:
        total_images = len(csv_list)
    else:
        csv_list = csv_list[:total_images]
    for minibatch_number in range(ceil(total_images / r_batch_size)):
        X_images, Y_images = get_images_from_list(
            csv_list[minibatch_number * r_batch_size:(minibatch_number + 1) * r_batch_size])

        for i in range(len(X_images)):
            # im = cv2.imread(X_images[i])
            gr = cv2.imread(Y_images[i], cv2.IMREAD_GRAYSCALE)
            # resizing
            # resized_image = cv2.resize(im, (img_width[0], img_height[0]), interpolation=cv2.INTER_AREA)
            resized_ground = cv2.resize(gr, (img_width[0], img_height[0]), interpolation=cv2.INTER_AREA)
            # X_total_imgs.append(resized_image)
            Y_total_imgs.append(resized_ground)
            y_name.append(Y_images[i])

        print("time: " + str(time.time() - start_time))
        print("length X: " + str(len(X_total_imgs)))
        print("length Y: " + str(len(Y_total_imgs)))
    return X_total_imgs, Y_total_imgs, y_name


def get_predictions(name):
    reconstr = []
    paths = glob.glob(PREDS_PATH + '*')
    for path in paths:
        gr = cv2.imread(glob.glob(path + '/*' + name + '*')[0], cv2.IMREAD_GRAYSCALE)
        reconstr.append(gr)
    return reconstr


def single_image_bagging(reconstructions):
    img_placeholder = np.divide(reconstructions[0], 255.0)
    for i in range(len(reconstructions) - 1):
        tmp_img = np.divide(reconstructions[i + 1], 255.0)
        img_placeholder = np.add(img_placeholder, tmp_img)
    final_image = np.divide(img_placeholder, float(len(reconstructions)))

    return final_image


def postProcessing(reconstr, thresh_value):

    gray_im = reconstr
    gray_im = gray_im.astype(np.float32)
    first_ret_1, first_thresh_1 = cv2.threshold(gray_im, 0.8, 1, cv2.THRESH_BINARY)
    _, simple_pp = cv2.threshold(gray_im, thresh_value, 1, cv2.THRESH_BINARY)

    first_thresh_2 = np.array(first_thresh_1, dtype=np.uint8)
    output_1 = cv2.connectedComponentsWithStats(first_thresh_2)
    max_area = 0
    best_index = 1

    if output_1[0] == 0:
        return simple_pp

    # find biggest object
    if output_1[0] > 2:
        for j in range(output_1[0] - 1):
            if output_1[2][j + 1][4] > max_area:
                max_area = output_1[2][j + 1][4]
                best_index = j + 1

    # find the object with "best center" inside
    if output_1[0] > 1:
        center = output_1[3][best_index]
        second_ret_1, second_thresh_1 = cv2.threshold(gray_im, thresh_value, 1, cv2.THRESH_BINARY)
        second_thresh_1 = np.array(second_thresh_1, dtype=np.uint8)
        output_2 = cv2.connectedComponentsWithStats(second_thresh_1)
        best_index = 1
        for k in range(output_2[0] - 1):
            if output_1[0] > 1:
                if center[0] > output_2[2][k + 1][0] and center[0] < output_2[2][k + 1][0] + output_2[2][k + 1][2] \
                        and center[1] > output_2[2][k + 1][1] and center[1] < output_2[2][k + 1][1] + \
                        output_2[2][k + 1][3]:
                    best_index = k + 1

        # # create image
        processed_image = (output_2[1] == best_index).astype(float)
    else:
        processed_image = simple_pp

    return processed_image


def single_image_big_jaccard(ground_truth, processed, sess, glbl_counter, thresh_value):
    gr = cv2.imread(ground_truth, cv2.IMREAD_GRAYSCALE)
    final_gr = np.divide(gr, 255.0)


    big_x_1 = cv2.resize(processed, (gr.shape[1], gr.shape[0]),
                         cv2.INTER_CUBIC)

    big_x_1 = big_x_1.astype(np.float32)
    _, big_processed = cv2.threshold(big_x_1, 0.49, 1, cv2.THRESH_BINARY)

    big_x_1 = np.expand_dims(big_x_1, -1)
    big_x_1 = np.expand_dims(big_x_1, 0)
    big_processed_1 = np.expand_dims(big_processed, -1)
    big_processed_1 = np.expand_dims(big_processed_1, 0)
    single_Y_minibatch = np.expand_dims(final_gr, -1)
    single_Y_minibatch = np.expand_dims(single_Y_minibatch, 0)

    acc_1 = sess.run(new_big_accuracy,
                     feed_dict={
                         big_x_reconst_placeholder: big_processed_1,
                         big_ground_placeholder: single_Y_minibatch
                     })

    acc_2 = sess.run(big_accuracy,
                     feed_dict={
                         big_x_reconst_placeholder: big_processed_1,
                         big_ground_placeholder: single_Y_minibatch
                     })


    print(ground_truth + ': ' + str(acc_1))
    return acc_1, acc_2


def test_with_bagging(thresh_value, predictions_path, write_flag=False):
    accuracy_1 = []
    t_accuracy = []
    accuracy_2 = []
    glbl_counter = 0
    histograph = np.zeros(20)
    batch_counter = test_start
    x_total_imgs, y_total_imgs, y_names = read_images(150, 'test_2017')
    start_time = time.time()
    test_Y_images, test_Y_names = y_total_imgs, y_names


    with tf.Session() as sess:

        for i in range(len(y_total_imgs)):
            reconstr = get_predictions(os.path.basename(test_Y_names[i])[:-17])
            glbl_counter += 1

            final_image = single_image_bagging(reconstr)



            processed = postProcessing(final_image, thresh_value)

            if write_flag:
                # save processed prediction as an image
                cv2.imwrite("/homes/my_d/data/TEST_PREDICTIONS/" + predictions_path + '/' + os.path.basename(test_Y_names[i])[:-17]
                            + "_1.png", np.multiply(final_image, 255.0))
                cv2.imwrite("/homes/my_d/data/TEST_PREDICTIONS/" + predictions_path + '/' + os.path.basename(test_Y_names[i])[:-17]
                            + "_2.png", np.multiply(processed, 255.0))

            acc_1, acc_2 = single_image_big_jaccard(test_Y_names[i], processed, sess, glbl_counter, thresh_value)
            accuracy_1.append(acc_1)
            accuracy_2.append(acc_2)
            if acc_1 < 0.65:
                t_accuracy.append(0)
            else:
                t_accuracy.append(acc_1)

            value_i = int(acc_1 // 0.05)
            histograph[value_i] += 1


    print("final accuracy_1: " + str(np.mean(accuracy_1)) + "; final accuracy_2:" + str(
        np.mean(accuracy_2)) + "; thresholded accuracy:" + str(np.mean(t_accuracy)))

    for i in range(20):
        print(str(i * 0.05) + ": " + str(histograph[i]))

    print("threshoding value: " + str(thresh_value))
    print(predictions_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--path', default=None, help='directory to save results')
    parser.add_argument('--thresh', type=float, default=0.5, help='threshold to use during the last PostProcessing step')

    opt = parser.parse_args()
    if opt.path is not None and not os.path.exists("/homes/my_d/data/" + opt.path):
        os.makedirs("/homes/my_d/data/" + opt.path)

    PREDS_PATH = "/homes/my_d/data/TEST_PREDICTIONS/to_bag/"
    test_with_bagging(opt.thresh, opt.path, opt.path is not None)

