import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import process_data # loads and pre-processes the training and ground truth data
from sklearn.model_selection import train_test_split

tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    vgg_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input, vgg_keep_prob, vgg_layer3, vgg_layer4, vgg_layer7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    fcn = tf.layers.conv2d(vgg_layer7_out, 4096, 1, strides=1, padding='same', use_bias=False, name='conv_1',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    fcn = tf.layers.batch_normalization(fcn, training=True, name='batch_1')

    x = tf.layers.conv2d_transpose(fcn, 512, 3, strides=2, padding='same', use_bias=False, name='conv_2',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    x = tf.layers.batch_normalization(x, training=True, name='batch_2')
    x = tf.layers.conv2d(x, 512, 1, strides=1, padding='same', use_bias=False, name='conv_3',
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    x = tf.layers.batch_normalization(x, training=True, name='batch_3')
    x = tf.add(x, vgg_layer4_out)

    x = tf.layers.conv2d(x, 512, 1, strides=1, padding='same', use_bias=False, name='conv_4',
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    x = tf.layers.batch_normalization(x, training=True, name='batch_4')
    x = tf.layers.conv2d_transpose(x, 256, 7, strides=2, padding='same', use_bias=False, name='conv_5',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    x = tf.layers.batch_normalization(x, training=True, name='batch_5')
    x = tf.layers.conv2d(x, 256, 1, strides=1, padding='same', use_bias=False, name='conv_6',
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    x = tf.layers.batch_normalization(x, training=True, name='batch_6')
    x = tf.add(x, vgg_layer3_out)

    x = tf.layers.conv2d(x, 64, 1, strides=1, padding='same', use_bias=False, name='conv_7',
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    x = tf.layers.batch_normalization(x, training=True, name='batch_7')
    x = tf.layers.conv2d_transpose(x, num_classes, 16, strides=8, padding='same', name='conv_8',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    x = tf.layers.batch_normalization(x, training=True, name='batch_8')

    x = tf.layers.conv2d(x, num_classes, 1, strides=1, padding='same', use_bias=False, name='conv_9',
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    return x
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    loss = tf.reduce_mean(cross_entropy_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return logits, optimizer, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.local_variables_initializer())

    for epoch in range(epochs):
        batch = 0
        for images, labels in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: images, correct_label: labels, keep_prob: 1.0, learning_rate:0.00025})
            batch += 1
            print('Epoch {:>2}, step: {}, loss: {}  '.format(epoch + 1, batch, loss))

tests.test_train_nn(train_nn)


def run():
    """
    Collects the data and pre-processes the images, loads the pre-trained vgg-model and trains the semantic
    segmentation decoder network on the data and test the trained model.
    :return: Nothing
    """
    epochs = 3
    batch_size = 10
    num_classes = 3
    image_shape = (160, 576)

    data_dir = './data/'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    print('Collecting Data')
    # pre-process the images to create more training data
    img_list = process_data.getData(image_shape)
    train_data, val_data = train_test_split(img_list, test_size=0.0)
    print('Finished collecting Data')

    # Download the pre-trained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # create the NN placeholders
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    shape = (None,) + image_shape + (num_classes,)
    correct_label = tf.placeholder(tf.float32, shape)
    #keep_prob = tf.placeholder(tf.float32)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(train_data, image_shape, num_classes)

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)

        # load the pre-trained vgg network
        nn_last_layer = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)

        # setup the NN training optimizer
        logits, optimizer, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        saver = tf.train.Saver()
        save_model_path = './model/semantic_segmentation_model.ckpt'

        # Train the NN
        train_nn(sess, epochs, batch_size, get_batches_fn, optimizer, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # save the trained model
        saver.save(sess, save_model_path)
        print("Model saved")

        # Test the model with the test images
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
