import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.python.training import saver

# cifar10 dataset 
CLASS = 10
TRAIN_SIZE = 50000

# Output format
SPACE = 15
SUMMARIZE = 10

#Hyperparameter
BATCH_SIZE = 10
EPOCH = 100
LR = 0.001
ITER = int(TRAIN_SIZE/BATCH_SIZE+0.5)

# define placeholder 
xp = tf.placeholder(tf.float32, shape = (None, None, None, 3))
yp = tf.placeholder(tf.float32, shape = (None, CLASS))

# define network
def network():
    # get input from placeholder
    x = xp
    y = yp
    # resize to 224*224 for classification
    x = tf.image.resize_images(x, (224, 224))
    # outputs from vgg16
    predictions, end_points = nets.vgg.vgg_16(x, num_classes = 10, is_training = True, dropout_keep_prob = 0.5)
    # print result
    predictions = tf.Print(predictions, [tf.argmax(predictions, axis = 1)],message = '{:{}}: '.format('Prediction', SPACE),summarize = SUMMARIZE)
    y = tf.Print(y, [tf.argmax(y, axis = 1)], message = '{:{}}: '.format('GT', SPACE), summarize = SUMMARIZE)
    # define loss function
    loss = slim.losses.softmax_cross_entropy(predictions, y)
    # define optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = LR).minimize(loss)
    
    return loss, optimizer

# shuffle data
def shuffle_unison(x,y):
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)

# load data from cifar10
def load_data():
    # load data from cifar10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # to one hot
    y_train_ = np.zeros([TRAIN_SIZE, CLASS])
    y_train_[np.arange(TRAIN_SIZE), y_train[:, 0]] = 1
    
    return x_train, y_train_, x_test, y_test

def train_network(x_train,y_train):
    with tf.Session() as sess:
        # get network
        loss, optimizer = network()
        # initial weight
        init = tf.global_variables_initializer()
        sess.run(init)
        # load weight exclude fc8
        variables_to_restore = slim.get_variables_to_restore(exclude=["vgg_16/fc8"])
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, './models/vgg_16.ckpt')
        # training
        for i in range(EPOCH):
            # shuffle data
            shuffle_unison(x_train, y_train)
            # split for batch
            x_train = np.array_split(x_train, ITER)
            y_train = np.array_split(y_train, ITER)
            for j in range(ITER):
                optimizer_, loss_ = sess.run([optimizer, loss], feed_dict={xp: x_train[j], yp: y_train[j]})
            print ('{:{}}: {}'.format('Epoch', SPACE, i))
            print ('{:{}}: {}'.format('Loss', SPACE, loss_))

def main():
    # load dataset from cifar10
    x_train, y_train, x_test, y_test = load_data()
    # train network
    train_network(x_train, y_train)

if __name__=='__main__':
    main()

