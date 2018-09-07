
" Input and target placeholder functions defined below"

fc_count = 0  # count of fully connected layers. Do not remove.


def input_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 784],
                          name="image_input")


def target_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 10],
                          name="image_target_onehot")


" One Layer Neural Network " 

def onelayer(X, Y, layersize=10):
    """
    Create a Tensorflow model for logistic regression (i.e. single layer NN)

    :param X: The input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned  (variables in the
    python sense, not in the Tensorflow sense, although some may be
    Tensorflow variables). They must be returned in the following order.
        w: Connection weights
        b: Biases
        logits: The input to the activation function
        preds: The output of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    # Create variables for weight and biases
    w = tf.Variable(tf.random_normal([int(X.shape[1]),layersize],stddev=0.1))
    b = tf.Variable(tf.zeros([layersize]))

    # matrix multiplication to creates logits
    logits = tf.matmul(X, w) + b

    # turn logits into probability
    preds = tf.nn.softmax(logits)

    # calculate batch cross-entropy loss for each image
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels =Y)

    # calculate batch loss for average cross entropy
    batch_loss = tf.reduce_mean(batch_xentropy)

    return w, b, logits, preds, batch_xentropy, batch_loss


" Two Layer Neural Network " 
" (one hidden layer, size determined within function) "
def twolayer(X, Y, hiddensize=30, outputsize=10):
    """
    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        W1: Connection weights for the first layer
        b1: Biases for the first layer
        W2: Connection weights for the second layer
        b2: Biases for the second layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    # first layer
    # Create variables for weights and biases (zeros)
    # w1 = tf.Variable(tf.random_normal([int(X.shape[1]), hiddensize]))
    w1 = tf.Variable(tf.random_normal([int(X.shape[1]),hiddensize],stddev = 0.1))
    b1 = tf.Variable(tf.random_normal([hiddensize], stddev=0.1))

    # Create logits for hidden network
    pre_logits = tf.matmul(X, w1) + b1
    
    # Use relu function
    relu_logits = tf.nn.relu(pre_logits)

    # second layer
    # Create variables for weights and biases
    w2 = tf.Variable(tf.zeros([hiddensize, outputsize]))
    b2 = tf.Variable(tf.zeros([outputsize]))

    # Create logits for second layer
    logits = tf.matmul(relu_logits,w2) + b2

    # Create preds for second layer
    preds = tf.nn.softmax(logits)

    # Create batch cross entropy loss
    batch_xentropy = batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels =Y)

    # Create batch loss for second layer
    batch_loss = tf.reduce_mean(batch_xentropy)


    return w1, b1, w2, b2, logits, preds, batch_xentropy, batch_loss


" Two Layer Convulutional Neural Network " 

def convnet(X, Y, convlayer_sizes=[10, 10], \
            filter_shape=[3, 3], outputsize=10, padding="same"):
    """
    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        conv1: A convolutional layer of convlayer_sizes[0] filters of shape filter_shape
        conv2: A convolutional layer of convlayer_sizes[1] filters of shape filter_shape
        w: Connection weights for final layer
        b: biases for final layer
        logits: The inputs to the activation functions
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch

    """
    # layer 1 convolutional
    conv1 = tf.layers.conv2d(inputs=X, filters=convlayer_sizes[0], kernel_size=filter_shape, padding=padding, activation=tf.nn.relu)
    
    # layer 2 convulational
    conv2 = tf.layers.conv2d(inputs=conv1, filters=convlayer_sizes[1], kernel_size=filter_shape, padding=padding, activation=tf.nn.relu)

    # connected layer
    flattened = tf.reshape(conv2, [-1, int(conv2.shape[1]) * int(conv2.shape[2]) * int(conv2.shape[3])])
    
    # weights
    w = tf.Variable(tf.ones([int(flattened.shape[1]),outputsize]))

    # biases
    b = tf.Variable(tf.zeros([outputsize]))

    # logits
    logits = tf.matmul(flattened, w) +b
    
    # preds 
    preds = tf.nn.softmax(logits)

    # batch cross entropy
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)

    #batch loss
    batch_loss = tf.reduce_mean(batch_xentropy)

    return conv1, conv2, w, b, logits, preds, batch_xentropy, batch_loss


def train_step(sess, batch, X, Y, train_op, loss_op, summaries_op):
    """
    Run one step of training.
    :param sess: the current session
    :param batch: holds the inputs and target outputs for the current minibatch
    batch[0] - array of shape [minibatch_size, 784] with each row holding the
    input images
    batch[1] - array of shape [minibatch_size, 10] with each row holding the
    one-hot encoded targets
    :param X: the input placeholder
    :param Y: the output target placeholder
    :param train_op: the tensorflow operation that will run one step of training
    :param loss_op: the tensorflow operation that will return the loss of your
    model on the batch input/output

    :return: a 3-tuple: train_op_result, loss, summary
    which are the results of running the train_op, loss_op and summaries_op
    respectively.
    """
    train_result, loss, summary = \
        sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1]})
    return train_result, loss, summary
