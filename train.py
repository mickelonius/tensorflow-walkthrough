import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
import tensorflow as tf
import numpy as np
from tensorflow import keras

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

# 3x3 convolution
def conv3x3(channels, stride=1, kernel=(3, 3)):
    return keras.layers.Conv2D(channels,
                               kernel,
                               strides=stride,
                               padding='same',
                               use_bias=False,
                               kernel_initializer=tf.random_normal_initializer()
                               )

class ResnetBlock(keras.Model):

    def __init__(self, channels, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()

        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path

        self.conv1 = conv3x3(channels, strides)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = conv3x3(channels)
        self.bn2 = keras.layers.BatchNormalization()

        if residual_path:
            self.down_conv = conv3x3(channels, strides, kernel=(1, 1))
            self.down_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        residual = inputs

        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        # this module can be added into self.
        # however, module in for can not be added.
        if self.residual_path:
            residual = self.down_bn(inputs, training=training)
            residual = tf.nn.relu(residual)
            residual = self.down_conv(residual)

        x = x + residual
        return x

class ResNet(keras.Model):

    def __init__(self, block_list, num_classes, initial_filters=16, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.num_blocks = len(block_list)
        self.block_list = block_list

        self.in_channels = initial_filters
        self.out_channels = initial_filters
        self.conv_initial = conv3x3(self.out_channels)

        self.blocks = keras.models.Sequential(name='dynamic-blocks')

        # build all the blocks
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):

                if block_id != 0 and layer_id == 0:
                    block = ResnetBlock(self.out_channels, strides=2, residual_path=True)
                else:
                    if self.in_channels != self.out_channels:
                        residual_path = True
                    else:
                        residual_path = False
                    block = ResnetBlock(self.out_channels, residual_path=residual_path)

                self.in_channels = self.out_channels

                self.blocks.add(block)

            self.out_channels *= 2

        self.final_bn = keras.layers.BatchNormalization()
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes) #, activation='softmax')

    def call(self, inputs, training=None):

        out = self.conv_initial(inputs)

        out = self.blocks(out, training=training)

        out = self.final_bn(out, training=training)
        out = tf.nn.relu(out)

        out = self.avg_pool(out)
        out = self.fc(out)


        return out

def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.

    # [b, 28, 28] => [b, 28, 28, 1]
    x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)

    # one hot encode the labels. convert back to numpy as we cannot use a combination of numpy
    # and tensors as input to keras
    y_train_ohe = tf.one_hot(y_train, depth=10).numpy()
    y_test_ohe = tf.one_hot(y_test, depth=10).numpy()

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    x_new = x_test[:3]
    np.save("my_mnist_tests.npy", x_new)

    num_classes = 10
    batch_size = 32
    epochs = 1



    # build model and optimizer
    model = ResNet([2, 2, 2], num_classes)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.build(input_shape=(None, 28, 28, 1))
    print("Number of variables in the model :", len(model.variables))
    model.summary()

    # train
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    # Weights only: when you restore this, it will throw up
    # bunch of warnings that optimizer's state for various parts of the
    # model are "unresolved"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # Train model
    model.fit(x_train, y_train_ohe,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[cp_callback],
              validation_data=(x_test, y_test_ohe),
              verbose=1)

    # Save as TFSavedModel
    model_version = "0001"
    model_name = "my_mnist_model"
    model_path = os.path.join(model_name, model_version)

    tf.saved_model.save(model, model_path)

    for root, dirs, files in os.walk(model_name):
        indent = '    ' * root.count(os.sep)
        print('{}{}/'.format(indent, os.path.basename(root)))
        for filename in files:
            print('{}{}'.format(indent + '    ', filename))



    # Evaluate on test set
    # Create a new model instance
    eval_model = ResNet([2, 2, 2], num_classes)

    # Save the weights using the `checkpoint_path` format
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print('Loading saved weights')
    eval_model.load_weights(latest)

    print('Compiling saved model')
    eval_model.compile(optimizer=keras.optimizers.Adam(0.001),
                       loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])

    print('Evaluating saved model')
    scores = eval_model.evaluate(x_test, y_test_ohe, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)


    print('Use saved model object')
    saved_model = tf.saved_model.load(model_path)
    x_hat = saved_model(tf.constant(x_new, dtype=tf.float32))
    print(x_hat)
    print(tf.keras.activations.softmax(x_hat, axis=-1))

    print('Use saved Keras model object')
    keras_model = tf.keras.models.load_model(model_path)
    x_hat = keras_model.predict(tf.constant(x_new, dtype=tf.float32))
    print(x_hat)
    print(tf.keras.activations.softmax(x_hat, axis=-1))

if __name__ == '__main__':
    main()