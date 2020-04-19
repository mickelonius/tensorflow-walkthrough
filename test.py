import tensorflow as tf

if __name__ == '__main__':
    print('tf.test.is_gpu_available()', tf.test.is_gpu_available())
    print('tf.config.list_physical_devices(\'GPU\')', tf.config.list_physical_devices('GPU'))
