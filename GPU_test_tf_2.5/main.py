import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
import numpy as np
import GPUtil
import time
import argparse
import sys
import warnings
warnings.filterwarnings(action='ignore')

# TO print GPU Stat
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print()        
        GPUs = GPUtil.getGPUs()
        for GPU in GPUs:
            print()
            print("GPU ID : {}, Memory Free : {},  Memory Used : {}, Memory Temp : {}"
                  .format(GPU.id, GPU.memoryFree,GPU.memoryUsed,GPU.temperature))
            print()
            
            
def init_gpus(soft_device_placement=True, log_device_placement=False, create_virtual_devices=False, memory_limit=11):

    tf.config.set_soft_device_placement(soft_device_placement)    
    tf.debugging.set_log_device_placement(log_device_placement)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # If there is only one GPU, create two logical virtual devices for developing
        # on a machine with only one GPU installed
        try:
            # Create 2 virtual GPUs on each physical GPU with the given memory_limit GPU memory
            if create_virtual_devices and len(gpus) == 1:
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit*10000),
                         tf.config.experimental.VirtualDeviceConfiguration(memory_limit*10000)]
                    )
            else:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

        # print out physical and logical GPUs
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    else:
        print("No visible GPU is detected...")
        print("Possibly a mismatch in CUDA version if using NVIDIA 2xxx GPU")
            

def main ():
    #Model Setting
    parser = argparse.ArgumentParser(description='tensorflow gpu_stress_test')
    parser.add_argument('--vram', type=int, default=24, 
                        help ='vram size of each gpu (default:24)')
    parser.add_argument('--num_gpus', type=int, default=2, 
                        help ='number of gpus within the machine (default:2)')
    parser.add_argument('--epochs', type=int, default=500, 
                        help ='number of epoch to train(default:500)')
    parser.add_argument('--num_classes' ,type=int, default=10, 
                        help='number of classification classes (default : 10)')
    args=parser.parse_args()
    init_gpus(
        log_device_placement=False,
        create_virtual_devices=False,
        memory_limit=args.vram
    )

    cifar = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = tf.concat([x_train, x_train, x_train, x_train, x_train], 0)
    y_train = tf.concat([y_train, y_train, y_train, y_train, y_train], 0)

    classes = args.num_classes
    batch = 512 * args.vram * args.num_gpus
    epoch = args.epochs

    train_num=0
    s_time=time.time()    

    strategy = tf.distribute.MirroredStrategy() # multi gpu
    with strategy.scope():
        parallel_model = ResNet50(weights=None, input_shape=(32, 32, 3), classes=classes)
        parallel_model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),metrics=[keras.metrics.SparseCategoricalAccuracy()])

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch, drop_remainder=True)
    
    print("======================================================================")  
    print("Train Start")
    print("======================================================================")  
    
    # Train Start
    parallel_model.fit(x_train, y_train, epochs=epoch, validation_data=test_dataset, validation_freq=1, batch_size=batch,callbacks=[CustomCallback()])    
    parallel_model.evaluate(x_test, y_test, verbose=2)
    print("======================================================================")  
    print("Train End")
    print("Total Elapsed Time : {}".format(time.time()-s_time))
    print("======================================================================")    
    tf.keras.backend.clear_session()
    

if __name__ == '__main__':
    main()