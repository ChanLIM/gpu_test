from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np
#Limit GPU VRAM
#tf.config.gpu.set_per_process_memory_fraction(0.75)
#Allow Growth GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import argparse


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--vram',type=int,default=8)
parser.add_argument('--num_gpus',type=int,default='1',help='gpu_num')
parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                   help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                   help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
args = parser.parse_args()


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

mirrored_strategy = tf.distribute.MirroredStrategy()
#mirrored_strategy=tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

mnist = tf.keras.datasets.mnist


batchsize=1000 *args.vram*args.num_gpus
#batchsize=1000
EPOCHS = 100

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=np.repeat(x_train,30,axis=0)
y_train=np.repeat(y_train,30,axis=0)
#x_test=np.repeat(x_test,10,axis=0)
#y_test=np.repeat(y_test,10,axis=0)

x_train, x_test = x_train / 255.0, x_test / 255.0

# 채널 차원을 추가합니다.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# 모델, Optimizer, Dataset multi gpu 화
with mirrored_strategy.scope():
    model = MyModel()
    optimizer = tf.keras.optimizers.Adam()
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batchsize)
    train_ds = mirrored_strategy.experimental_distribute_dataset(train_ds)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchsize)
    loss_object = tf.losses.SparseCategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):

    def step_fn(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            train_accuracy(labels, predictions)
            labels = tf.one_hot(labels, depth=10)
            print(predictions.shape,labels.shape)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels)
            #cross_entropy = loss_object(labels, predictions)
            loss = tf.reduce_sum(cross_entropy) * (1.0 / batchsize)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(list(zip(gradients, model.trainable_variables)))
        return cross_entropy

    per_example_losses = mirrored_strategy.experimental_run_v2(step_fn, args=(images, labels,))
    mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=0)
    loss  = mean_loss / batchsize
    train_loss(loss)

  


@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)
  test_loss(t_loss)
  test_accuracy(labels, predictions)


for epoch in range(EPOCHS):
  #print("1")
  with mirrored_strategy.scope():
      #print("2")
      for images, labels in train_ds:
          #print("3")
          train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  
  print ("EPOCH : %d\t train_loss : %0.6f\t train_AP : %0.4f\t test_loss : %0.6f\t test_AP : %0.4f" %(epoch+1,
                         train_loss.result(),train_accuracy.result()*100,test_loss.result(),test_accuracy.result()*100))
#  print (template.format(epoch+1,
#                         train_loss.result(),
#                         train_accuracy.result()*100,
#                         test_loss.result(),
#                         test_accuracy.result()*100))






