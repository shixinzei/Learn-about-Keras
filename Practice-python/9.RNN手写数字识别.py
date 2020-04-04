# encoding:utf-8
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, TFOptimizer, Nadam
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import SimpleRNN  # 最简单的RNN还有LSTM和GRU

# 数据长度- 一行有28个像素
input_size = 28
# 序列长度- 一共有28行
time_steps = 28
# 隐藏层cell个数
cell_size = 50

# 载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train_shape:', x_train.shape)
print('y_train_shape:', y_train.shape)

# (600000,784) ->(60000,28,28)
# (60000,28,28)
x_train = x_train / 255.0
x_test = x_test / 255.0
print('x_train_shape:', x_train.shape)
print('x_test_shape:', x_test.shape)

# 换one hot格式
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)  # one hot
print('y_train_shape:', y_train.shape)
print('y_test_shape:', y_test.shape)

# 创建模型
model = Sequential()

# 循环神经网络 此时是输入层+隐藏层+输出层
model.add(SimpleRNN(
    units=cell_size,  # 输出到隐藏层
    input_shape=(time_steps, input_size),  # 输入
))

# 输出层 激活函数:softmax
model.add(Dense(10, activation='softmax'))

# 定义优化器
adam = Adam(lr=1e-4)

# 定义优化器，loss function，训练过程中计算准确率
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
model.fit(x_train, y_train, batch_size=64, epochs=10)

# 评估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test)
train_loss, train_accuracy = model.evaluate(x_train, y_train)
print('\ntrain loss', train_loss)
print('train accuracy', train_accuracy)
print('test loss', test_loss)
print('test accuracy', test_accuracy)

# 绘制图形并保存图形
plot_model(model,
           to_file="model.png",
           show_shapes=True,
           show_layer_names=True,
           rankdir='TB')  # rankdir 有'TB'从上到下 竖着;'LR'从左到右 横着

plt.figure(figsize=(10, 10))
img = plt.imread("model.png")
plt.imshow(img)
plt.axis('off')
plt.show()

# train loss 0.01356080276845023
# train accuracy 0.996066689491272
# test loss 0.024042980225158682
# test accuracy 0.9916999936103821