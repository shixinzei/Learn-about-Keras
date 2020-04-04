# encoding:utf-8
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, TFOptimizer,Nadam

# 载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# （60000,28,28）
print('x_shape:', x_train.shape)
# （60000）
print('y_shape:', y_train.shape)

# （60000,28,，28）->（60000，784）
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
print('x_shape:', x_train.shape)
print('y_shape:', y_train.shape)

# 换one hot格式  独热编码
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型
model = Sequential([
    # 隐藏层1: 784-200
    Dense(units=200,
          input_dim=784,
          bias_initializer='one',  # 偏置项初始化
          activation='tanh'),

    # 针对过拟合：Dropout
    Dropout(0.4),

    # 隐藏层2: 200-100
    Dense(units=100,
          bias_initializer='one',
          activation='tanh'),

    # 针对过拟合：Dropout
    Dropout(0.4),

    # 输出层 100-10
    Dense(units=10,
          bias_initializer='one',
          activation='softmax')
])

# 定义优化器
sgd = SGD(lr=0.30)

# 定义优化器,loss function ,训练过程中计算准确率
model.compile(
    optimizer=sgd,
    loss='mse',
    # loss='categorical_crossentropy',  # 交叉熵
    metrics=['accuracy']
)

# 训练模型
model.fit(x_train, y_train, batch_size=64, epochs=30)

# 评估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test)
train_loss, train_accuracy = model.evaluate(x_train, y_train)
print('\ntest loss', test_loss)
print('test accuracy', test_accuracy)
print('\ntrain loss', train_loss)
print('train accuracy', train_accuracy)


