# encoding:utf-8
import numpy as np
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

# 载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# （60000,28,28）
print('x_train_shape:', x_train.shape)
# （60000）
print('y_train_shape:', y_train.shape)
# （10000,28,28）
print('x_test_shape:', x_test.shape)
# （10000）
print('y_test_shape:', y_test.shape)

# （60000,28,，28）->（60000，784）
x_train = x_train.reshape(x_train.shape[0], -1)/255.0
x_test = x_test.reshape(x_test.shape[0], -1)/255.0

# 换one hot格式  独热编码
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型,输入784个神经元,输出10个神经元
model = Sequential([
            Dense(units=200,
                  input_dim=784,
                  bias_initializer='one',
                  activation='tanh'),
            Dropout(0.5),
            Dense(units=10,
                  activation='softmax',
                  bias_initializer='zero',
                  )
        ])

# 定义优化器
sgd = SGD(lr=0.2)

# 定义优化器,loss function ,训练过程中计算准确率
model.compile(
    optimizer=sgd,
    loss='categorical_crossentropy',  # 交叉熵
    metrics=['accuracy']
)

# 训练模型
model.fit(x_train, y_train, batch_size=32, epochs=30)

# 评估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('\ntest loss', test_loss)
print('test accuracy', test_accuracy)
train_loss, train_accuracy = model.evaluate(x_train, y_train)
print('\ntrain loss', train_loss)
print('train accuracy', train_accuracy)

print()
print('\ntest loss', test_loss)
print('test accuracy', test_accuracy)
print('\ntrain loss', train_loss)
print('train accuracy', train_accuracy)

# 绘制图形并保存图形
plot_model(model,
           to_file="model.png",
           show_shapes=True,
           show_layer_names=True,
           rankdir='TB')  # rankdir 有'TB'从上到下 竖着;'LR'从左到右 横着

plt.figure(figsize=(10,10))
img = plt.imread("model.png")
plt.imshow(img)
plt.axis('off')
plt.show()

