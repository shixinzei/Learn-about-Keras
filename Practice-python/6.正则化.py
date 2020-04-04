# encoding:utf-8
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

# 载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# （60000,28,28）
print('x_shape:', x_train.shape)
# （60000）
print('y_shape:', y_train.shape)

# （60000,28,，28）->（60000，784）
x_train = x_train.reshape(x_train.shape[0], -1)/255.0
x_test = x_test.reshape(x_test.shape[0], -1)/255.0
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
              activation='tanh',
              kernel_regularizer=l2(0.0003)),  # 使用正则化（Regularization）技巧来减少模型的过拟合效果
        Dropout(0.4),
        # 隐藏层2: 200-100
        Dense(units=100,
              bias_initializer='one',
              activation='relu',
              kernel_regularizer=l2(0.0003)),  # 使用正则化（Regularization）技巧来减少模型的过拟合效果
        Dropout(0.4),
        # 隐藏层3: 100-20
        Dense(units=20,
              bias_initializer='one',
              activation='relu',
              kernel_regularizer=l2(0.0003)),  # 使用正则化（Regularization）技巧来减少模型的过拟合效果
        Dropout(0.4),
        # 输出层: 100-10
        Dense(units=10,
              bias_initializer='one',
              activation='softmax',
              kernel_regularizer=l2(0.0003))  # 使用正则化（Regularization）技巧来减少模型的过拟合效果
    ])

# 定义优化器
sgd = SGD(lr=0.2)
# 定义优化器,loss function ,训练过程中计算准确率
model.compile(
    optimizer=sgd,
    # loss='mse',
    loss='categorical_crossentropy',  # 交叉熵
    metrics=['accuracy']
)
# 训练模型
model.fit(x_train, y_train, batch_size=32, epochs=20)

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

# train loss 0.3468583889802297
# train accuracy 0.9540666937828064
# test loss 0.36486811208724973
# test accuracy 0.9501000046730042

