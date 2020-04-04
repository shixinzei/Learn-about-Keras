# encoding:utf-8
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, TFOptimizer, Nadam
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

# 载入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train_shape:', x_train.shape)
print('y_train_shape:', y_train.shape)

# (60000,28,28)->(60000,28,28,1)
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
print('x_train_shape:', x_train.shape)
print('x_test_shape:', x_test.shape)

# 换one hot格式 独热编码
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print('y_train_shape:', y_train.shape)
print('y_test_shape:', y_test.shape)

# 定义顺序模型
model = Sequential()

# 第一个卷积层
# input_shape 输入平面
# filters 卷积核/滤波器个数
# kernel_size 卷积窗口大小
# strides 步长
# padding padding方式 same/valid
# activation 激活函数
model.add(Convolution2D(
    input_shape=(28, 28, 1),
    filters=32,  # 卷积核 滤波器
    kernel_size=5,
    strides=1,
    padding='same',  # padding:same 和上一层平面大小一样
    activation='relu'  # 激活函数
))

# 第一个池化层 14*14
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
))

# 第二个卷积层
model.add(Convolution2D(64, 5, strides=1, padding='same', activation='relu'))

# 第二个池化层
model.add(MaxPooling2D(2, 2, 'same'))

# 把第二个池化层的输出扁平化为1维
model.add(Flatten())

# 第一个全连接层
model.add(Dense(1024, activation='relu'))

# Dropout
model.add(Dropout(0.5))

# 第二个全连接层
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

# train loss 0.26415830355286596
# train accuracy 0.9276166558265686
# test loss 0.27433498376607895
# test accuracy 0.9236999750137329