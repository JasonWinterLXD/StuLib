import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pylab as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 查看训练集的形状
x_train.shape, y_train.shape

# 显示第0个样本对应的图片
plt.imshow(x_train[0])

# 显示第0个样本对应的数字
y_train[0]

# 定义CNN的输入层和输出层
# 输入层
input_shape = (28, 28, 1)

# 输出层
num_classes = 10

# 数据标准化处理，将原灰度级别取值范围[0, 255]缩放至区间[0, 1]
# 将图像的灰度值缩放至区间[0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 显示标准化处理后的训练集、测试集的形状
x_train.shape, x_test.shape

# 将通道信息加入数据集，即将数据形状改为（28,28,1)。因为输入层定义的形状为（28,28,1)，所以需要将数据的形状也改为(28,28,1)，即增加一个颜色的维度。
# 调用函数np.expand_disms()增加维数
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# 输出y_train
print(x_test.shape[0], "test samples")

# 对训练集中标签进行独热编码
y_train = keras.utils.to_categorical(y_train, num_classes)

# 对测试集中标签进行独热编码
y_test = keras.utils.to_categorical(y_test, num_classes)

# 显示训练集的标签y_train
y_train

# 对数据x_train,x_test进行探索性分析
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# 模型构建
num_filters_1 = 32  # 过滤器数量1
num_filters_2 = 64  # 过滤器数量2
kernel_size = (3, 3)  # 卷积核的大小
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),

        # 卷积层
        layers.Conv2D(num_filters_1, kernel_size, strides=1, padding='valid', activation='relu', use_bias=True,
                      bias_initializer=keras.initializers.zeros()),

        # 池化层
        layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid'),

        # 第二次卷积池化过程
        layers.Conv2D(num_filters_2, kernel_size, strides=1, padding='valid', activation='relu'),

        layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid'),

        # 扁平化输出
        keras.layers.Flatten(),

        # dropput层
        keras.layers.Dropout(rate=0.5),

        # 全连接层
        keras.layers.Dense(units=num_classes, activation='softmax'),
    ]
)

# 显示训练结果信息
model.summary()

# 训练模型

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 批（batch）大小
batch_size = 128
epochs = 10

reslut = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# 评价方法
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print(score)

plt.plot(range(1, epochs + 1), reslut.history["accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# 预测测试集的第1个样本的输出
predictions = model.predict(x_test[:1])
print(predictions)

# 显示预测标签结果
np.argmax(predictions[0])
