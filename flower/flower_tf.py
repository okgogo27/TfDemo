import tensorflow as tf
import numpy as np
import csv

# 创建神经网络
def flower_network(data,units):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.random_normal([4, units]), name='weight')
        biases = tf.Variable(tf.zeros([units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(data, weights) + biases)
    layer.append(hidden1)
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.random_normal([units,3],
                             name='weights'))
        biases = tf.Variable(tf.zeros(3),
                             name='biases')
        result = tf.nn.softmax(tf.matmul(hidden1, weights) + biases)
    layer.append(result)
    return result

# 创建损失函数
def get_loss(real,result):
    cross_entropy = -tf.reduce_sum(real * tf.log(result)) # 交叉熵
    return cross_entropy

if __name__ == '__main__':
    layer = []  # 神经网络集合
    types = 3   # 分类数（y）
    features = 4  # 特征数（x）

    # 占位符变量
    x = tf.placeholder(tf.float32, shape=(1, features))
    y = tf.placeholder(tf.float32, shape=(1, types))


    result = flower_network(x, 10) # 模型

    loss = get_loss(y, result)  # 损失函数

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # 梯度下降

    with tf.Session() as sess:
        # 训练阶段
        sess.run(tf.global_variables_initializer()) # 初始化变量
        for i in range(100):
            with open('iris_training.csv') as csvfile:
                reader = csv.reader(csvfile)
                index = 0
                train_num = 0
                train_correct_num = 0
                for row in reader:
                    # 排除标题
                    if index is 0:
                        index = 1
                        continue

                    train_num += 1 # 训练总数+1

                    feature = np.array([row[0], row[1], row[2], row[3]], dtype=float) # 构造特征矩阵（x）

                    # 构造目标矩阵 （y）
                    if row[4] is '0':
                        target = np.array([1, 0, 0])
                    elif row[4] is '1':
                        target = np.array([0, 1, 0])
                    elif row[4] is '2':
                        target = np.array([0, 0, 1])
                    else:
                        continue

                    _, loss_value = sess.run([train_op, loss], feed_dict={x: [feature], y: [target]}) # 训练模型
                    print(loss_value) # 打印损失值

                    result = sess.run(layer[-1], feed_dict={x: [feature]}) # 预测模型
                    position = np.argwhere(result == max(result[0])) # 获取最大概率的元素下标
                    if int(position[0][1]) is int(row[4]):
                        train_correct_num += 1

        print('train:  ' + str(train_correct_num / train_num)) # 打印训练期准确率

        #
        # 测试阶段
        test_num = 0
        test_correct_num = 0
        with open('iris_test.csv') as csvfile_test:
            reader = csv.reader(csvfile_test)
            index = 0
            for row in reader:
                if index is 0:
                    index = 1
                    continue
                test_num += 1
                feature = np.array([row[0], row[1], row[2], row[3]], dtype=float)
                result = sess.run(layer[-1], feed_dict={x: [feature]})
                position = np.argwhere(result == max(result[0]))
                if int(position[0][1]) is int(row[4]):
                    test_correct_num += 1
        print('test:  ' + str(test_correct_num / test_num))
