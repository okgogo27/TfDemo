import tensorflow as tf
import re
import constant
import numpy as np
from mongo_helper import MongoHelper
from bson.objectid import ObjectId
import traceback

# 创建模型模板
def create_model_template():
    for data in mongo_db.findList(constant.spider_articles,{'page_url':rex}):
        # 预测目标模板（y）
        if data['type'] and data['type'][0] not in target:
            target.append(data['type'][0])
        # 特征模板 （x）
        item = mongo_db.findOne(constant.spider_segment,{'crawlId':str(data['_id'])})
        if item:
            dict = item['segment']
            for key in dict.keys():
                if dict[key] < 5: # 特征频次小于5的剔除
                    continue
                if key not in features:
                    features.append(key)

# 构建特征矩阵（x)
def create_x(data):
    feature = []
    for item in features:
        if item in data:
            # 排除错误数据
            if data[item]<0:
                return -1

            feature.append(data[item])
        else:
            feature.append(0)
    return feature

# 构建神经网络
def article_network(data, hidden1_units):
    # 第一层神经网络（relu激活函数）
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.random_normal([len(features), hidden1_units]), name='weight')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(data, weights) + biases)
    layer.append(hidden1)
    # 第二层神经网络（softmax输出结果）
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.random_normal([hidden1_units,len(target)],
                             name='weights'))
        biases = tf.Variable(tf.zeros(len(target)),
                             name='biases')
        result = tf.nn.softmax(tf.matmul(hidden1, weights) + biases)
    layer.append(result)
    return result

# 构建损失函数
def get_loss(real,result):
    cross_entropy = -tf.reduce_sum(real * tf.log(result)) # 交叉熵
    return cross_entropy

if __name__ == '__main__':

    features = [] # 特征数（x）
    target = []  # 分类数（y）
    layer = []  # 神经网络集合
    rex = re.compile('.*' + 'eeworld' + '.*', re.IGNORECASE) # 正则表达 仅抽取www.eeworld.com网站的文章

    mongo_db = MongoHelper() # 初始化mongodb

    create_model_template() # 创建模型模板

    # 占位符变量
    x = tf.placeholder(tf.float32, shape=(1, len(features)))
    y = tf.placeholder(tf.float32, shape=(1, len(target)))

    result = article_network(x, 10) # 构建神经网络

    loss = get_loss(y, result) # 损失函数

    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 梯度下降，构建模型

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # 初始化变量

        train_correct_num = 0  # 模型训练期 正确预测数据总数
        data_num = 0 # 数据总数
        test_data_num = 0 # 模型测试期数据总数
        test_correct_num = 0 # 模型测试期 正确预测数据总数

        for item in mongo_db.findList(constant.spider_articles, {'page_url': rex}):
            data_num += 1
            # 训练模型阶段
            # 2000条模型用作训练数据
            if data_num < 2000:
                segment = mongo_db.findOne(constant.spider_tfidf, {'crawlId': str(item['_id'])})
                if segment:
                    # 构建特征矩阵(x)
                    feature = create_x(segment['segment'])

                    # 排除错误数据
                    if feature is -1:
                        train_correct_num-=1
                        continue

                    # 构建目标矩阵（y）
                    if feature.count(0) == len(feature) or not item['type']:
                        continue
                    type = []
                    for data in target:
                        if data == item['type'][0]:
                            type.append(1)
                        else:
                            type.append(0)

                    # 训练模型
                    _, loss_value = sess.run([train_op, loss], feed_dict={x: [feature], y: [type]})

                    # 预测模型
                    pre_data = sess.run(layer[-1], feed_dict={x: [feature]})
                    index = np.argwhere(pre_data == max(pre_data[0])) # 获取最大概率的元素下标
                    if index.any() and target[index[0][1]] == item['type'][0]:
                        train_correct_num += 1
                    print(loss_value)  # 打印损失值

            # 测试模型阶段
            else:
                test_data_num += 1
                segment = mongo_db.findOne(constant.spider_tfidf, {'crawlId': str(item['_id'])})
                if segment:
                    # 构建特征矩阵(x)
                    feature = create_x(segment['segment'])

                    # 排除错误数据
                    if feature is -1:
                        test_data_num-=1
                        continue

                    # 特征矩阵（x）为0  剔除
                    # 该条数据不含预测目标（y） 剔除
                    if feature.count(0) == len(feature) or not item['type']:
                        continue

                    # 预测模型
                    pre_data = sess.run(layer[-1], feed_dict={x: [feature]})
                    index = np.argwhere(pre_data == max(pre_data[0]))
                    if index.any() and target[index[0][1]] == item['type'][0]:
                        test_correct_num += 1

        print('train_accuracy:   ' + str(train_correct_num / 2000))
        print('test_accuracy:   ' + str(test_correct_num / test_data_num))

        print('---------------over-------------')


        # 模型预测测试入口
        # 控制台输入 crawId
        # 返回target概率数组，预测目标与正确目标
        while True:
            try:
                str = input('crawlId: ')
                test = mongo_db.findOne(constant.spider_segment, {'crawlId': str})
                test_data = create_x(test['segment'])
                value_y = sess.run(layer[-1], feed_dict={x: [test_data]})
                print(value_y)
                index = np.argwhere(value_y == max(value_y[0]))
                print('pre:   ' + target[index[0][1]])
                print('rel:   ' + (mongo_db.findOne(constant.spider_articles, {'_id': ObjectId(str)}))['type'][0])
            except:
                print(traceback.format_exc())
                continue


