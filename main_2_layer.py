import tensorflow as tf
import numpy as np
import dataloader as dl

trainset, testset = dl.load_data(160, 18)

x_train = trainset[0]
yy_train = trainset[1]

x_test = testset[0]
yy_test = testset[1]

num_classes = 3
y_train = np.zeros((x_train.shape[0], num_classes))
y_test = np.zeros((x_test.shape[0], num_classes))

# One shot code for test dataset
for i in range(x_train.shape[0]):
    index = int(yy_train[i]) - 1
    y_train[i, index] = 1

for j in range(x_test.shape[0]):
    index = int(yy_test[j]) - 1
    y_test[j, index] = 1

cross_validation = 10
LEARNING_RATE = 0.001
ITERATION = 501
HIDDEN_SIZE_1 = 6
HIDDEN_SIZE_2 = 5
INPUT_SIZE = 13
CLASS_NUMBER = 3

k = 0

x = tf.placeholder("float", [None, INPUT_SIZE])
y = tf.placeholder("float", [None, CLASS_NUMBER])

w1 = tf.Variable(tf.random_normal([INPUT_SIZE, HIDDEN_SIZE_1]))
b1 = tf.Variable(tf.random_normal([HIDDEN_SIZE_1]))
h1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([HIDDEN_SIZE_1, HIDDEN_SIZE_2]))
b2 = tf.Variable(tf.random_normal([HIDDEN_SIZE_2]))
h2 = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)

w3 = tf.Variable(tf.random_normal([HIDDEN_SIZE_2, CLASS_NUMBER]))
b3 = tf.Variable(tf.random_normal([CLASS_NUMBER]))

predict_y = tf.nn.softmax(tf.matmul(h2, w3) + b3)

# 交叉熵损失函数
cross_entropy = - tf.reduce_sum(y * tf.log(predict_y))
loss = tf.reduce_mean(cross_entropy)

# 梯度下降
train_op = tf.train.MomentumOptimizer(LEARNING_RATE, momentum=0.9).minimize(loss)

equal_op = tf.equal(tf.argmax(y, 1), tf.argmax(predict_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(equal_op, "float"))

# 初始化 op
init_op = tf.initialize_all_variables()

# 创建一个保存变量的op
# saver = tf.train.Saver()


def confusion_mat(cor_y, pre_y):
    conf_matrix = np.zeros((3, 3))
    for num in range(cor_y.shape[0]):
        if cor_y[num] == 0 and pre_y[num] == 0:
            conf_matrix[0][0] += 1
        elif cor_y[num] == 0 and pre_y[num] == 1:
            conf_matrix[0][1] += 1
        elif cor_y[num] == 0 and pre_y[num] == 2:
            conf_matrix[0][2] += 1
        elif cor_y[num] == 1 and pre_y[num] == 0:
            conf_matrix[1][0] += 1
        elif cor_y[num] == 1 and pre_y[num] == 1:
            conf_matrix[1][1] += 1
        elif cor_y[num] == 1 and pre_y[num] == 2:
            conf_matrix[1][2] += 1
        elif cor_y[num] == 2 and pre_y[num] == 0:
            conf_matrix[2][0] += 1
        elif cor_y[num] == 2 and pre_y[num] == 1:
            conf_matrix[2][1] += 1
        else:
            conf_matrix[2][2] += 1
    return conf_matrix


def calculate_confusion(con):
    a = con[0][0]
    b = con[1][1]
    c = con[2][2]
    d = con[1][0] + con[2][0]
    e = con[0][1] + con[2][1]
    f = con[0][2] + con[1][2]
    row_0 = con[0][0] + con[0][1] + con[0][2]
    row_1 = con[1][0] + con[1][1] + con[1][2]
    row_2 = con[2][0] + con[2][1] + con[2][2]

    TPR_0 = a / row_0
    TPR_1 = b / row_1
    TPR_2 = c / row_2
    FPR_0 = d / (row_1 + row_2)
    FPR_1 = e / (row_0 + row_2)
    FPR_2 = f / (row_0 + row_1)

    return TPR_0, TPR_1, TPR_2, FPR_0, FPR_1, FPR_2


with tf.Session() as sess:
    fold_accuracy = np.zeros(cross_validation)
    fold_tpr_1 = np.zeros(cross_validation)
    fold_tpr_2 = np.zeros(cross_validation)
    fold_tpr_3 = np.zeros(cross_validation)
    fold_fpr_1 = np.zeros(cross_validation)
    fold_fpr_2 = np.zeros(cross_validation)
    fold_fpr_3 = np.zeros(cross_validation)

    sess = tf.Session()
    sess.run(init_op)
    # 迭代
    for cv in range(cross_validation):
        sess = tf.Session()
        sess.run(init_op)

        print("This is %d cross validation:" % (cv+1))

        indices = np.arange(k*16, (k+1)*16, 1)
        k += 1

        x_cv_train = np.delete(x_train, indices, axis=0)
        y_cv_train = np.delete(y_train, indices, axis=0)
        x_cv_test = np.take(x_train, indices, axis=0)
        y_cv_test = np.take(y_train, indices, axis=0)

        for step in range(ITERATION):

            sess.run(train_op, feed_dict={x: x_cv_train, y: y_cv_train})
            cv_train_accuracy = sess.run(accuracy_op, feed_dict={x: x_cv_train, y: y_cv_train})
            cv_test_accuracy = sess.run(accuracy_op, feed_dict={x: x_cv_test, y: y_cv_test})

            if step % 10 == 0:
                print("    Iteration-%d accuracy: train-%f test-%f" %
                    (step, cv_train_accuracy*100.0, cv_test_accuracy*100.0))

        cv_confusion_matrix = np.zeros((3, 3))
        predict_cv_y = sess.run(predict_y, feed_dict={x: x_cv_test, y: y_cv_test})
        correct_cv_y_class = np.argmax(y_cv_test, 1)
        predict_cv_y_class = np.argmax(predict_cv_y, 1)

        cv_confusion_matrix = confusion_mat(correct_cv_y_class, predict_cv_y_class)

        TPR_0, TPR_1, TPR_2, FPR_0, FPR_1, FPR_2 = calculate_confusion(cv_confusion_matrix)

        # print(confusion_matrix)
        print("True positive rate for class 1 = " + str(TPR_0))
        print("True positive rate for class 2 = " + str(TPR_1))
        print("True positive rate for class 3 = " + str(TPR_2))
        print("False positive rate for class 1 = " + str(FPR_0))
        print("False positive rate for class 2 = " + str(FPR_1))
        print("False positive rate for class 3 = " + str(FPR_2))
        print('\n')

        fold_accuracy[cv] = cv_test_accuracy
        fold_tpr_1[cv] = TPR_0
        fold_tpr_2[cv] = TPR_1
        fold_tpr_3[cv] = TPR_2
        fold_fpr_1[cv] = FPR_0
        fold_fpr_2[cv] = FPR_1
        fold_fpr_3[cv] = FPR_2


    print('\nMean accuracy = ' + str(np.mean(fold_accuracy)) +
            ', Standard deviation = +-' + str(np.std(fold_accuracy)))
    print('\nMean for class 1 TPR  = ' + str(np.mean(fold_tpr_1)) +
          ', Standard deviation = +-' + str(np.std(fold_tpr_1)))
    print('\nMean for class 2 TPR = ' + str(np.mean(fold_tpr_2)) +
          ', Standard deviation = +-' + str(np.std(fold_tpr_2)))
    print('\nMean for class 3 TPR = ' + str(np.mean(fold_tpr_3)) +
          ', Standard deviation = +-' + str(np.std(fold_tpr_3)))
    print('\nMean for class 1 FPR = ' + str(np.mean(fold_fpr_1)) +
          ', Standard deviation = +-' + str(np.std(fold_fpr_1)))
    print('\nMean for class 2 FPR = ' + str(np.mean(fold_fpr_2)) +
          ', Standard deviation = +-' + str(np.std(fold_fpr_2)))
    print('\nMean for class 3 FPR = ' + str(np.mean(fold_fpr_3)) +
          ', Standard deviation = +-' + str(np.std(fold_fpr_3)))

    sess = tf.Session()
    sess.run(init_op)
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    for step in range(ITERATION):
        sess.run(train_op, feed_dict={x: x_train, y: y_train})
        train_accuracy = sess.run(accuracy_op, feed_dict={x: x_train, y: y_train})
        test_accuracy = sess.run(accuracy_op, feed_dict={x: x_test, y: y_test})

        if step % 5 == 0:
            print("    Iteration-%d accuracy: train-%f test-%f" %
                (step, train_accuracy*100.0, test_accuracy*100.0))

    confusion_matrix = np.zeros((3, 3))
    predict_y = sess.run(predict_y, feed_dict={x: x_test, y: y_test})
    correct_y_class = np.argmax(y_test, 1)
    predict_y_class = np.argmax(predict_y, 1)

    confusion_matrix = confusion_mat(correct_y_class, predict_y_class)

    TPR0, TPR1, TPR2, FPR0, FPR1, FPR2 = calculate_confusion(confusion_matrix)

    print("True positive rate for class 1 = " + str(TPR0))
    print("True positive rate for class 2 = " + str(TPR1))
    print("True positive rate for class 3 = " + str(TPR2))
    print("False positive rate for class 1 = " + str(FPR0))
    print("False positive rate for class 2 = " + str(FPR1))
    print("False positive rate for class 3 = " + str(FPR2))
    print('\n')


