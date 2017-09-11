import images_train
import images_test
import labels_test
import labels_train
import tensorflow as tf

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 225])
y_ = tf.placeholder(tf.float32, shape=[None, 15])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 9])
b_conv1 = bias_variable([9])
x_image = tf.reshape(x, [-1,15,15,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 9, 18])
b_conv2 = bias_variable([18])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 18, 36])
b_conv3 = bias_variable([36])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([2 * 2 * 36, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 2*2*36])

h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 15])
b_fc2 = bias_variable([15])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
tf.summary.scalar('cross_entropy', cross_entropy)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("/home/lxiao/Documents/AlphaGold/summaries/train", sess.graph)
test_writer = tf.summary.FileWriter("/home/lxiao/Documents/AlphaGold/summaries/test")
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
'''
saver.restore(sess, "/home/lxiao/Documents/save/AlphaGold3/alpha_gold3.ckpt")
print("Model restored")
'''
for i in range(5000):
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:images_train.images, y_: labels_train.labels, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))

    summary, acc = sess.run([merged, accuracy], feed_dict={x: images_test.images, y_: labels_test.labels, keep_prob: 1.0})
    test_writer.add_summary(summary, i)
    print('Test accuracy at step %s: %s' % (i, acc))

    save_path = saver.save(sess, "/home/lxiao/Documents/save/AlphaGold3/alpha_gold3.ckpt")
    print("Model saved in the file: %s" % save_path)

  if i % 10 == 0:
    print('Step %s completed' % (i))
    
  summary, _ = sess.run([merged, train_step], feed_dict={x: images_train.images, y_: labels_train.labels, keep_prob: 0.5})
  train_writer.add_summary(summary, i)


