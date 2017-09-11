import images_train
import images_test
import labels_test
import labels_train
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 225])
y_ = tf.placeholder(tf.float32, shape=[None, 15])
'''
W = tf.Variable(tf.zeros([225,15]))
b = tf.Variable(tf.zeros([15]))
'''
'''
sess.run(tf.global_variables_initializer())
y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for _ in range(0):
  train_step.run(feed_dict={x: images_train.images, y_: labels_train.labels})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x:images_test.images, y_: labels_test.labels}))

print(accuracy.eval(feed_dict={x:images_train.images, y_: labels_train.labels}))
'''


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
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
'''
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations
'''


W_conv1 = weight_variable([4, 4, 1, 15])
#variable_summaries(W_conv1, name = 'W_conv1')
b_conv1 = bias_variable([15])
#variable_summaries(b_conv1)
x_image = tf.reshape(x, [-1,15,15,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#tf.summary.histogram('h_conv1', h_conv1)

'''
W_conv2 = weight_variable([4, 4, 60, 30])
#variable_summaries(W_conv2)
b_conv2 = bias_variable([30])
#variable_summaries(b_conv2)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#tf.summary.histogram('h_conv2', h_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([2, 2, 30, 15])
#variable_summaries(W_conv2)
b_conv3 = bias_variable([15])
#variable_summaries(b_conv2)

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#tf.summary.histogram('h_conv2', h_conv2)
h_pool3 = max_pool_2x2(h_conv3)
'''



W_fc1 = weight_variable([15 * 15 * 15, 1024])
#variable_summaries(W_fc1)
b_fc1 = bias_variable([1024])
#variable_summaries(b_fc1)

h_pool1_flat = tf.reshape(h_conv1, [-1, 15*15*15])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
#tf.summary.histogram('h_fc1', h_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 15])
#variable_summaries(W_fc2)
b_fc2 = bias_variable([15])
#variable_summaries(b_fc2)

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#tf.summary.histogram('y_conv', y_conv)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
tf.summary.scalar('cross_entropy', cross_entropy)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)





merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("/home/lxiao/Documents/AlphaGold/summaries/train2", sess.graph)
test_writer = tf.summary.FileWriter("/home/lxiao/Documents/AlphaGold/summaries/test2")



saver = tf.train.Saver()

step = tf.Variable(0)
version = tf.Variable(0)


sess.run(tf.global_variables_initializer())

saver.restore(sess, "/home/lxiao/Documents/save/AlphaGold3_1/alpha_gold.ckpt")
print("Model restored")

accmax,ct = sess.run([accuracy,accuracy], feed_dict={x: images_test.images, y_: labels_test.labels, keep_prob: 1.0})
train_accmax,cn = sess.run([accuracy,accuracy],feed_dict={x:images_train.images, y_: labels_train.labels, keep_prob: 1.0})


for j in range (1000000):
  i = sess.run(step)
  v = sess.run(version)

  if (i%3==0) :
    print('Step %s started' % (i))
    train_accuracy = accuracy.eval(feed_dict={
        x:images_train.images, y_: labels_train.labels, keep_prob: 1.0})
    print("version %d, training accuracy %g"%(v, train_accuracy))

    acc = sess.run(accuracy, feed_dict={x: images_test.images, y_: labels_test.labels, keep_prob: 1.0})

    print('Test accuracy at version %s: %s' % (v, acc))
  
    if acc < accmax:
      saver.restore(sess, "/home/lxiao/Documents/save/AlphaGold3_1/alpha_gold.ckpt")
      print("Model restored to %d, train acc %g, test acc %g"%(v,cn,ct))

    if acc > accmax:
      accmax = acc
      save_path = saver.save(sess, "/home/lxiao/Documents/save/AlphaGold3_1/alpha_gold.ckpt")
      version += 1
      ct = accmax
      cn = train_accuracy
      v = sess.run(version)
      print("Test improved! Model saved to %d,train acc %g, test acc %g"%(v,cn,ct))
      summary = sess.run(merged, feed_dict={x: images_test.images, y_: labels_test.labels, keep_prob: 1.0})
      test_writer.add_summary(summary, i)
      summary = sess.run(merged,feed_dict={
        x:images_train.images, y_: labels_train.labels, keep_prob: 1.0})
      train_writer.add_summary(summary, i)

  sess.run(train_step, feed_dict={x: images_train.images, y_: labels_train.labels, keep_prob: 0.0001})
  step = step + 1
'''
    if (acc == accmax) & (train_accuracy > train_accmax):
      train_accmax = train_accuracy
      save_path = saver.save(sess, "/home/lxiao/Documents/save/AlphaGold3_2/alpha_gold.ckpt")
      version += 1
      cn = train_accmax
      v = sess.run(version)
      print("Train improved! Model saved to %d,train acc %g, test acc %g"%(v,cn,ct))
      summary = sess.run(merged,feed_dict={
        x:images_train.images, y_: labels_train.labels, keep_prob: 1.0})
      train_writer.add_summary(summary, i)

    if (acc == accmax) & (train_accuracy < train_accmax):
      saver.restore(sess, "/home/lxiao/Documents/save/AlphaGold3_2/alpha_gold.ckpt")
      print("Model restored to %d, train acc %g, test acc %g"%(v,cn,ct))

'''     

'''
|((v<6)&(i%2==0))|((v<10)&(i%3)==0)|((v<20) & (i%5==0))|((v<60)&(i%10==0))|((v>=60)&(i % 50 == 0))


  train_step.run(feed_dict={x: images_train.images, y_: labels_train.labels, keep_prob: 0.5})
'''



