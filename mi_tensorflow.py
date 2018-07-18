import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None,1])
z = tf.placeholder(tf.float32, [None,1])
z_ = tf.placeholder(tf.float32, [None,1])

n_hidden=10

Wx=tf.Variable(tf.random_normal(stddev=0.1,shape=[1,n_hidden]))
Wz=tf.Variable(tf.random_normal(stddev=0.1,shape=[1,n_hidden]))
b=tf.Variable(tf.constant(0.1,shape=[n_hidden]))

hidden_joint=tf.nn.relu(tf.matmul(x,Wx)+tf.matmul(z,Wz)+b)
hidden_marg=tf.nn.relu(tf.matmul(x,Wx)+tf.matmul(z_,Wz)+b)

Wout=tf.Variable(tf.random_normal(stddev=0.1,shape=[n_hidden,1]))
bout=tf.Variable(tf.constant(0.1,shape=[1]))

out_joint=tf.matmul(hidden_joint,Wout)+bout
out_marg=tf.matmul(hidden_marg,Wout)+bout

lower_bound=tf.reduce_mean(out_joint)-tf.log(tf.reduce_mean(tf.exp(out_marg)))

train_step = tf.train.AdamOptimizer(0.005).minimize(-lower_bound)


N = 20000
sigma = 1
def sample_data():
    x = np.sign(np.random.normal(0., 1., [N,1])).astype(np.float32)
    z = x+np.random.normal(0., np.sqrt(sigma), [N,1]).astype(np.float32)
    return x, z


def compute_mi(x, z):
    p_z_x = np.exp(-(z-x)**2/(2*sigma))
    p_z_x_minus = np.exp(-(z+1)**2/(2*sigma))
    p_z_x_plus  = np.exp(-(z-1)**2/(2*sigma))
    mi = np.average(np.log(p_z_x/(0.5*p_z_x_minus+0.5*p_z_x_plus)))
    return mi

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        x_sample, z_sample = sample_data()
        z_shuffle=np.random.permutation(z_sample)
        _, mi= sess.run([train_step, lower_bound],
                        feed_dict={x:x_sample,
                                   z:z_sample,
                                   z_:z_shuffle})
        print("Iteration {} - lower_bound={} (real {})"
              "".format(i, mi, compute_mi(x_sample, z_sample)))
