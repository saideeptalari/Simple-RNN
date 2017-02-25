from __future__ import print_function
import numpy as np
import tensorflow as tf
from numpy.random import shuffle
from tensorflow.contrib.rnn import LSTMCell
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

#create dataset
def create_dataset(num_samples, seq_length=10):
    x = np.linspace(0,20,num_samples)
    X = x*np.sin(x) + x*np.cos(2*x)
    data = np.split(X,int(num_samples/seq_length))
    output = []
    input_data = []
    for i, chunk in enumerate(data):
        o = np.roll(chunk, -1)
        try:
            o[-1] = data[i+1][0]
        except IndexError:
            o[-1] = o[-2]
        output.append(o)
    return np.array(data).reshape(-1,10,1), np.array(output).reshape(-1,10)

lstm_units = 64

#Input shape: (num_samples,seq_length,input_dimension)
#Output shape: (num_samples, target)
input_data = tf.placeholder(tf.float32,shape=[None,None,1])
output_data = tf.placeholder(tf.float32,shape=[None,None])
cell = LSTMCell(lstm_units,num_proj=1,state_is_tuple=True)
out,_ = tf.nn.dynamic_rnn(cell,input_data,dtype=tf.float32)   #shape: (None, 10, 1)

pred = tf.squeeze(out)    #shape: (None, 10)

cost = tf.reduce_mean(tf.square(pred - output_data))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)


init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
costs = []

for epoch in xrange(100):
    inp_data,out_data = create_dataset(1000, 10)
    _,c = sess.run([optimizer,cost],feed_dict={input_data: inp_data, output_data: out_data})
    print("Epoch: {}, Cost: {}".format(epoch,c))
    costs.append(c)

predicted = sess.run(pred, feed_dict={input_data: inp_data})

plt.grid("off")
plt.plot(out_data.flatten(), label="Actual")
plt.legend()
plt.show()
plt.plot(costs,label="Cost Function")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.legend()
plt.show()
plt.plot(out_data.flatten(), label="True")
plt.plot(predicted.flatten(), "r--", label="Predicted")
plt.legend()
plt.show()
sess.close()
