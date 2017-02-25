from __future__ import print_function
import numpy as np
import tensorflow as tf
from numpy.random import shuffle
from tensorflow.contrib.rnn import LSTMCell
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

#create dataset
def create_dataset(num_samples):
    data = ["{0:012b}".format(i) for i in xrange(num_samples)]
    shuffle(data)
    data = [list(map(int,i)) for i in data]
    data = np.array(data)
    data = data.reshape(num_samples,12,1)

    output = np.zeros([num_samples,12],dtype=np.int)
    for sample,out in zip(data,output):
        count = 0
        for c,bit in enumerate(sample):
            if bit[0]==1:
                count += 1
            out[c] = 1 - int(count%2==0)
    return data,output

lstm_units = 64

#Input shape: (num_samples,seq_length,input_dimension)
#Output shape: (num_samples, target)
input_data = tf.placeholder(tf.float32,shape=[None,None,1])
output_data = tf.placeholder(tf.int64,shape=[None,None])
cell = LSTMCell(lstm_units,num_proj=2,state_is_tuple=True)
out,_ = tf.nn.dynamic_rnn(cell,input_data,dtype=tf.float32)   #shape: (None, 12, 2)

pred = tf.argmax(out,axis=2)    #shape: (None, 12)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=output_data,logits=out))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

correct = tf.equal(output_data,pred)
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
costs = []

for epoch in xrange(100):
    inp_data,out_data = create_dataset(4096)
    _,c,acc = sess.run([optimizer,cost,accuracy],feed_dict={input_data: inp_data, output_data: out_data})
    print("Epoch: {}, Cost: {}, Accuracy: {}%".format(epoch,c,acc*100))
    costs.append(c)

inp_data = [[[1],[1],[0],[0],[1],[1],[1],[1],[1],[0],[0],[1],[1],[1],[0],[1]]]

print("Input data:", inp_data)
print("Predicted: ", sess.run(pred,feed_dict={input_data: inp_data}))

plt.grid("off")
plt.plot(costs,label="Cost Function")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.legend()
plt.show()

sess.close()
