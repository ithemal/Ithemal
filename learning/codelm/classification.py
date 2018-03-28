#classifying if it's an opcode or not
#non - queued data

import operator
import tensorflow as tf
import numpy as np
import random  
    
def generate_datasets(x, y, percentage):
    assert x.shape[0] == y.shape[0]
    size = y.shape[0]
    split = (size * percentage) // 100
    train_x  = x[:split,:]
    train_y = y[:split]
    test_x = x[(split + 1):,:]
    test_y = y[(split + 1):]
    return train_x, train_y, test_x, test_y
    
  
def generate_batch(x, y, batch_size):

  population = range(x.shape[0])
  embedding_size = x.shape[1]
  selected = random.sample(population,batch_size)
  batch_x = np.ndarray(shape = [batch_size, embedding_size])
  batch_y = np.ndarray(shape = [batch_size,1])
  for i,index in enumerate(selected):
      batch_x[i] = x[index,:]
      batch_y[i] = y[index]

  return batch_x, batch_y


def train(train_x, train_y, test_x, test_y, batch_size):

    embedding_size = train_x.shape[1]
    learning_rate = 1.0
    epochs = 10

    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape = [None, embedding_size])
        y = tf.placeholder(tf.float32, shape = [None, 1])
        
        W = tf.Variable(tf.random_normal(shape = [embedding_size, 1], dtype = tf.float32))
        b = tf.Variable(tf.random_normal(shape = [1], dtype = tf.float32))
        
        output = tf.add(tf.matmul(x,W),b)
        output = tf.sigmoid(output)
        output = tf.clip_by_value(output, 1e-10, 0.999999)
        

        centropy = -tf.reduce_mean(y * tf.log(output)
                                                      + (1 - y) * tf.log(1 - output))
        #centropy = tf.losses.sigmoid_cross_entropy(y,output)
    
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(centropy)
        
        init_op = tf.global_variables_initializer()
        
        #accuracy operator
        accuracy = tf.reduce_mean(tf.abs(tf.round(output) - y))
    

    with tf.Session(graph=graph) as sess:
        # initialise the variables
        sess.run(init_op)
        total_batch = int(train_x.shape[0]) / batch_size
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = generate_batch(train_x, train_y, batch_size)
                _, c = sess.run([optimizer, centropy], 
                                feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
        print(sess.run(accuracy, feed_dict={x: test_x, y: test_y}))




