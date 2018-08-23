import time
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
from model import Model 
import pdb

num_classes = 2
image_shape = (160, 576)
data_dir = './data'
runs_dir = './runs'

# define epochs &  batch_size
epochs = 2
batch_size = 10
KEEP_PROB = 0.5




def optimize(cross_entropy_loss,learning_rate):

    #learning_rate = tf.placeholder(dtype = tf.float32)
    #here we use Adam optimizer 
    #define optimizer and train operation
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return train_op



def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'



    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)


        Model_vg=Model(sess,vgg_path,num_classes=num_classes)#initialize the encoder with the vgg model
        
        
        '''
        #This is to visualize the training parameters (good way to unsertand the model)
        #Here we have both variables from the vgg model and our own FCN model 
        v=tf.trainable_variables(scope=None)
        for t in v:
            print(t)
        '''

        logits,cross_entropy_loss=Model_vg.generate_loss(num_classes=2)
        
        

        # optimize function
        train_op= optimize(cross_entropy_loss=cross_entropy_loss,learning_rate=1e-4)
        # initialize session variables

        sess.run(tf.global_variables_initializer())


        for epoch in range(epochs):
            for image, label in get_batches_fn(batch_size):
                #Training the Network
                _, loss = sess.run([train_op, cross_entropy_loss],
                            feed_dict={Model_vg.Input: image, Model_vg.correct_label_mask: label, 
                            Model_vg.keep: KEEP_PROB})
                # Print data on the learning process
            print("Epoch: {}".format(epoch + 1), "/ {}".format(epochs), " Loss: {:.3f}".format(loss))


        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, Model_vg.keep, Model_vg.Input)
        pdb.set_trace()


if __name__ == '__main__':
    run()
