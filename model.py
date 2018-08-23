import pdb
import tensorflow as tf

class Model(object):

    def __init__(self,sess,vgg_path,vgg="True",num_classes=2):
        """
        This is basically using tranfer learning
        Tranfer learning enables network to use previos knowledge inorder to generalize well and learn the new task with little amound of data
        We use pre-trained VGG net to build over model self.correct_label_mask = tf.placeholder(dtype = tf.float32, shape = (None, None, None, num_classes))
        This vgg net will help our model to achieve best results 
        """

        output_channels=num_classes #This is number of classes of the output mask

        self.correct_label_mask = tf.placeholder(dtype = tf.float32, shape = (None, None, None, num_classes))
        if vgg:
            vgg_tag = 'vgg16'
        

            #names of differnet layers and initial values of pre-trained vgg16 net
            vgg_input_tensor_name = 'image_input:0' 
            vgg_keep_prob_tensor_name = 'keep_prob:0' 
            vgg_layer3_out_tensor_name = 'layer3_out:0'
            vgg_layer4_out_tensor_name = 'layer4_out:0'
            vgg_layer7_out_tensor_name = 'layer7_out:0'

            tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

            graph = tf.get_default_graph()
            #load tensors (by name) for required layers
            self.Input = graph.get_tensor_by_name(vgg_input_tensor_name) #input layer
            self.keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name) #Differnet layer information
            self.layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
            self.layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
            self.layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
        else: 
            print("Modigy the fcn_model function since you don't use VGG fine tune") 

        self.fcn_model(output_channels) 


    def fcn_model(self,num_classes):
        #You can modify this function

        #num_classes=Number of output channels of the predicted images
 
    	#Here Building the encoder using  Pre-Trained vgg layers
    	#Different layers used to build skip connections 
    	#Please refer to the paper - https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf


    	#In the encorder we first trainfor each feature map extracted from the VGG in to another CNN feature map using 1*1 convolution
    	# We can use 1*1 convolution to reduce parameters 
    	#In the paper authors have used fully connected layers
    	with tf.variable_scope('Encoder_Model'):
        	conv_1x1_layer7 = tf.layers.conv2d(self.layer7_out, num_classes, 1, strides=(1,1), padding='same', 
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        	# conv 1x1 instead of fully connected layer over layer 4
        	conv_1x1_layer4 = tf.layers.conv2d(self.layer4_out, num_classes, 1, strides=(1,1), padding='same', 
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        	# conv 1x1 instead of fully connected layer over layer 3
        	conv_1x1_layer3 = tf.layers.conv2d(self.layer3_out, num_classes, 1, strides=(1,1), padding='same', 
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01))



    	#From here onwards we take each over encoding feature map and do upsampling with inverese convolution 
    	#upsampling and skip connections:
    	with tf.variable_scope('Decoder_Model'):
        	upsample_layer7 = tf.layers.conv2d_transpose(conv_1x1_layer7, num_classes, 4, strides=(2, 2), 
                                                 padding='same', 
                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        	skip_con_1 = tf.add(upsample_layer7, conv_1x1_layer4)

        	upsample_skip1 = tf.layers.conv2d_transpose(skip_con_1, num_classes, 4, strides=(2, 2), padding='same', 
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        	skip_con_2 = tf.add(upsample_skip1, conv_1x1_layer3)

        	self.predicted_mask_o = tf.layers.conv2d_transpose(skip_con_2, num_classes, 16, strides=(8, 8), padding='same',
                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))

 

      
    def generate_loss(self,num_classes):
        """
        This function is to create your own loss function. Here U have use cross entropy loss as given in the original paper
        """
     
        #place_holder for the correct label masks
        
        #reshape logits and labels two 2D tensors with one dimension matching no. of classes

        logits = tf.reshape(self.predicted_mask_o, (-1, num_classes))
        labels = tf.reshape(self.correct_label_mask, (-1, num_classes))

        #define loss function
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)) 


        return logits, cross_entropy_loss
  