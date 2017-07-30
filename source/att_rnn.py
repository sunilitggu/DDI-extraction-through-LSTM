import tensorflow as tf
import numpy as np


class RNN_Relation(object):
	def __init__(self, num_classes, word_dict_size, d1_dict_size, d2_dict_size, type_dict_size, sentMax, wv, w_emb_size=50, d1_emb_size=5, d2_emb_size=5, type_emb_size=5, num_filters=100, l2_reg_lambda = 0.02, pooling='max'):

		tf.reset_default_graph()
 		emb_size = w_emb_size + d1_emb_size + d2_emb_size  

		self.sent_len = tf.placeholder(tf.int64, [None], name='sent_len')
		self.w  = tf.placeholder(tf.int32, [None, None], name="x")
 		self.d1 = tf.placeholder(tf.int32, [None, None], name="x3")
		self.d2 = tf.placeholder(tf.int32, [None, None], name='x4')
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		# Initialization
#		W_wemb =    tf.Variable(tf.random_uniform([word_dict_size, w_emb_size], -1.0, +1.0))
		W_wemb = tf.Variable(wv)
		W_d1emb = tf.get_variable('W_d1emb', shape = [d1_dict_size, d1_emb_size])
		W_d2emb =   tf.get_variable('W_d2emb', shape = [d2_dict_size, d2_emb_size])
		
		# embedding Layer
		emb0 = tf.nn.embedding_lookup(W_wemb, self.w)				# word embedding NxMx50
 		emb3 = tf.nn.embedding_lookup(W_d1emb, self.d1)				# PoS embedding  NxMx5
		emb4 = tf.nn.embedding_lookup(W_d2emb, self.d2)				# PoS embedding  NxMx5


		X = tf.concat([emb0, emb3, emb4],2)					# (N,M,65)
		print 'X', X.get_shape()
		
		# recurrent Layer
		cell_f = tf.contrib.rnn.LSTMCell(num_units=num_filters, state_is_tuple=True)
		cell_b = tf.contrib.rnn.LSTMCell(num_units=num_filters, state_is_tuple=True)
		outputs, states = tf.nn.bidirectional_dynamic_rnn(
									cell_fw	=cell_f, 
									cell_bw	=cell_b, 
									dtype	=tf.float32, 
									sequence_length=self.sent_len, 
									inputs	=X
								)

		output_fw, output_bw = outputs						# NxMx100
		states_fw, states_bw = states
		print 'output_fw', output_fw.get_shape()

		h = tf.concat([output_fw, output_bw], 2)				# NxMx200
		print 'h', h.get_shape()

		M = tf.tanh(h)								# NxMx200
		W_a2 = tf.get_variable("W_a2", shape=[2*num_filters, 1]) 		# 200 x 1
		print "W_a2", W_a2.get_shape()
       		tmp3 = tf.matmul(tf.reshape(M, shape=[-1, 2*num_filters]), W_a2)  	# NMx1
		print "tmp3", tmp3.get_shape()
		alpha = tf.nn.softmax(tf.reshape(tmp3, shape=[-1, sentMax], name="att"))# NxM	
		self.ret_alpha = alpha

		# attentive sum pooling		
		alpha = tf.expand_dims(alpha, 1) 					# Nx1xM
 		print "alpha", alpha.get_shape()
		h2 =  tf.reshape(tf.matmul(alpha, h, name="r"), shape=[-1, 2*num_filters] )
		print 'h2', h2.get_shape()						# Nx200

		#drop out
		h2 = tf.nn.dropout(h2, self.dropout_keep_prob)
		h2 = tf.tanh(h2)
		#fully connected layer
		W = tf.get_variable("W", shape=[2*num_filters, num_classes])
		b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

		scores = tf.nn.xw_plus_b(h2, W, b, name="scores")			# 200x8
		print 'score', scores.get_shape()

		self.predictions = tf.argmax(scores, 1, name="predictions")
		losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
		#self.loss = tf.reduce_mean(losses) + l2_reg_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b) + tf.nn.l2_loss(W_a1) +tf.nn.l2_loss(W_a2) )
		self.loss = tf.reduce_mean(losses) + l2_reg_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b) + tf.nn.l2_loss(W_a2) )

		self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

		self.optimizer = tf.train.AdamOptimizer(1e-2)
		self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
		session_conf = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
		self.sess = tf.Session(config=session_conf)  
		self.sess.run(tf.initialize_all_variables())

	def train_step(self, W_batch, Sent_len, d1_batch, d2_batch, t_batch, y_batch, drop_out):
		#Padding data 
    		feed_dict = {
				self.w 		:W_batch,
				self.d1		:d1_batch,
				self.d2		:d2_batch,
				self.sent_len 	:Sent_len,
				self.dropout_keep_prob: drop_out,
				self.input_y 	:y_batch
	    			}
   		_, step, loss, accuracy, predictions = self.sess.run([self.train_op, self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)
    		#print ("step "+str(step) + " loss "+str(loss) +" accuracy "+str(accuracy))
		return loss

	def test_step(self, W_batch, Sent_len, d1_batch, d2_batch, t_batch, y_batch):

    		feed_dict = {
				self.w 		:W_batch,
				self.d1		:d1_batch,
				self.d2		:d2_batch,
				self.sent_len 	:Sent_len,
				self.dropout_keep_prob: 1.0,
				self.input_y 	:y_batch
	    			}
    		step, loss, accuracy, predictions = self.sess.run([self.global_step, self.loss, self.accuracy, self.predictions], feed_dict)
    		#print "Accuracy in test data", accuracy
		return predictions, accuracy

	def test_step_alpha(self, W_batch, Sent_len, d1_batch, d2_batch, t_batch, y_batch):

    		feed_dict = {
				self.w 		:W_batch,
				self.d1		:d1_batch,
				self.d2		:d2_batch,
				self.sent_len 	:Sent_len,
				self.dropout_keep_prob: 1.0,
				self.input_y 	:y_batch
	    			}
    		step, loss, accuracy, predictions, alpha = self.sess.run([self.global_step, self.loss, self.accuracy, self.predictions, self.ret_alpha], feed_dict)
		return alpha, predictions    		





