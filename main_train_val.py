from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import numpy as np
import sklearn as sk
import random
import csv
import re
import collections
import pickle
import sys
sys.path.append("source")
from utils import *

from joint_ablstm import *
#from rnn_train import *
#from cnn_train import *
#from att_rnn import *

embSize = 100
d1_emb_size=10
d2_emb_size=10
type_emb_size=10
numfilter = 200

#out_file = sys.argv[1]
#sent_out = sys.argv[2]

out_file = 'results/multi_results.txt'
sent_out = 'results/multi_sents_'

num_epochs = 18
#N = 4
check_point = [4,7,10,13,17]
batch_size=200
reg_para = 0.001
drop_out = 1.0

ftrain = "dataset/ddi/neg_filtered/train_data95.txt"
fval = "dataset/ddi/neg_filtered/train_data05.txt"
ftest = "dataset/ddi/neg_filtered/test_data.txt"

#wefile = "/home/sunil/embeddings/cbow_300d_gvkcorpus.txt"
wefile = "/home/sunil/embeddings/glove_100d_w9_pubmed.txt"

Tr_sent_contents, Tr_entity1_list, Tr_entity2_list, Tr_sent_lables = dataRead(ftrain)
Tr_word_list, Tr_d1_list, Tr_d2_list, Tr_type_list = makeFeatures(Tr_sent_contents, Tr_entity1_list, Tr_entity2_list)

V_sent_contents, V_entity1_list, V_entity2_list, V_sent_lables = dataRead(fval)
V_word_list, V_d1_list, V_d2_list, V_type_list = makeFeatures(V_sent_contents, V_entity1_list, V_entity2_list)

Te_sent_contents, Te_entity1_list, Te_entity2_list, Te_sent_lables = dataRead(ftest)
Te_word_list, Te_d1_list, Te_d2_list, Te_type_list = makeFeatures(Te_sent_contents, Te_entity1_list, Te_entity2_list)

print "train_size", len(Tr_word_list)
print "val_size", len(V_word_list)
print "test_size", len(Te_word_list)

train_sent_lengths, val_sent_lengths, test_sent_lengths = findSentLengths([Tr_word_list, V_word_list, Te_word_list])
sentMax = max(train_sent_lengths + val_sent_lengths + test_sent_lengths)

print "max sent length", sentMax

train_sent_lengths = np.array(train_sent_lengths, dtype='int32')
val_sent_lengths = np.array(train_sent_lengths, dtype='int32')
test_sent_lengths = np.array(test_sent_lengths, dtype='int32')


label_dict = {'false':0, 'advise': 1, 'mechanism': 2, 'effect': 3, 'int': 4}
#label_dict = {'false':0, 'true':1}

word_dict = makeWordList([Tr_word_list, V_word_list, Te_word_list])
d1_dict = makeDistanceList([Tr_d1_list, V_d1_list, Te_d1_list])
d2_dict = makeDistanceList([Tr_d2_list, V_d2_list, Te_d2_list])
type_dict = makeDistanceList([Tr_type_list, V_type_list, Te_type_list])

print "word dictonary length", len(word_dict)

# Word Embedding
wv = readWordEmb(word_dict, wefile, embSize)		

# Mapping Train
W_train =   mapWordToId(Tr_word_list, word_dict)
d1_train = mapWordToId(Tr_d1_list, d1_dict)
d2_train = mapWordToId(Tr_d2_list, d2_dict)
T_train = mapWordToId(Tr_type_list,type_dict)

Y_t = mapLabelToId(Tr_sent_lables, label_dict)
Y_train = np.zeros((len(Y_t), len(label_dict)))
for i in range(len(Y_t)):
	Y_train[i][Y_t[i]] = 1.0

#Mapping Validation
W_val =   mapWordToId(V_word_list, word_dict)
d1_val = mapWordToId(V_d1_list, d1_dict)
d2_val = mapWordToId(V_d2_list, d2_dict)
T_val = mapWordToId(V_type_list,type_dict)

Y_t = mapLabelToId(V_sent_lables, label_dict)
Y_val = np.zeros((len(Y_t), len(label_dict)))
for i in range(len(Y_t)):
	Y_val[i][Y_t[i]] = 1.0

# Mapping Test
W_test =   mapWordToId(Te_word_list, word_dict)
d1_test = mapWordToId(Te_d1_list, d1_dict)
d2_test = mapWordToId(Te_d2_list, d2_dict)
T_test = mapWordToId(Te_type_list, type_dict)
Y_t = mapLabelToId(Te_sent_lables, label_dict)
Y_test = np.zeros((len(Y_t), len(label_dict)))
for i in range(len(Y_t)):
	Y_test[i][Y_t[i]] = 1.0

#padding
W_train, d1_train, d2_train, T_train, W_val, d1_val, d2_val, T_val, W_test, d1_test, d2_test, T_test = paddData([W_train, d1_train, d2_train, T_train, W_val, d1_val, d2_val, T_val, W_test, d1_test, d2_test, T_test], sentMax) 

print "train", len(W_train)
print "test", len(W_test)


with open('train_test_rnn_data.pickle', 'wb') as handle:
	pickle.dump(W_train, handle)
	pickle.dump(d1_train, handle)
	pickle.dump(d2_train, handle)
	pickle.dump(T_train, handle)
	pickle.dump(Y_train, handle)
	pickle.dump(train_sent_lengths, handle)

	pickle.dump(W_val, handle)
	pickle.dump(d1_val, handle)
	pickle.dump(d2_val, handle)
	pickle.dump(T_val, handle)
	pickle.dump(Y_val, handle)
	pickle.dump(val_sent_lengths, handle)

	pickle.dump(W_test, handle)
	pickle.dump(d1_test, handle)
	pickle.dump(d2_test, handle)
	pickle.dump(T_test, handle)
	pickle.dump(Y_test, handle)
	pickle.dump(test_sent_lengths, handle)

	pickle.dump(wv, handle)
	pickle.dump(word_dict, handle)
	pickle.dump(d1_dict, handle)
	pickle.dump(d2_dict, handle)
	pickle.dump(type_dict, handle)	 
	pickle.dump(label_dict, handle) 
	pickle.dump(sentMax, handle)
"""
with open('train_test_rnn_data.pickle', 'rb') as handle:
	W_train = pickle.load(handle)
	d1_train= pickle.load(handle)
	d2_train= pickle.load(handle)
	T_train = pickle.load(handle)
	Y_train = pickle.load(handle)
	train_sent_lengths = pickle.load(handle)

	W_val = pickle.load(handle)
	d1_val = pickle.load(handle)
	d2_val = pickle.load(handle)
	T_val = pickle.load(handle)
	Y_val = pickle.load(handle)
	val_sent_lengths = pickle.load(handle)

	W_test = pickle.load(handle)
	d1_test = pickle.load(handle)
	d2_test = pickle.load(handle)
	T_test = pickle.load(handle)
	Y_test = pickle.load(handle)
	test_sent_lengths = pickle.load(handle)

	wv = pickle.load(handle)
	word_dict= pickle.load(handle)
	d1_dict = pickle.load(handle)
	d2_dict = pickle.load(handle)
	type_dict = pickle.load(handle)
	label_dict = pickle.load(handle)
	sentMax = pickle.load(handle)
"""

#vocabulary size
word_dict_size = len(word_dict)
d1_dict_size = len(d1_dict)
d2_dict_size = len(d2_dict)
type_dict_size = len(type_dict)
label_dict_size = len(label_dict)

rev_word_dict = makeWordListReverst(word_dict)
rev_label_dict = {0:'false', 1:'advise', 2:'mechanism', 3:'effect', 4:'int'}

fp = open(out_file, 'a+')	# keep precision recall
fsent = open(sent_out, 'w') 	# keep sentence and its results

print 'drop_out, reg_rate', drop_out, reg_para

rnn = RNN_Relation(label_dict_size, 		# output layer size
			word_dict_size, 		# word embedding size
			d1_dict_size, 			# position embedding size	
			d2_dict_size, 			# position embedding size
			type_dict_size, 		# type emb. size
			sentMax, 			# length of sentence
			wv,				# word embedding
			d1_emb_size=d1_emb_size, 	# emb. length
			d2_emb_size=d2_emb_size, 	
			type_emb_size=type_emb_size,	
			num_filters=numfilter, 		# number of hidden nodes in RNN
			w_emb_size=embSize, 		# dim. word emb
			l2_reg_lambda=reg_para		# l2 reg
		)


train_len = len(W_train)

loss_list = []

test_res = []
val_res = []

fscore_val = []
fscore_test = []


def test_step(W, sent_lengths, d1, d2, T, Y):
	n = len(W)
#	print 'n',n

	ra = n/batch_size
	samples = []
	for i in range(ra):
		samples.append(range(batch_size*i, batch_size*(i+1)))
	samples.append(range(batch_size*(i+1), n))

	acc = [] 
	pred = []
	for i in samples:
		p,a = rnn.test_step(W[i], sent_lengths[i], d1[i], d2[i], T[i], Y[i])
		# acc.extend(a)
		pred.extend(p)

#	print 'pred', len(pred)
	return pred, acc

num_batches_per_epoch = int(train_len/batch_size) + 1
iii = 0		#Check point number
for epoch in range(num_epochs):	
	shuffle_indices = np.random.permutation(np.arange(train_len))
	W_tr =  W_train[shuffle_indices]
	d1_tr = d1_train[shuffle_indices]
	d2_tr = d2_train[shuffle_indices]
	T_tr = T_train[shuffle_indices]
	Y_tr = Y_train[shuffle_indices]
	S_tr = train_sent_lengths[shuffle_indices]
	loss_epoch = 0.0
	for batch_num in range(num_batches_per_epoch):	
		start_index = batch_num*batch_size
		end_index = min((batch_num + 1) * batch_size, train_len)
		loss = rnn.train_step(W_tr[start_index:end_index], S_tr[start_index:end_index], d1_tr[start_index:end_index], 
			d2_tr[start_index:end_index], T_tr[start_index:end_index], Y_tr[start_index:end_index], drop_out)
		loss_epoch += loss

	print loss_epoch

	loss_list.append(round(loss_epoch, 5) ) 

	# if (epoch%N) == 0:

	if epoch in check_point:

		iii += 1

		saver = tf.train.Saver()
		path = saver.save(rnn.sess, 'saved_models/model_'+str(iii)+'.ckpt')

		# Validation
		y_pred_val, acc = test_step(W_val, val_sent_lengths, d1_val, d2_val, T_val, Y_val)
		y_true_val = np.argmax(Y_val, 1)
 

#		print 'y_true_val', np.shape(y_true_val)
#		print 'y_pred_val', np.shape(y_pred_val)

		fscore_val.append( f1_score(y_true_val, y_pred_val, [1,2,3,4], average='micro') )
		val_res.append([y_true_val, y_pred_val])

		# Testing
		y_pred_test, acc = test_step(W_test, test_sent_lengths, d1_test, d2_test, T_test, Y_test)
		y_true_test = np.argmax(Y_test, 1)
		
		fscore_test.append( f1_score(y_true_test, y_pred_test, [1,2,3,4], average='micro') )
		test_res.append([y_true_test, y_pred_test])

	#print 'val',fscore_val
	#print 'test',fscore_test

ind = np.argmax(fscore_val)		#Best epoch from validation set
y_true,y_pred = test_res[ind]		#actual prediction

fp.write('\n Results in Test Set (Best Index) '+str(ind)+'\n')
fp.write(str(precision_score(y_true, y_pred,[1,2,3,4], average='micro' )))
fp.write('\t')
fp.write(str(recall_score(y_true, y_pred, [1,2,3,4], average='micro' )))
fp.write('\t')
fp.write(str(f1_score(y_true, y_pred, [1,2,3,4], average='micro' )))
fp.write('\t')
fp.write('\n')

fp.write('class 1\t')
fp.write(str(precision_score(y_true, y_pred,[1], average='micro' )))
fp.write('\t')
fp.write(str(recall_score(y_true, y_pred, [1], average='micro' )))
fp.write('\t')
fp.write(str(f1_score(y_true, y_pred, [1], average='micro' )))
fp.write('\n')
 
fp.write('class 2\t')
fp.write(str(precision_score(y_true, y_pred,[2], average='micro' )))
fp.write('\t')
fp.write(str(recall_score(y_true, y_pred, [2], average='micro' )))
fp.write('\t')
fp.write(str(f1_score(y_true, y_pred, [2], average='micro' )))
fp.write('\n')

fp.write('class 3\t')
fp.write(str(precision_score(y_true, y_pred,[3], average='micro' )))
fp.write('\t')
fp.write(str(recall_score(y_true, y_pred, [3], average='micro' )))
fp.write('\t')
fp.write(str(f1_score(y_true, y_pred, [3], average='micro' )))
fp.write('\n')

fp.write('class 4\t')
fp.write(str(precision_score(y_true, y_pred,[4], average='micro' )))
fp.write('\t')
fp.write(str(recall_score(y_true, y_pred, [4], average='micro' )))
fp.write('\t')
fp.write(str(f1_score(y_true, y_pred, [4], average='micro' )))
fp.write('\n')

fp.write(str(confusion_matrix(y_true, y_pred)))
fp.write('\n')


for sent, slen, y_t, y_p in zip(W_test, test_sent_lengths, y_true, y_pred) :

	sent_l = [str(rev_word_dict[sent[kk]]) for kk in range(slen) ]
	s = ' '.join(sent_l) 
	fsent.write(s)
	fsent.write('\n')
	fsent.write( rev_label_dict[y_t] )
	fsent.write('\n')
	fsent.write( rev_label_dict[y_p] )
	fsent.write('\n')
	fsent.write('\n')

fsent.close()

rnn.sess.close()










