import numpy as np
import sklearn as sk
import random
import csv
import re
import collections
#from geniatagger import GeniaTagger
#tagger = GeniaTagger("/home/sunil/packages/geniatagger-3.0.2/geniatagger")
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
import pickle

def preProcess(sent):
	sent = sent.lower()
	sent = sent.replace('/',' ')

#	sent = sent.replace('(','')
#	sent = sent.replace(')','')
#	sent = sent.replace('[','')
#	sent = sent.replace(']','')
	sent = sent.replace('.','')
#	sent = sent.replace(',',' ')
#	sent = sent.replace(':','')
#	sent = sent.replace(';','')
	
	sent = tokenizer.tokenize(sent)
	sent = ' '.join(sent)
	sent = re.sub('\d', 'dg',sent)
	return sent


def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1


def makePaddedList(sent_contents, maxl, pad_symbol= '<pad>'):	 
	T = []
 	for sent in sent_contents:
		t = []
		lenth = len(sent)
		for i in range(lenth):
			t.append(sent[i])
		for i in range(lenth,maxl):
			t.append(pad_symbol)
		T.append(t)	

	return T

def makeWordList(lista):
	sent_list = sum(lista, [])
	wf = {}
	for sent in sent_list:
		for w in sent:
 			if w in wf:
				wf[w] += 1
			else:
				wf[w] = 0

	wl = []	
	i = 1

	wl.append('<pad>')
	wl.append('<unkown>')
	for w,f in wf.iteritems():		
		wl.append(w)
	return wl

def makeDistanceList(lista):
	sent_list = sum(lista, [])
	wf = {}
	for sent in sent_list:
		for w in sent:
 			if w in wf:
				wf[w] += 1
			else:
				wf[w] = 0

	wl = []	
	i = 1
	for w,f in wf.iteritems():		
		wl.append(w)
	return wl

def makeWordListReverst(word_list):
	wl = {}
	v = 0
	for k in word_list:
		wl[v] = k
		v += 1
	return wl


def mapWordToId(sent_contents, word_list):
	T = []
	for sent in sent_contents:
		t = []
		for w in sent:
			t.append(word_list.index(w))
		T.append(t)
	return T

def mapLabelToId(sent_lables, label_dict):
	if len(label_dict) > 2:
		return [label_dict[label] for label in sent_lables]
	else:
		return [int (label != 'false') for label in sent_lables]

"""	
Given his tenuous respiratory status , he was transferred to the FICU with closer observation .
his tenuous respiratory status|1|4|problem
closer observation|13|14|test
TeCP
"""
						
def makeFeaturesCRE( fname ):

	print "Reading data and Making features"
	fp = open(fname, 'r')
	samples = fp.read().strip().split('\n\n')
	 
  	sent_list  	= []		#2-d array [[w1,w2,....] ...]
  	sent_lables    	= []		#1-d array
	d1_list	 	= []
	d2_list 	= []
	type_list 	= []
	length_list 	= []
  	for sample in samples:
		
		sent, entity1, entity2, relation = sample.strip().split('\n')
		# PreProcess
		sent = sent.lower()			# pre processing
		sent = re.sub('\d', 'dg',sent)		# Pre processing

		e1, e1_s, e1_e, e1_t = entity1.split('|') 
		e2, e2_s, e2_e, e2_t = entity2.split('|')

		word_list = sent.split()
		word_1 = word_list[0:int(e1_s)]  
		word_2 = word_list[int(e1_e)+1:int(e2_s)]
		word_3 = word_list[int(e2_e)+1:]
		words = word_1 + [e1_t] + word_2 + [e2_t] + word_3
		s1 = words.index(e1_t)
		s2 = words.index(e2_t) 

		# distance1 feature	
		d1 = []
		for i in range(len(words)):
		    if i < s1 :
			d1.append(str(i - s1))
		    elif i > s1 :
			d1.append(str(i - s1 ))
		    else:
			d1.append('0')

		#distance2 feature		
		d2 = []
		for i in range(len(words)):
		    if i < s2:
			d2.append(str(i - s2))
		    elif i > s2:
			d2.append(str(i - s2))
		    else:
			d2.append('0')

		#type feature
		t = []
		for i in range(len(words)):
			t.append('Out')
		t[s1] = e1_t		
		t[s2] = e2_t

		sent_lables.append(relation)
		sent_list.append(words)
 		d1_list.append(d1)
		d2_list.append(d2)
 		type_list.append(t) 
		length_list.append(len(words))

    	return sent_list, d1_list, d2_list, type_list, length_list, sent_lables



def dataRead(fname):
	print "Input File Reading"
	fp = open(fname, 'r')
	samples = fp.read().strip().split('\n\n')
	sent_lengths   = []		#1-d array
  	sent_contents  = []		#2-d array [[w1,w2,....] ...]
  	sent_lables    = []		#1-d array
  	entity1_list   = []		#2-d array [[e1,e1_t] [e1,e1_t]...]
  	entity2_list   = []		#2-d array [[e1,e1_t] [e1,e1_t]...]
  	for sample in samples:
		sent, entities, relation = sample.strip().split('\n')
#		if len(sent.split()) > 100:
#			continue
		e1, e1_t, e2, e2_t = entities.split('\t') 
		sent_contents.append(sent.lower())
		entity1_list.append([e1, e1_t])
		entity2_list.append([e2, e2_t])
		sent_lables.append(relation)

  	return sent_contents, entity1_list, entity2_list, sent_lables 


def makeFeatures(sent_list, entity1_list, entity2_list):
	print 'Making Features'
	word_list = []
	d1_list = []
	d2_list = []
	type_list = []
 	for sent, ent1, ent2 in zip(sent_list, entity1_list, entity2_list):
		sent = preProcess(sent)
#		print sent
		sent_list1 = sent.split()
 		
		entity1 = preProcess(ent1[0]).split()
		entity2 = preProcess(ent2[0]).split()
  		s1 = sent_list1.index('druga')
		s2 = sent_list1.index('drugb') 
		# distance1 feature	
		d1 = []
		for i in range(len(sent_list1)):
		    if i < s1 :
			d1.append(str(i - s1))
		    elif i > s1 :
			d1.append(str(i - s1 ))
		    else:
			d1.append('0')
		#distance2 feature		
		d2 = []
		for i in range(len(sent_list1)):
		    if i < s2:
			d2.append(str(i - s2))
		    elif i > s2:
			d2.append(str(i - s2))
		    else:
			d2.append('0')
		#type feature
		t = []
		for i in range(len(sent_list1)):
			t.append('Out')
		t[s1] = ent1[1]		
		t[s2] = ent2[1]

		word_list.append(sent_list1)
 		d1_list.append(d1)
		d2_list.append(d2)
 		type_list.append(t) 

    	return word_list, d1_list, d2_list, type_list

def readWordEmb(word_list, fname, embSize=100):
	print "Reading word vectors"
	wv = []
	wl = []
	with open(fname, 'r') as f:
		for line in f :			
			vs = line.split()
			if len(vs) < embSize :
				continue
			vect = map(float, vs[1:])
			wv.append(vect)
			wl.append(vs[0])
	wordemb = []
	count = 0
	for word in word_list:
		if word in wl:
			wordemb.append(wv[wl.index(word)])
		else:
			count += 1
			wordemb.append(np.random.rand(embSize))
			#wordemb.append( np.random.uniform(-np.sqrt(3.0/embSize), np.sqrt(3.0/embSize) , embSize) )

	wordemb[word_list.index('<pad>')] = np.zeros(embSize)
	wordemb = np.asarray(wordemb, dtype='float32')

	print "number of unknown word in word embedding", count
	return wordemb

def findLongestSent(Tr_word_list, Te_word_list):
	combine_list = Tr_word_list + Te_word_list
	a = max([len(sent) for sent in combine_list])
	return a
 
def findSentLengths(tr_te_list):
	lis = []
	for lists in tr_te_list:
		lis.append([len(l) for l in lists])
	return lis
 
def paddData(listL, maxl): #W_batch, d1_tatch, d2_batch, t_batch)
	rlist = []
 	for mat in listL:		
		mat_n = []
		for row in mat:
			lenth = len(row)
			t = []
			for i in range(lenth):
				t.append(row[i])
			for i in range(lenth, maxl):
				t.append(0)
			mat_n.append(t)
		rlist.append(np.array(mat_n)) 
	return rlist

def makeBalence(Tr_sent_contents, Tr_entity1_list, Tr_entity2_list, Tr_sent_lables):
	sent_contents=[]; entity1_list=[]; entity2_list=[]; sent_lables=[];
	other = []
	clas = []
	for sent,e1,e2,lab in zip(Tr_sent_contents, Tr_entity1_list, Tr_entity2_list, Tr_sent_lables):
		if lab == 'false' :
			other.append([sent, e1, e2, lab])
		else:
			clas.append([sent, e1, e2, lab])

 	random.shuffle(other)
	
	neg = other[0 : 3*len(clas)]
	l = neg+clas
	for sent,e1,e2,lab in l:
		sent_contents.append(sent)
		entity1_list.append(e1)
		entity2_list.append(e2)
		sent_lables.append(lab)
	return sent_contents, entity1_list, entity2_list, sent_lables


