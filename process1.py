import os
import xml.etree.ElementTree as ET
from xml.parsers import expat 
from lxml import etree
import os, types, sys
reload(sys)
sys.setdefaultencoding('utf-8')


def read(dir_list):	
	data = []
	for d in dir_list:
	    file_list = os.listdir(d)
	    for fname in file_list:
#		print fname
  		parser = ET.XMLParser(encoding="UTF-8") #etree.XMLParser(recover=True)
  		tree = ET.parse(d+'/'+fname, parser=parser)
  		root = tree.getroot()  		
  		for sent in root:
			sent_id = sent.attrib['id']
			sent_text = sent.attrib['text'].strip()
			ent_dict = {}
			pair_list = []
			for c in sent:
				if c.tag == 'entity':
					d_type = c.attrib['type']
					d_id = c.attrib['id']
					d_ch_of=c.attrib['charOffset']
					d_text = c.attrib['text']
					ent_dict[d_id] = [d_text, d_type, d_ch_of]
				elif c.tag == 'pair':
					p_id = c.attrib['id']
					e1 = c.attrib['e1']
					entity1 = ent_dict[e1] 
					e2 = c.attrib['e2']
					entity2 = ent_dict[e2]
					ddi = c.attrib['ddi']
					if ddi == 'true':
						if 'type' in c.attrib:
							ddi = c.attrib['type']
						else:
							ddi = 'int'
					pair_list.append([entity1, entity2, ddi])
			data.append([sent_id, sent_text, pair_list])
	return data

tr_med = "/home/sunil/project/ddi/DDICorpus/Train/MedLine"
tr_drug = "/home/sunil/project/ddi/DDICorpus/Train/DrugBank"
te_med = "/home/sunil/project/ddi/DDICorpus/Test/MedLine"
te_drug = "/home/sunil/project/ddi/DDICorpus/Test/DrugBank"

def readData():
	tr_data = read([tr_med, tr_drug])
	te_data = read([te_med, te_drug])
	
#	
	fw = open('dataset/train_data.txt','w')
	for sid, stext, pair in tr_data:
		if len(pair) == 0:
			continue;

		fw.write(sid+"\t"+stext+"\n")
		for e1, e2, ddi in pair:
			fw.write(e1[0]+'\t'+e1[1]+'\t'+e1[2]+'\t'+e2[0]+'\t'+e2[1]+'\t'+e2[2]+'\t'+ddi)
			fw.write('\n')
		fw.write('\n')

	fw = open('dataset/test_data.txt','w')
	for sid, stext, pair in te_data:
		if len(pair) == 0:
			continue;
		fw.write(sid+"\t"+stext+"\n")
#		if len(pair) == 0:
#			fw.write('\n')
		for e1, e2, ddi in pair:
			fw.write(e1[0]+'\t'+e1[1]+'\t'+e1[2]+'\t'+e2[0]+'\t'+e2[1]+'\t'+e2[2]+'\t'+ddi)
			fw.write('\n') 
		fw.write('\n')
	return tr_data,te_data

a,b = readData()

 
