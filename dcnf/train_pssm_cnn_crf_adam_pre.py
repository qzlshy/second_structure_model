import numpy as np
import tensorflow as tf
import math
import random
import cnn_crf
import os
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"]="0"

slim = tf.contrib.slim

seqdic={'A':0, 'R':1, 'D':2, 'C':3, 'Q':4, 'E':5, 'H':6, 'I':7, 'G':8, 'N':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, 'X':20}

labeldic={'-':0, 'H':1, 'E':2}

seq=[]
for line in open("./train.fasta"):
	seq.append(line.rstrip("\n"))

seq_l=[]
for i in range(len(seq)):
	seq_l.append(len(seq[i]))

seq_l=np.array(seq_l)

pssm=np.loadtxt("./train.pssm")
cnt=np.mean(pssm)
std=np.std(pssm)
pssm=pssm-np.mean(pssm)
pssm=pssm/np.std(pssm)

t=0
pssm_list=[]
for a in seq:
	n=len(a)
	pssm_list.append(pssm[t:t+n])
	t=t+n

dssp=[]
for line in open("./train.dssp"):
	dssp.append(line.rstrip("\n"))

seq_test=[]
for line in open("./blind.fasta"):
	seq_test.append(line.rstrip("\n"))

seq_l_test=[]
for i in range(len(seq_test)):
	seq_l_test.append(len(seq_test[i]))

seq_l_test=np.array(seq_l_test)

pssm_test=np.loadtxt("./blind.pssm")
pssm_test=pssm_test-cnt
pssm_test=pssm_test/std

t=0
pssm_test_list=[]
for a in seq_test:
	n=len(a)
	pssm_test_list.append(pssm_test[t:t+n])
	t=t+n

dssp_test=[]
for line in open("./blind.dssp"):
	dssp_test.append(line.rstrip("\n"))

seq_num=[]
for a in seq:
	tmp=np.zeros(len(a))
	for i in range(len(a)):
		tmp[i]=seqdic[a[i]]
	seq_num.append(tmp)

dssp_num=[]
for a in dssp:
	tmp=np.zeros(len(a))
	for i in range(len(a)):
		tmp[i]=labeldic[a[i]]
	dssp_num.append(tmp)

seq_num_test=[]
for a in seq_test:
	tmp=np.zeros(len(a))
	for i in range(len(a)):
		tmp[i]=seqdic[a[i]]
	seq_num_test.append(tmp)

dssp_num_test=[]
for a in dssp_test:
	tmp=np.zeros(len(a))
	for i in range(len(a)):
		tmp[i]=labeldic[a[i]]
	dssp_num_test.append(tmp)

def g_data(n,m):
	l=m
	data1=np.zeros([l,20])
	label=np.zeros([l,3])
	for i in range(len(seq_num[n])):
		label[i][int(dssp_num[n][i])]=1.0
		data1[i]=pssm_list[n][i]
	data=data1.reshape([1,l,1,20])
	label=label.reshape([1,l,3])
	return data,label

def g_data_test(n,m):
	l=m
	data1=np.zeros([l,20])
	label=np.zeros([l,3])
	for i in range(len(seq_num_test[n])):
		label[i][int(dssp_num_test[n][i])]=1.0
		data1[i]=pssm_test_list[n][i]
	data=data1.reshape([1,l,1,20])
	label=label.reshape([1,l,3])
	return data,label


learning_rate = 0.0001
training_epochs = 10

x=tf.placeholder("float",[None,None,1,20],name='input')
y=tf.placeholder("float",[None,None,3],name='label')
zn=tf.placeholder(tf.int32,[None],name='zn')

y_rsp=tf.reshape(y,[-1,3])
y_max=tf.argmax(y_rsp,1)
y_max=tf.reshape(y_max,[1,-1])

cnn=cnn_crf.cnn_v(x)
cnn_rsp=tf.reshape(cnn,[-1,100])
net=slim.fully_connected(cnn_rsp, 3, activation_fn=None, scope='net2')
net_rsp=tf.reshape(net,[1,-1,3])
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(net_rsp, y_max, zn)
loss=tf.reduce_mean(-log_likelihood)
viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(net_rsp, transition_params, zn)
y_rsp_rsp=tf.cast(tf.reshape(y_max,[-1]),tf.int32)
sequence_rsp=tf.reshape(viterbi_sequence,[-1])
accuracy_ = tf.cast(tf.equal(y_rsp_rsp,sequence_rsp),tf.float32)
ac_num = tf.reduce_sum(accuracy_)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

print("train_num:",len(seq_num))

init = tf.global_variables_initializer()

num=list(range(len(seq_num)))

result=[]
saver = tf.train.Saver(max_to_keep=0)
with tf.Session() as sess:
	sess.run(init)
	for cw in range(10):
		save_name="model/my-model-"+str(cw)
		sess.run(init)
		saver.restore(sess,save_name)
		t1=0.0
		t2=0.0
		for i in range(len(seq_num_test)):
			batch_x,batch_y=g_data_test(i,len(seq_num_test[i]))
			batch_zn=[len(batch_x[0])]
			n1=sess.run(ac_num,feed_dict={x:batch_x, y:batch_y, zn:batch_zn})
			t1=t1+n1
			t2=t2+len(batch_x[0])
		print("acc:",t1/t2)
		result.append(t1/t2)

mean=np.mean(result)
std=np.std(result)
print('mean:',mean,std)

