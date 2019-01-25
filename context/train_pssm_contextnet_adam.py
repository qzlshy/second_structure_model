import numpy as np
import tensorflow as tf
import math
import random
import context
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

slim = tf.contrib.slim

seqdic={'A':0, 'R':1, 'D':2, 'C':3, 'Q':4, 'E':5, 'H':6, 'I':7, 'G':8, 'N':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, 'X':20}

labeldic={'-':0, 'H':1, 'E':2}

seq=[]
for line in open("train.fasta"):
	seq.append(line.rstrip("\n"))

seq_l=[]
for i in range(len(seq)):
	seq_l.append(len(seq[i]))

seq_l=np.array(seq_l)

pssm=np.loadtxt("train.pssm")
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
for line in open("train.dssp"):
	dssp.append(line.rstrip("\n"))

seq_test=[]
for line in open("blind.fasta"):
	seq_test.append(line.rstrip("\n"))

seq_l_test=[]
for i in range(len(seq_test)):
	seq_l_test.append(len(seq_test[i]))

seq_l_test=np.array(seq_l_test)

pssm_test=np.loadtxt("blind.pssm")
pssm_test=pssm_test-cnt
pssm_test=pssm_test/std

t=0
pssm_test_list=[]
for a in seq_test:
	n=len(a)
	pssm_test_list.append(pssm_test[t:t+n])
	t=t+n

dssp_test=[]
for line in open("blind.dssp"):
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
net=context.cnn_context(x)

net_rsp=tf.reshape(net,[-1,3])
y_rsp=tf.reshape(y,[-1,3])
net_soft=tf.nn.softmax(net_rsp)


c_e=tf.nn.softmax_cross_entropy_with_logits(logits=net_rsp, labels=y_rsp)
loss=tf.reduce_mean(c_e)

accuracy_ = tf.cast(tf.equal(tf.argmax(net_rsp,1),tf.argmax(y_rsp,1)),tf.float32)
ac_num = tf.reduce_sum(accuracy_)
accuracy = tf.reduce_mean(accuracy_)

saver = tf.train.Saver(max_to_keep=0)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

print("train_num:",len(seq_num))

init = tf.global_variables_initializer()

num=list(range(len(seq_num)))

with tf.Session() as sess:
	sess.run(init)
	for cw in range(10):
		save_name="model/my-model-"+str(cw)
		acc=0.0
		sess.run(init)
		for epoch in range(training_epochs):
			print("epoch:",epoch)
			random.shuffle(num)
			t1=0
			t2=0
			for i in range(len(seq_num)):
				batch_x, batch_y=g_data(num[i],len(seq_num[num[i]]))
				_,l1,n1=sess.run([optimizer,loss,ac_num],feed_dict={x:batch_x, y:batch_y})
				t1=t1+n1
				t2=t2+len(seq_num[num[i]])
			print("train:",t1/t2)
			t1=0.0
			t2=0.0
			for i in range(len(seq_num_test)):
				batch_x, batch_y=g_data_test(i,len(seq_num_test[i]))
				n1=sess.run(ac_num,feed_dict={x:batch_x, y:batch_y})
				t1=t1+n1
				t2=t2+len(seq_num_test[i])
			if t1/t2>acc:
				saver.save(sess, save_name)
				acc=t1/t2
			print("acc:",t1/t2)

