#!/usr/bin/env python
# Tensorflow Python Image Recognition Demo
# Model definition can be downloaded from 
# https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
import tensorflow as tf


def readData(name):
	with open(name) as f:
		return f.read()

def getExt(fname): return fname.rsplit(".",1)[-1]

# Read graph from definition file
def importGraph(name="data/tensorflow_inception_graph.pb"):
	gd=tf.GraphDef()
	gd.ParseFromString(readData(name))
	tf.import_graph_def(gd)
	return tf.get_default_graph()

def readLabels(name="data/imagenet_comp_graph_label_strings.txt"):
	with open(name) as f:
		return map(str.strip, f.readlines())

def importImage(name="data/grace_hopper.jpg"):
	data=tf.constant(readData(name))
	img = tf.image.decode_jpeg(data) if getExt(name)=="jpg" else tf.image.decode_png(data)
	fimg = tf.to_float(img)
	eimg = tf.expand_dims(fimg, 0)
	rimg = tf.image.resize_bilinear(eimg,tf.constant([224, 224]))
	normimg = tf.sub(rimg,tf.constant(117.0))
	with tf.Session() as sess:
		return sess.run(normimg)
def classify(graph, img):
	softmax2 = graph.get_operation_by_name("import/softmax2")
	with tf.Session() as sess:
		return sess.run(softmax2.outputs[0], {"import/input:0": img})

if __name__ == "__main__":
	from sys import argv
	img = importImage(argv[1] if len(argv)>1 else "data/grace_hopper.jpg")
	graph = importGraph()
	labels = readLabels()
	rc=classify(graph, img)
	with tf.Session() as sess:
		topk = sess.run(tf.nn.top_k(rc,10))
		for i in range(10):
			print "%s: %f " % (labels[topk[1][0][i]], topk[0][0][i])


