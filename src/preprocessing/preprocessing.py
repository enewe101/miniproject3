import hashlib
import json
import cv2
import numpy as np
import os
import csv
import random
import copy
import time
from collections import Counter, defaultdict
from scipy.cluster import vq
from sklearn.cluster import KMeans


class FeatureExtractor(object):

	NUM_TRAINING_IMAGES = 500
	NUM_TEST_IMAGES = 200

	RAW_DATA_DIR = 'data/raw'
	RAW_IMG_DIR = 'train_images'
	RAW_LABELS_FNAME = 'data_and_scripts/train_outputs.csv'
	PROCESSED_DIR = 'data/processed'
	SIFT_WORDS_FNAME = 'sift_words.json'
	SIFT_FEATURES_FNAME = 'sift_%s.json'
	SIFT_VOCAB_FNAME = 'sift_vocab.json'

	def __init__(
			self,
			train_image_idxs=range(1,NUM_TRAINING_IMAGES+1),
			test_image_idxs = range(1,NUM_TEST_IMAGES+1),
		):

		self.train_image_idxs = train_image_idxs
		self.test_image_idxs = test_image_idxs

		# sifter is an object used to extract sift features, made when needed
		self.sifter = None
		self.sift_features = None

		# read all of the training outputs
		self.read_outputs()
		self.inputs_dir = os.path.join(self.RAW_DATA_DIR, self.RAW_IMG_DIR)


	def compute_sift(self, idx):

		fname = os.path.join(self.inputs_dir, '%d.png' % idx)
		image = cv2.imread(fname)

		if self.sifter is None:
			self.sifter = cv2.SIFT()
			
		kp, des = self.sifter.detectAndCompute(image, None)
		return des


	def compute_all_sift_features(self):

		print 'computing all sift features from scratch'

		self.sift_features = {}
		reading_list = [
			('train', self.train_image_idxs),
			('test', self.test_image_idxs)
		]
		num_to_read = float(self.NUM_TRAINING_IMAGES + self.NUM_TEST_IMAGES)

		for data_part, idx_list in reading_list:
			for idx in idx_list:

				# show progress
				if idx % 100 == 0:
					progress = idx
					if data_part == 'test':
						progress += self.NUM_TRAINING_IMAGES 
					print '%2.1f%%' % (100 * progress / num_to_read)

				key = '%s_%d' % (data_part, idx)
				self.sift_features[key] = self.compute_sift(idx)

		return self.sift_features


	def get_sift_fname(self):
		'''
			get a string that is likely to be unique and that depends on the
			particular images that have been read
		'''
		contents = repr(self.NUM_TRAINING_IMAGES) + repr(self.NUM_TEST_IMAGES)
		hsh =  hashlib.sha224(contents).hexdigest()[:10]
		return self.SIFT_FEATURES_FNAME % hsh


	def read_sift_features(self):
		'''
			reads in the sift features and converts them into numpy arrays.
		'''

		print 'reading precomputed sift features from file...'
		precomputed_sift_fname = os.path.join(
			self.PROCESSED_DIR, self.get_sift_fname())
		
		self.sift_features =  json.loads(open(precomputed_sift_fname).read())

		# convert the sift feature vectors to np arrays
		for key in self.sift_features:
			self.sift_features[key] = np.array(self.sift_features[key])

		return self.sift_features


	def get_all_sift_features(self, use_cache=True):

		# check if we have the sift features in memory already
		if self.sift_features is not None and use_cache:
			return self.sift_features

		# otherwise, look for precomputed sift features on disk
		precomputed_sift_fname = os.path.join(
			self.PROCESSED_DIR, self.get_sift_fname())

		if os.path.isfile(precomputed_sift_fname) and use_cache:
			self.sift_features = self.read_sift_features()
			return self.sift_features


		# as a last resort, compute them from scratch.  cache them.
		self.sift_features = self.compute_all_sift_features()
		fh = open(precomputed_sift_fname, 'w')
		fh.write(json.dumps(self.sift_features))

		return self.sift_features


	def as_bag(self, dict_of_list_of_vectors):

		'''
			concatenate the sift feature lists for all images into one
			big list of features.  This is useful for k means clustering
		'''

		return np.concatenate(dict_of_list_of_vectors.values())


	def find_sift_vocab(self, k, use_cache=True):

		# open a file for writing
		fname = os.path.join(self.PROCESSED_DIR, self.SIFT_VOCAB_FNAME)
		fh = open(fname, 'w')

		# load up all of the sift features, subsample to a reasonable amount
		sift_features = self.as_bag(self.get_all_sift_features(use_cache))
		print len(sift_features)
		sift_features = random.sample(sift_features, 500)

		# do k-means clustering
		print 'clustering to build feature vocabulary'
		start = time.time()
		kmeans = KMeans(n_clusters=k, n_init=1)
		kmeans.fit(sift_features)
		print time.time() - start

		sift_features_array = np.array(sift_features)
		start = time.time()
		self.sift_words, distortion = vq.kmeans(sift_features_array, k)
		print time.time() - start

		# write the cluster centers to file
		fh.write(json.dumps(self.sift_words.tolist(), indent=2))


	def read_outputs(self):
		outputs_fname = os.path.join(self.RAW_DATA_DIR, self.RAW_LABELS_FNAME)
		outputs_reader = csv.reader(open(outputs_fname))

		# skip the headings row
		outputs_reader.next()

		# load into memory
		self.outputs = dict([row for row in outputs_reader])



# Deprecated -- scipy has a very fast implementation
class KMeansFinder(object):

	def __init__(self):
		pass


	def cluster(self, vectors, k):

		# initialize
		self.vectors = vectors
		self.k = k
		self.means = random.sample(vectors, k)
		self.changed = True
		self.point_allocation = {}
		self.clusters = {}
		self.before_first_allocation = True

		while self.changed:
			self.iterate()
			convergence =  (len(vectors) - self.changed) / float(len(vectors))
			print '%2.1f%% convergence' % (100 * convergence)

		return copy.deepcopy(self.means)


	def iterate(self):
		self.changed = 0
		self.reallocate()
		self.compute_means()
	

	def compute_means(self):
		'''
			find the midpoint of each cluster based on all the vectors 
			allocated to that cluster.
		'''

		self.means = [
			np.mean([self.vectors[p] for p in self.clusters[m]], 0).tolist()
			for m in range(self.k)
		]


	def calc_distance(self, p1, p2):
		'''
			calculates the euclidean distance between two points
		'''
		return np.sqrt(
			reduce(
				lambda x,y: x + (y[0] - y[1])**2, 
				zip(p1, p2), 0
			)
		)



	def reallocate(self):
		'''
			Finds the closest mean of all the k means.  If a point is allocated
			to a new closest mean (different from last time), then the 
			self.changed flag is set to true.  This helps determine 
			convergence.
		'''
		self.clusters = defaultdict(lambda: [])
		for point_idx, point in enumerate(self.vectors):

			if point_idx % 100 == 0:
				progress = point_idx / float(len(self.vectors))
				print '\t%2.1f%% allocated' % progress

			# calculate the distance from this point to all the means, and
			# find the closest mean
			closest_mean = self.allocate(point)

			if (
				self.before_first_allocation or 
				self.point_allocation[point_idx] != closest_mean
			):
				self.changed += 1

			self.point_allocation[point_idx] = closest_mean
			self.clusters[closest_mean].append(point_idx)

		self.before_first_allocation = False


	def allocate(self, point):
		distance, closest_mean = sorted([
			(self.calc_distance(point, m), m_idx) 
			for m_idx, m in enumerate(self.means)
		])[0]

		return closest_mean






