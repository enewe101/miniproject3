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
try:
	from sklearn.cluster import KMeans
	from sklearn.externals import joblib
except ImportError:
	print 'sklearn is not currently available'


class FeatureExtractor(object):

	NUM_TRAINING_IMAGES = 50000
	NUM_TEST_IMAGES = 20000

	RAW_DATA_DIR = 'data/raw'
	TRAIN_IMAGE_DIR = 'train_images'
	TEST_IMAGE_DIR = 'train_images'
	RAW_LABELS_FNAME = 'data_and_scripts/train_outputs.csv'
	PROCESSED_DIR = 'data/processed'
	VOCAB_DIR = 'sift_vocab'
	SIFT_WORDS_FNAME = 'sift_words.csv'
	SIFT_WORDS_DIR = 'sift_words'
	SIFT_FEATURES_DIR = 'sift_features'
	SIFT_VOCAB_FNAME = 'sift_vocab.json'

	def __init__(
			self,
			train_image_idxs=range(1,NUM_TRAINING_IMAGES+1),
			test_image_idxs=range(1,NUM_TEST_IMAGES+1),
			limit=None
		):
		
		self.train_image_idxs = train_image_idxs[:limit]
		self.test_image_idxs = test_image_idxs[:limit]

		# sifter is an object used to extract sift features, made when needed
		self.sifter = None
		self.sift_features = None

		# read all of the training outputs
		self.read_outputs()
		self.train_img_dir=os.path.join(self.RAW_DATA_DIR,self.TRAIN_IMAGE_DIR)
		self.test_img_dir = os.path.join(self.RAW_DATA_DIR,self.TEST_IMAGE_DIR)
 


	def compute_or_load_sift(self, data_part, idx, use_cache=True):

		# first we try to load from file
		sift_fname = self.get_sift_fname(data_part, idx)
		if use_cache and os.path.isfile(sift_fname):
			return self.read_sift(sift_fname)

		# otherwise try to calculate
		image_fname = self.get_image_fname(data_part, idx)
		image = cv2.imread(image_fname)

		if self.sifter is None:
			self.sifter = cv2.SIFT()
			
		kp, des = self.sifter.detectAndCompute(image, None)

		# write the feature to disk for later
		self.write_sift(sift_fname, des)

		return des


	def read_sift(self, fname):
		'''
			reads in the sift features and converts them into numpy arrays.
		'''
		fh = open(fname)
		return np.load(fh)


	def write_sift(self, fname, des):

		fh = open(fname, 'w')
		np.save(fh, des)
		return True


	def get_image_fname(self, data_part, idx):

		fname = '%d.png' % idx
		if data_part == 'test':
			return os.path.join(self.test_img_dir, fname)
		elif data_part == 'train':
			return os.path.join(self.train_img_dir, fname)

		raise ValueError('data_part must be `test` or `train`.')
			

	def get_sift_key(self, data_part, idx):
		return '%s_%d' % (data_part, idx)


	def get_sift_fname(self, data_part, idx):
		'''
			get a string that is likely to be unique and that depends on the
			particular images that have been read
		'''
		fname = '%s.np' % self.get_sift_key(data_part, idx)
		fpath = os.path.join(self.PROCESSED_DIR, self.SIFT_FEATURES_DIR, fname)
		return fpath


	def get_sift_features(self, use_cache=True):

		# check if we have the sift features in memory already
		if self.sift_features is not None and use_cache:
			return self.sift_features

		# otherwise, compute or load each sift image's sift features
		print 'getting sift features, this may take several minutes'
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

				# we adopt a naming convention to index sift features in 
				# memory and on disk
				key = self.get_sift_key(data_part, idx)
				self.sift_features[key] = self.compute_or_load_sift(
					data_part, idx, use_cache)

		return self.sift_features


	def as_bag(self, dict_of_list_of_vectors):

		'''
			concatenate the sift feature lists for all images into one
			big list of features.  This is useful for k means clustering
		'''

		# Get a list of sift feature descriptions for all images.
		# Eliminate any descriptions for images having no sift features.
		null_array = np.array(None)
		list_of_list_of_vectors = filter(
			lambda x: not np.array_equal(x, null_array),
			dict_of_list_of_vectors.values()
		)
		return np.concatenate(list_of_list_of_vectors)


	def get_vocab_fname(self, k, limit):

		# get a unique string based on the settings that determine the vocab
		fname = 'k-%d' % k

		if limit is not None:
			fname += '.limit-%d' % limit

		# hash the settings that determine which images are included
		# to keep vocabs for different subsets of images separate
		relevant_settings = repr(self.train_image_idxs) 
		relevant_settings += repr(self.test_image_idxs)
		unique_string = hashlib.sha224(relevant_settings).hexdigest()[:6]
		fname += '.' + unique_string
		
		# add the file extension
		fname += '.pkl'

		# prepend with the right path
		return os.path.join(self.PROCESSED_DIR, self.VOCAB_DIR, fname) 


	def get_sift_vocab(self, k=100, limit=None, use_cache=True):

		'''
			gets a codebook and a trained clusterer for the set of sift 
			features in the images identified by self.train_image_idxs
			and self.test_image_idxs.
		'''
		fname = self.get_vocab_fname(k, limit)

		if use_cache and os.path.isfile(fname):
			fitted = joblib.load(fname)
			code_book = fitted.cluster_centers_
			return fitted, code_book

		# load up all of the sift features, subsample to a reasonable amount
		sift_features = self.as_bag(self.get_sift_features(use_cache))
		print 'There are %d sift features in total.' % len(sift_features)
		if limit is not None:
			sift_features = random.sample(sift_features, limit)

		# do k-means clustering
		print 'clustering to build feature vocabulary'
		start = time.time()
		kmeans = KMeans(
			n_clusters=k,
			n_init=1,
			n_jobs=8,
			precompute_distances=True
		)
		fitted = kmeans.fit(sift_features)
		code_book = fitted.cluster_centers_
		print time.time() - start

		# write the fitted model to file
		joblib.dump(fitted, fname, compress=9)

		return fitted, code_book


	def get_sift_words_fname(self, data_part, k, limit):

		fname = data_part
		# get a unique string based on the settings that determine the vocab
		fname += '.k-%d' % k

		if limit is not None:
			fname += '.limit-%d' % limit

		fname += '.csv'

		# prepend with the right path
		return os.path.join(self.PROCESSED_DIR, self.SIFT_WORDS_DIR, fname) 



	def as_sift_word_counts(self, k=100, limit=None, use_cache=True):


		# get the sift word vocabulary
		fitted, code_book = self.get_sift_vocab(k, limit, use_cache)
		sift_features = self.get_sift_features()

		# for each image, make an entry in the sift words csv file 
		for data_part in ['test', 'train']:

			# Open a csv writer to write out the sift word descriptions
			sift_words_fname = self.get_sift_words_fname(data_part, k, limit)
			sift_words_fh = open(sift_words_fname, 'w')
			sift_words_writer = csv.writer(sift_words_fh)

			# get the right index list
			if data_part == 'train':
				indexes = self.train_image_idxs 
			elif data_part == 'test':
				indexes = self.test_image_idxs
			else:
				raise ValueError('data_part must be `test` or `train`.')

			# iterate over all the images identified for this data_part
			# converting the sift feature descriptions into sift word counts
			for idx in indexes:
				key = self.get_sift_key(data_part, idx)
				these_features = sift_features[key]

				# map the sift features into sift words
				try:
					sift_words = fitted.predict(these_features)
				except ValueError:
					print 'bad feature vector:\n%s' % str(these_features)
					sift_words = []

				# format the description as counts of sift words
				counts = np.zeros(k)
				for word in sift_words:
					counts[word] += 1

				sift_words_writer.writerow([idx] + [int(x) for x in counts])



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






