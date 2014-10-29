import hashlib
import json
import cv2
import numpy as np
import os
import csv
import random
import copy
import time
import sys
from collections import Counter, defaultdict
from scipy.cluster import vq
try:
	from sklearn.cluster import MiniBatchKMeans
	from sklearn.externals import joblib
except ImportError:
	print 'sklearn is not currently available'


class FeatureExtractor(object):

	NUM_TRAINING_IMAGES = 50000
	NUM_TEST_IMAGES = 20000

	RAW_DATA_DIR = 'data/raw'
	TRAIN_IMAGE_DIR = 'train_images'
	TEST_IMAGE_DIR = 'test_images'
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
		print '\nGetting sift features, this may take several minutes...'
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
					print (
						'\tLoading sift features... %2.1f%%' 
						% (100 * progress / num_to_read)
					)

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
			print '\nUsing existing codebook for translation'
			fitted = joblib.load(fname)
			code_book = fitted.cluster_centers_
			return fitted, code_book

		# load up all of the sift features, subsample to a reasonable amount
		sift_features = self.as_bag(self.get_sift_features(use_cache))

		# print a message
		print '\nThere are %d sift features in total.' % len(sift_features)
		if limit is None:
			print 'Using all features.'
		else:
			print 'Randomly subsampling %d features' % limit

		if limit is not None:
			sift_features = random.sample(sift_features, limit)

		# do k-means clustering
		print (
			'\nClustering to build feature vocabulary.  This may take '
			'several hours...'
		)

		start = time.time()
		kmeans = MiniBatchKMeans(
			n_clusters=k,
			n_init=1
		)
		fitted = kmeans.fit(sift_features)
		code_book = fitted.cluster_centers_
		print time.time() - start

		# write the fitted model to file
		print ('\nClustering finished, recording cluster notes to file.')
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
		fitted, code_book = self.get_sift_vocab(
			k, limit, use_cache)
		sift_features = self.get_sift_features()

		print '\nConverting image descriptions to sift word counts'
		# for each image, make an entry in the sift words csv file 
		for data_part in ['train', 'test']:

			# Open a csv writer to write out the sift word descriptions
			sift_words_fname = self.get_sift_words_fname(data_part, k, limit)
			sift_words_fh = open(sift_words_fname, 'w')
			sift_words_writer = csv.writer(sift_words_fh)

			task_length = float(
				len(self.train_image_idxs) + len(self.test_image_idxs))

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

				# show progress
				if idx % 100 == 0:
					progress = idx 
					if data_part == 'test':
						progress += len(self.train_image_idxs)

					print 'translating... %2.1f%%' % (100*progress / task_length)

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




if __name__ == '__main__':
	if len(sys.argv) < 2 or len(sys.argv) > 3:
		raise ValueError(
			'You must supply the size of sift vocabulary to use,'
			' as an integer, e.g.'
			'\n\t`python preprocessing.py 100`'
			'\nYou may also'
			' (optionally) subsample from the total sift features with a'
			' second argument, e.g. '
			'\n\t`python preprocessing.py 100 5000`'
		)

	k = int(sys.argv[1])
	fx = FeatureExtractor()

	if len(sys.argv) > 2:
		limit = int(sys.argv[2])
		fx.as_sift_word_counts(k=k, limit=limit)

	else:
		fx.as_sift_word_counts(k=k)

	print 'Finished.'

