from sklearn import svm, cross_validation
import numpy as np


def get_sift_words_fname(data_part, k, limit):

	fname = data_part
	# get a unique string based on the settings that determine the vocab
	fname += '.k-%d' % k 

	if limit is not None:
		fname += '.limit-%d' % limit

	fname += '.csv'

	# prepend with the right path
	return os.path.join(self.PROCESSED_DIR, self.SIFT_WORDS_DIR, fname) 


def read_sift_words(data_part='train', k=100, limit=None):
	sift_word_fh = open(get_sift_words_fname(
		data_part=data_part, k=k, limit=limit))
	reader = csv.reader(sift_word_fh)
	
	# the id's are redundant
	return [row[1:] for row in reader]


def read_image_classes():
	outputs_fh = open('data/raw/data_and_scripts/train_outputs.csv')
	reader = csv.reader(outputs_fh)

	# the id's are redundant
	return [row[1] for row in reader]



def cross_validate(C=1, gamma=1e-5, kernel='rbf', cv=5):

	# Read in the sift-word features (these must already be calculated)
	X = read_sift_words()
	Y = read_image_classes()

	# Get a learner with the desired hyperparameters
	clf = svm.SVC(C=C, gamma=gamma, kernel=kernel)

	# Test the learner in `cv`-fold cross-validation.
	scores = cross_validation.cross_val_score(clf, X, Y, cv=cv)

	# report the average score
	return np.mean(scores)


if __name__== '__main__':
	cross_validate()
