To perform classification by SIFT-bag-of-features.

1) Extract sift features by running (in this directory):
	`python preprocessing.py`
	Note, this can take several hours

2) go to ../advanced/ and run the classifier:
	`python svm.py`
	Note, this can take several hours

The score in cross-validation is printed to the console when 2) finishes.
