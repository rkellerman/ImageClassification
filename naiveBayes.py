# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
from __future__ import division
import util
import classificationMethod
import math
import numpy as np

dictionary = {}
condProbDict = {}


class NaiveBayesClassifier(classificationMethod.ClassificationMethod):

  

  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
	self.legalLabels = legalLabels
	self.type = "naivebayes"
	self.k = 1 # this is the smoothing parameter, ** use it in your train method **
	self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
	
  def setSmoothing(self, k):
	"""
	This is used by the main method to change the smoothing parameter before training.
	Do not modify this method.
	"""
	self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
	"""
	Outside shell to call your method. Do not modify this method.
	"""  
	  
	# might be useful in your code later...
	# this is a list of all features in the training set.
	self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
	
	if (self.automaticTuning):
		kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
	else:
		kgrid = [self.k]
		
	self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
	  
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
	"""
	Trains the classifier by collecting counts over the training data, and
	stores the Laplace smoothed estimates so that they can be used to classify.
	Evaluate each value of k in kgrid to choose the smoothing parameter 
	that gives the best accuracy on the held-out validationData.
	
	trainingData and validationData are lists of feature Counters.  The corresponding
	label lists contain the correct label for each datum.
	
	To get the list of all possible features or labels, use self.features and 
	self.legalLabels.
	"""

	"*** YOUR CODE HERE ***"


	global dictionary  # keys are labels, vals are dictionaries
	global condProbDict
	i = 0

	for label in trainingLabels:

		if label in dictionary:
			temp = dictionary[label]
			temp[0] += 1
		else:
			temp = [1, {}]
			dictionary[label] = temp
		image = trainingData[i]
		for pixel in image:

			val = image[pixel]
			if pixel in dictionary[label][1]:
				temp = dictionary[label][1][pixel]
				if val == 0:
					temp[0] += 1
				else:
					temp[1] += 1
				dictionary[label][1][pixel] = temp
			else:
				if val == 0:
					temp = [1, 0]
				else:
					temp = [0, 1]
				dictionary[label][1][pixel] = temp

		i += 1

	n = len(trainingLabels)
	k = 1
		
 	for label in dictionary.keys():

 		c_y = dictionary[label][0]
		P_y = c_y / n

		pixelDict = {}

		for pixel in dictionary[label][1]:
			
			count = dictionary[label][1][pixel]

			prob = [0, 0]
			prob[0] = (count[0] + k) / ((count[0] + k) + (count[1] + k))
			prob[1] = (count[1] + k) / ((count[0] + k) + (count[1] + k))

			pixelDict[pixel] = prob


		condProbDict[label] = (P_y, pixelDict)

	return
		
  def classify(self, testData):
	"""
	Classify the data based on the posterior distribution over labels.
	
	You shouldn't modify this method.
	"""
	guesses = []
	self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
	for datum in testData:
	  posterior = self.calculateLogJointProbabilities(datum)
	  guesses.append(np.argmax(posterior))
	  self.posteriors.append(posterior)
	return guesses
	  
  def calculateLogJointProbabilities(self, datum):
	"""
	Returns the log-joint distribution over legal labels and the datum.
	Each log-probability should be stored in the log-joint counter, e.g.    
	logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
	
	To get the list of all possible features or labels, use self.features and 
	self.legalLabels.
	"""
	logJoint = util.Counter()
	
	"*** YOUR CODE HERE ***"

	logJoint = []
	for label in dictionary:
		prob = math.log(condProbDict[label][0])

		for pixel in datum:
			val = datum[pixel]

			if val == 0:
				condProb = condProbDict[label][1][pixel][0]
			else:
				condProb = condProbDict[label][1][pixel][1]

			prob += math.log(condProb)

		logJoint.append(prob)

	return np.asarray(logJoint)
  
  def findHighOddsFeatures(self, label1, label2):
	"""
	Returns the 100 best features for the odds ratio:
			P(feature=1 | label1)/P(feature=1 | label2) 
	
	Note: you may find 'self.features' a useful way to loop through all possible features
	"""
	featuresOdds = []
	   
	"*** YOUR CODE HERE ***"
	util.raiseNotDefined()

	return featuresOdds
	

	
	  
