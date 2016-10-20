
# __author__ = Benjamin Kalish

import sys
import csv
import numpy as np
import numpy.ma as ma
import scipy
import matplotlib.pyplot as plt

def nfoldpolyfit(X, Y, maxK, n, verbose):
#	NFOLDPOLYFIT Fit polynomial of the best degree to data.
#   NFOLDPOLYFIT(X,Y,maxDegree, nFold, verbose) finds and returns the coefficients 
#   of a polynomial P(X) of a degree between 1 and N that fits the data Y 
#   best in a least-squares sense, averaged over nFold trials of cross validation.
#
#   P is a vector (in numpy) of length N+1 containing the polynomial coefficients in
#   descending powers, P(1)*X^N + P(2)*X^(N-1) +...+ P(N)*X + P(N+1). use
#   numpy.polyval(P,Z) for some vector of input Z to see the output.
#
#   X and Y are vectors of datapoints specifying  input (X) and output (Y)
#   of the function to be learned. Class support for inputs X,Y: 
#   float, double, single
#
#   maxDegree is the highest degree polynomial to be tried. For example, if
#   maxDegree = 3, then polynomials of degree 0, 1, 2, 3 would be tried.
#
#   nFold sets the number of folds in nfold cross validation when finding
#   the best polynomial. Data is split into n parts and the polynomial is run n
#   times for each degree: testing on 1/n data points and training on the
#   rest.
#
#   verbose, if set to 1 shows mean squared error as a function of the 
#   degrees of the polynomial on one plot, and displays the fit of the best
#   polynomial to the data in a second plot.
#   
#
#   AUTHOR: Benjamin Kalish

	setsize = (len(X)/n)
	polyfits = [[None for x in range(maxK+1)] for y in range(n)]
	mean_squared_errors = [[0.0 for x in range(maxK+1)] for y in range(n)]


	# train and test
	for nf in range(0, n):

		#initialize training sets
		Xtrain, Ytrain = X[nf+setsize:], Y[nf+setsize:]
		Xtest, Ytest = X[nf:nf+setsize], Y[nf:nf+setsize]

		#fit polynomial regression functions to data for all values in maxK
		for deg in range(0, maxK + 1): 
			polyfits[nf][deg] = np.poly1d(np.polyfit(Xtrain, Ytrain, deg))
			
			# test points
			square_error_sum = 0.0
			for tr_ex in range (0, setsize):
				y = polyfits[nf][deg](Xtest[tr_ex])
				square_error_sum += (Ytest[tr_ex] - y) ** 2

			mean_squared_errors[nf][deg] = square_error_sum / float(setsize)

	if(verbose):

		# compute average MSE for each value of k and plot as a function of k
		average_MSEs = np.average(mean_squared_errors, axis=0)
		plt.figure(1, figsize=(15, 5))
		plt.subplot(121)
		plt.plot(range(0, 10), average_MSEs, '.-')
		plt.ylim(0,0.5)
		plt.xlabel('k')
		plt.ylabel('Average MSE')
		plt.title('Mean Squared Error vs. k')

		# find best polyfit (index of lowest MSE) and plot regression with scatterplot of data
		bestfit = np.where(average_MSEs == np.amin(average_MSEs))
		bestfit_function = np.polyfit(X, Y, bestfit[0])
		p = np.poly1d(bestfit_function)
		xp = np.linspace(-1, 1, 100)
		plt.subplot(122)
		plt.plot(X, Y, '.', xp, p(xp), '-')
		plt.ylim(-1,1)
		plt.xlabel('x')
		plt.ylabel('y')
		plt.title('Best-Fitting Polynomial Regression')
		# plt.savefig("PolyPlot.png")
		plt.show()

		return bestfit_function




def main():
	# read in system arguments, first the csv file, max degree fit, number of folds, verbose
	rfile = sys.argv[1]
	maxK = int(sys.argv[2])
	nFolds = int(sys.argv[3])
	verbose = bool(sys.argv[4])
	
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X = []
	Y = []
	# put the x coordinates in the list X, the y coordinates in the list Y
	for i, row in enumerate(dat):
		if i > 0:
			X.append(float(row[0]))
			Y.append(float(row[1]))
	X = np.array(X)
	Y = np.array(Y)
	nfoldpolyfit(X, Y, maxK, nFolds, verbose)

if __name__ == "__main__":
	main()
