import matplotlib.pyplot as plt
import numpy as np

class Plot(object):
	def __init__(self):
		pass
		
	@static
	def basic_plot(self, x, y, xlabel="x-axis", ylabel="y-axis"):
		"""Generate a basic plot
		
		Params
		======
		x (list): x-axis
		y (list): y-axis
		xlabel (string): x-axis label
		ylabel (string): y-axis label
		"""
		# plot the scores
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.plot(np.arange(x, y)
		plt.ylabel(ylabel)
		plt.xlabel(xlabel)
		plt.show()