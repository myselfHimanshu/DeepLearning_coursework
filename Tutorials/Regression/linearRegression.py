import numpy as np

def compute_error(b, m, points):
	totalError = 0
	for i in range(len(points)):
		x = points[i,0]
		y = points[i,1]
		totalError += (y-(m*x + b))**2
	
	return totalError/float(len(points)) 


def step_gradient(b, m, points, lr):
	b_grad = 0
	m_grad = 0
	N = float(len(points))

	for i in range(len(points)):
		x = points[i,0]
		y = points[i,1]
		b_grad += float("{0:.8f}".format(-(2./N)*(y - ((m*x)+b))))
		m_grad += float("{0:.8f}".format(-(2./N)*x*(y - ((m*x)+b))))

	new_b = b - (lr*b_grad)
	new_m = m - (lr*m_grad)
	#print(new_b, new_m)
	return [new_b, new_m] 

def gradient_descent_run(points, b, m, lr, iter):
	for i in range(iter):
		b,m = step_gradient(b, m, np.array(points), lr)
	return [b,m]


def run():
	#Step1: Collecting the data
	#hours studied vs test scores
	points = np.genfromtxt("data.csv", delimiter=",")

	#Step2: Defining the hyperparameters
	learning_rate = 0.0001
	initial_b = 0. #initial y-intercept
	initial_m = 0. #initial slope guess

	num_iterations = 5000
	print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, 
			compute_error(initial_b, initial_m, points)))
	print("Running....")
	[b,m] = gradient_descent_run(points, initial_b, initial_m, learning_rate, num_iterations)
	print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m,
			compute_error(b, m, points)))



if __name__ == "__main__":
	run()