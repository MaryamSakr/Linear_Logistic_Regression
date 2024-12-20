import numpy as np



class LinearRegression:

    def __init__(self, x_train, y_train, alpha, iterations):
        self.x_train = x_train
        self.y_train = y_train
        self.alpha = alpha
        self.iterations = iterations
        self.m = x_train.shape[0]
        self.n = x_train.shape[1]
        self.w = np.zeros(self.n)
        self.b = 0

    def cost_function(self):
        cost = 0
        for i in range(self.m):
            estimation = np.dot(self.x_train[i],self.w) + self.b
            cost += (estimation - self.y_train[i]) ** 2
        return cost / (2*self.m)

    def linear_model(self, x):
        return np.dot(self.w, x) + self.b

    def compute_gradient(self):
        dj_dw = np.zeros(self.n)
        dj_db = 0
        for i in range(0, self.m):
            cost = self.linear_model(self.x_train[i]) - self.y_train[i]
            # for w
            for j in range(0, self.n):
                dj_dw[j] += cost * self.x_train[i][j]
            # for b
            dj_db += cost
        dj_dw /= self.m
        dj_db /= self.m
        return dj_dw, dj_db

    def gradient_descent(self):
        # need deepcopy in w_init as it is an array
        # while b is int will pass by value
        cost = self.cost_function()
        print(f"iteration {0}  COST {cost:.5f}  W = {self.w}  B = {self.b:0.5f}")
        cost_array = [cost]
        for i in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient()
            self.w = self.w - (self.alpha * dj_dw)
            self.b = self.b - (self.alpha * dj_db)
            cost = self.cost_function()
            cost_array.append(cost)
            print(f"iteration {i+1}  COST {cost:.5f}  W = {self.w}  B = {self.b:0.10f}")
        return cost_array

    def run_linear_regression(self):
        return self.gradient_descent()

    def predict(self,x):
        y = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            y[i] = self.linear_model(x[i])
        return y

