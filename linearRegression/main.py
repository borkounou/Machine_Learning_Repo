from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Creating data
np.random.seed(10)
# Creating 100 samples
# These are sample data: you can use your own data 
m = 100
X = np.linspace(0,10,m).reshape(m,1)
y = X + np.random.rand(m,1)

# Show the graph 
plt.scatter(X,y)
plt.show()
plt.savefig("graph.png")
plt.close()

# Estimator for linear regression
model = LinearRegression()

# Train the model
model.fit(X,y)

# Evaluate the model
model.score(X,y)

# Predict 

predictions = model.predict(X)
# plot the predicted data graph
plt.scatter(X,y)
plt.plot(X,predictions, c='r')
plt.show()
plt.savefig('predicted_graph')
plt.close()

# end of linear regression using scikit-learn


