from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Calculate model accuracy
accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

# Save trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully as model.pkl")