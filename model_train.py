import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("D:\CE DIPLOMA DOC\AI & Data science\Machine Learning\Dataset\projects_3.csv")

# Features and target
X = df[["Last SPI", "Test Score", "Attendance (%)"]]
y = df["Next SPI"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test accuracy
print("Model Score:", model.score(X_test, y_test))

# Save model
import pickle
pickle.dump(model, open("student_model.pkl", "wb"))
