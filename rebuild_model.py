from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle

iris = load_iris()
X, y = iris.data, iris.target
model = DecisionTreeClassifier()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model berhasil disimpan sebagai model.pkl")
