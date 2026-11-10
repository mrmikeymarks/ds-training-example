# /// script
# dependencies = ["scikit-learn", "joblib"]
# ///
"""
Package the recommended model so another program can use it.

    uv run save_model.py        (writes iris_model.joblib next to this file)

Once you have picked a model, you train it on ALL the data you have (no more
held-out test set, that job is done) and save the fitted model to a file.
"""
import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()

# Train the winner from train_sklearn.py on every flower we have.
model = LogisticRegression(max_iter=1000).fit(iris.data, iris.target)

# Save the fitted model plus everything a client needs to interpret its answers.
package = {
    "model": model,
    "species": list(iris.target_names),                  # 0, 1, 2 -> names
    "inputs": ["sepal length", "sepal width", "petal length", "petal width"],   # in cm, in this order
    "sklearn_version": sklearn.__version__,              # the client should match this
}
joblib.dump(package, "iris_model.joblib")
print(f"Saved iris_model.joblib (trained on {len(iris.data)} flowers with scikit-learn {sklearn.__version__})")
