# /// script
# dependencies = ["scikit-learn"]
# ///
"""
The classic first data science example: teach a model to tell iris species apart,
then compare three different kinds of model on the same data.

    uv run train_sklearn.py        (uv installs scikit-learn for you, see README.md)

Everything runs on your machine. The iris dataset ships inside scikit-learn,
so nothing is downloaded at run time and no cloud account is involved.
"""
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. DATA: 150 flowers. X = 4 measurements per flower (cm), y = its species (0, 1 or 2).
iris = load_iris()
X, y = iris.data, iris.target

# 2. SPLIT: keep 30% of the flowers aside to test on, so we grade the model on flowers it never saw.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 3. TRAIN three models on the same training flowers. fit() is where the learning happens.
#    A decision tree learns yes/no questions ("petal length < 2.45?").
#    K nearest neighbors just looks up the 5 most similar training flowers and takes a vote.
#    Logistic regression draws straight dividing lines (despite the name, it is a classifier).
models = {
    "Decision tree":       DecisionTreeClassifier(random_state=0),
    "K nearest neighbors": KNeighborsClassifier(n_neighbors=5),
    "Logistic regression": LogisticRegression(max_iter=1000),   # default 100 steps is not always enough to settle
}
for name, model in models.items():
    model.fit(X_train, y_train)

    # 4. EVALUATE: score() is the fraction of test flowers it labels correctly.
    print(f"{name}: {model.score(X_test, y_test):.0%} accuracy on unseen flowers")

# 5. CROSS-VALIDATE: one split is a small sample (45 test flowers, so one mistake is 2%).
#    cross_val_score repeats the whole split-train-test cycle 10 times, each time holding
#    out a different tenth of the flowers, and reports all 10 scores. The average is a
#    fairer comparison, and the spread shows how much luck of the split matters.
print()
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=10)
    print(f"{name}: {scores.mean():.1%} average over 10 splits (lowest {scores.min():.0%}, highest {scores.max():.0%})")

# 6. USE IT on a new flower: sepal length, sepal width, petal length, petal width (cm).
print()
for name, model in models.items():
    species = model.predict([[6.5, 3.0, 5.5, 2.0]])[0]
    print(f"{name} says a 6.5 x 3.0 x 5.5 x 2.0 cm flower is: {iris.target_names[species]}")
