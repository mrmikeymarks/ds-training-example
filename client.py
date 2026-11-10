# /// script
# dependencies = ["scikit-learn", "joblib"]
# ///
"""
The client side: load the saved model and ask it about a flower. No training here.

    uv run client.py 6.5 3.0 5.5 2.0      (sepal length, sepal width, petal length, petal width, in cm)
    uv run client.py                      (runs a built-in example)

Needs iris_model.joblib from save_model.py in the same folder. A .joblib file is
a Python pickle: only load ones you made or trust, and use the same scikit-learn
version that saved it.
"""
import sys
import joblib
import sklearn

package = joblib.load("iris_model.joblib")
if package["sklearn_version"] != sklearn.__version__:
    print(f"Warning: model saved with scikit-learn {package['sklearn_version']}, you have {sklearn.__version__}")

measurements = [float(x) for x in sys.argv[1:]] or [6.5, 3.0, 5.5, 2.0]
if len(measurements) != 4:
    sys.exit("Give four numbers: " + ", ".join(package["inputs"]) + " (cm)")

model = package["model"]
answer = model.predict([measurements])[0]
confidence = model.predict_proba([measurements])[0][answer]
print(f"{measurements} cm -> {package['species'][answer]} ({confidence:.0%} sure)")
