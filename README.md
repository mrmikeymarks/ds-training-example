# Simplest data science training examples

Small Python examples that train a model on your own machine. No Google, no
OpenAI, no cloud account, no API key. Each one follows the four steps every
data science project uses: **data -> split -> train -> evaluate**.

## Setup

`train_stdlib.py` needs nothing but Python, so you can run it straight away.
Everything else needs a few libraries, which are listed in `requirements.txt`.
The usual way to install them is into a **virtual environment**: a private
folder of packages just for this project, so nothing you install here touches
the rest of your computer.

You need Python 3.9 or newer. Check what you have:

```bash
python3 --version
```

**1. Create the virtual environment.** Run this once, from inside the project
folder. It makes a folder called `.venv`.

```bash
python3 -m venv .venv
```

**2. Turn it on.** Do this in every new terminal window before working on the
project.

```bash
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`, which is how you can tell it is on.

**3. Install the libraries.** Run this once, and again if `requirements.txt`
ever changes.

```bash
pip install -r requirements.txt
```

**4. Check that it worked.** You should see an accuracy of about 98% for each
of three models.

```bash
python3 train_sklearn.py
```

Good to know:

- `deactivate` turns the environment off again. Closing the terminal does the
  same.
- To start over, delete the `.venv` folder and repeat the steps.
- On Windows, type `python` instead of `python3`, and turn the environment on
  with `.venv\Scripts\activate`.
- If you use the `virtualenv` tool, `virtualenv .venv` does the same job as
  step 1.

### Prefer uv?

If you have [uv](https://docs.astral.sh/uv/), you can skip the setup. Type
`uv run` in place of `python3`, for example `uv run train_sklearn.py`. Each
script names the libraries it needs at the top of the file, and `uv` installs
them into a throwaway environment the first time you run it. The notebook has
its own `uv` command in section 5.

## 1. `train_stdlib.py` — zero installs

Fits a straight line (linear regression) to hours-studied vs exam-score data
using plain Python. Nothing to install, so it runs without the setup above.

```bash
python3 train_stdlib.py
```

What you learn: a "model" is just a formula with numbers in it, and "training"
means adjusting those numbers to shrink the error on the training data.

## 2. `train_sklearn.py` — the classic iris example

Trains three models, a decision tree, k nearest neighbors and logistic
regression, to classify iris flowers with scikit-learn, the standard Python
machine learning library, and prints each one's accuracy on the same held-out
test flowers, then a fairer 10-way cross-validated average. The dataset ships inside scikit-learn, so
nothing is downloaded when the script runs.

```bash
python3 train_sklearn.py
```

What you learn: the real-world workflow. Load a dataset, hold back a test set,
fit a model in one line, measure accuracy on data it never saw, then use it.
All three models score 98% on the single split because iris is an easy dataset and
the test set is only 45 flowers, so one mistake is worth about 2%. That is why
the script then runs 10-fold cross-validation: it repeats the split-train-test
cycle 10 times with a different held-out tenth each time and averages the
scores. The average is a fairer comparison, and the lowest and highest scores
show how much the luck of the split matters.

## 3. `visualize.py` — pictures of both

Draws two PNGs so you can see what the scripts did instead of reading numbers.

```bash
python3 visualize.py
```

- `plot_regression.png`: the hours-vs-score points with the learned line, and the
  training error shrinking step by step. That shrinking curve *is* training.
- `plot_iris.png`: the iris flowers on their two petal measurements, then the
  border each model draws between species. The decision tree makes box-shaped
  cuts, logistic regression draws straight diagonal lines, k=1 traces a jagged
  edge around individual flowers, and k=5 smooths that edge out. Ringed dots
  are test flowers a model got wrong.
- `plot_for_client.png`: the same idea trimmed to the three shortlisted models
  with no code names, ready to attach to a client update. `CLIENT_UPDATE.md`
  explains how to write that update and includes a worked example.

The iris models in this figure use only petal length and width so the border
can be drawn on a flat page, which is why their accuracy differs slightly from
the four-measurement scores printed by `train_sklearn.py`.

## 4. `save_model.py` and `client.py` — handing the model to another program

Once you have chosen a model, you train it on all your data and save the
fitted model to a file. Any other Python program can then load that file and
ask it questions without training anything.

```bash
python3 save_model.py
```

```bash
python3 client.py 6.5 3.0 5.5 2.0
```

`save_model.py` writes `iris_model.joblib`, which bundles the fitted logistic
regression with the species names, the input order, and the scikit-learn
version. `client.py` loads it and prints the species for the four measurements
you pass on the command line, with how sure the model is.

Two things to know about `.joblib` files: they only work reliably with the same
scikit-learn version that saved them, which is why the version is stored and
checked, and they are Python pickles, so only load files you made or trust.

## 5. `walkthrough.ipynb` — step through all of it in a notebook

A Jupyter notebook that runs the same code as the scripts above one step at a
time, with the tables and charts drawn right under each step. Part 1 is the
straight line from `train_stdlib.py`, Part 2 is the three iris models from
`train_sklearn.py` with the pictures from `visualize.py`, and Part 3 saves a
model and loads it back like `save_model.py` and `client.py`.

```bash
jupyter lab walkthrough.ipynb
```

That opens JupyterLab in your browser. Click a cell and press Shift + Enter to
run it and move to the next one. The outputs of a full run are saved in the
file, so you can also read it without running anything, for example on GitHub
or in VS Code.

With `uv` instead of the virtual environment:

```bash
uv run --python 3.13 --with-requirements requirements.txt jupyter lab walkthrough.ipynb
```

The `--python 3.13` part stops `uv` from borrowing whatever older Python
environment happens to be active in your terminal.

Part 3 of the notebook writes its own file, `notebook_model.joblib`, so it does
not replace the `iris_model.joblib` that `client.py` reads.

## Where to go next

- Add a fourth model, `RandomForestClassifier` from `sklearn.ensemble`, to the
  `models` dictionary and see how it compares.
- Change `n_neighbors=5` to 1 or 15 and watch how accuracy moves.
- Load your own CSV with `pandas.read_csv` and feed its columns to the same code.
