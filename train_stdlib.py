"""
The simplest possible "train a model" example. Zero installs.

    python3 train_stdlib.py

We teach the computer the relationship between hours studied and exam score
by fitting a straight line:   score = weight * hours + bias
"""

# 1. DATA: (hours studied, exam score). This is the training set.
data = [(1, 52), (2, 58), (3, 64), (4, 71), (5, 75), (6, 82), (7, 88), (8, 93)]

# Hold back the last two rows so we can test on data the model never saw.
train, test = data[:-2], data[-2:]

# 2. MODEL: a straight line with two numbers we need to learn.
weight, bias = 0.0, 0.0

def predict(hours):
    return weight * hours + bias

# 3. TRAINING: nudge weight and bias to shrink the error, over and over.
learning_rate = 0.01
for step in range(5000):
    grad_w = grad_b = 0.0
    for hours, score in train:
        error = predict(hours) - score
        grad_w += 2 * error * hours / len(train)
        grad_b += 2 * error / len(train)
    weight -= learning_rate * grad_w
    bias -= learning_rate * grad_b

# 4. EVALUATE on the held-out rows, then use the model on a brand-new input.
print(f"Learned: score = {weight:.2f} * hours + {bias:.2f}")
for hours, actual in test:
    print(f"{hours} hours -> predicted {predict(hours):.1f}, actual {actual}")
print(f"Prediction for 9 hours: {predict(9):.1f}")
