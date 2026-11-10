# How to give a client an update on model work

A client update is not a lab notebook. The client wants to know four things,
in this order: what you did, what it shows, what it means for them, and what
happens next. Keep it to one page, lead with the result, use one table and
one picture, and write every sentence in plain words a non-technical reader
can repeat to their boss. Put the caveats in, briefly, so the numbers are
honest. End with a clear ask or a clear next step so the update moves the
project forward instead of just reporting on it.

Rules of thumb:

- **Headline first.** The first sentence is the result. Details come after.
- **One number per claim, rounded.** "About 97%" beats "97.333%".
- **Say what the number was measured on.** Accuracy on 45 flowers is a small
  sample, and the client should hear that from you, not discover it later.
- **Plain words.** Say "the model got 98 out of 100 right", not "achieved 0.98
  accuracy on the held-out set". Introduce a technical term only if the client
  will meet it again.
- **Show, don't list.** One chart with a one-line caption replaces a paragraph.
- **Make the picture for the client.** No file names or code in the title, only
  the models that are in the table, and every number on the chart should match
  a number in the text. If a working figure must be reused, say in one sentence
  why its numbers differ.
- **Close with the decision.** What you recommend, what you need from them, and
  when they will hear from you next.

Below is an example written for this project, using the real numbers from
`train_sklearn.py` and the client version of the chart, `plot_for_client.png`,
which `visualize.py` also produces. The "package the recommended model" step in
the example is what `save_model.py` and `client.py` do.

---

## Example update

**Subject: Flower classifier, week 1: three working models, all about 97% accurate**

Hi [client name],

**Where we are.** We now have three working models that identify an iris
flower's species from four simple measurements, built on a standard public
sample of 150 flowers. On flowers they had never seen, each one got 44 of 45
right. That single test is a small sample, so we also re-tested every model
ten times on different slices of the data (15 flowers per slice, so one
mistake there is about 7%), which gives a fairer picture:

| Model | How it decides | Average over 10 tests | Worst single test |
|---|---|---|---|
| Logistic regression | Draws straight dividing lines | 97% | 93% (1 mistake in 15) |
| K nearest neighbors | Votes among the 5 most similar flowers | 97% | 87% (2 in 15) |
| Decision tree | Asks a series of yes/no questions | 96% | 87% (2 in 15) |

**What it means.** All three are good enough for a first version. Logistic
regression was the most consistent across the ten tests, so it is our
recommended starting point. The one flower every model gets wrong sits right
on the boundary between two very similar species, which is a limit of the
measurements rather than of any one model.

**The picture.** The attached chart (`plot_for_client.png`) shows where each
model draws the border between species. Coloured zones are the model's answer
for a flower at that spot, dots are the flowers it learned from, and ringed
dots are test flowers it got wrong. The point to take from it: all three draw
the border in nearly the same place, and the only mistakes sit right on it.
The chart uses two of the four measurements so it fits on a page, which is why
its percentages differ slightly from the table.

**Next steps.** This week we will (1) confirm the result on your own sample
data, and (2) package the recommended model so your team can run it. To do
that we need about 50 of your flowers in a spreadsheet, one row per flower
with the four measurements in cm and the species your team identified, by
Friday [date].

I will send the next update on [date].

[Your name]
