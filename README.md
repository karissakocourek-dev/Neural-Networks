# Wearable Activity Detective

An interactive health data exhibit. A small neural network tries to tell walking, walking upstairs and walking downstairs apart from smartphone sensor data. Visitors guess from an accelerometer trace, then see the network's answer, explore its mistakes, and can ask an optional in-browser chatbot (WebLLM) about the exhibit.

## Data credit

Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. (2013). *Human Activity Recognition Using Smartphones*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

The dataset itself is not stored in this repository. Download it from the link above to rerun the training.

## Files

| File | What it is |
| --- | --- |
| `index.html`, `style.css`, `app.js` | The exhibit website |
| `site_guide.md` | The guide the chatbot answers from |
| `results.json` | Metrics, tuning results and the featured mistake |
| `gallery.json` | Held-out test windows with traces and the network's predictions |
| `train_and_evaluate.py` | Trains the network, compares it with baselines, writes the two JSON files |
| `CHAT_TESTS.md` | Chatbot test log |

## Results (test set: 1387 windows from 9 people not used for training or tuning)

- Baseline (always "Walking"): 35.8%
- Neural network (one hidden layer, 64 units): 93.8%
- Logistic regression (extra comparison): 97.2%

## Rerun the training

```
python train_and_evaluate.py "path/to/UCI HAR Dataset" output_folder
```

Needs numpy and scikit-learn. Copy the two JSON files it writes next to `index.html`. The site guide numbers were written from `results.json`, so update `site_guide.md` if you retrain.

## Run the site

Open the GitHub Pages link. Opening `index.html` directly from a folder will not load the data files because of browser rules.
