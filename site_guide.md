SITE GUIDE: Wearable Activity Detective

PURPOSE
This exhibit asks whether a small neural network can tell three kinds of walking apart: walking on flat ground, walking upstairs and walking downstairs. Visitors guess from an accelerometer trace, then see the network's answer. It is a class exercise, not a medical tool.

DATA
The data is the UCI "Human Activity Recognition Using Smartphones" dataset (Anguita, Ghio, Oneto, Parra and Reyes-Ortiz, 2013). 30 volunteers aged 19 to 48 wore a Samsung Galaxy S II smartphone on the waist. The phone recorded acceleration and rotation 50 times per second. The recordings are cut into windows of 2.56 seconds (128 readings) that overlap by half. This exhibit uses only the three walking activities and ignores sitting, standing and laying. The dataset's creators split the volunteers into a training group and a test group.

MODEL
Input: for each window, 561 summary numbers that the dataset's creators computed from the sensor signals (averages, spreads, frequency measures and similar). The numbers are standardized before use.
Output: three probabilities, one each for walking, walking upstairs and walking downstairs. The biggest one is the network's answer.
Network: one hidden layer of 64 units (scikit-learn MLPClassifier), trained on 3285 windows from 21 people.
Tuning: the hidden-layer size and a regularization setting were chosen using 5 of the training people (persons 3, 15, 16, 21, 22) as a validation group. The chosen network scored 98.1% on them. The final network was then retrained on all 21 training people.

TEST RESULTS
The test set has 1387 windows from 9 people who were never used for training or tuning. It has 496 walking, 471 upstairs and 420 downstairs windows.
Baseline (always guess "Walking", the most common class in the training data): 35.8% accurate.
Neural network: 93.8% accurate, 1301 of 1387 windows correct and 86 wrong.
Extra comparison, logistic regression (a simpler model): 97.2% accurate. It did better than the neural network on this test set.
Accuracy differed a lot between people: from 78.9% for person 20 to 100.0% for person 2. Person 20 had 31 of the 86 mistakes.
The most common mistakes: 36 downstairs windows were called upstairs, 27 upstairs windows were called walking, and 14 walking windows were called upstairs.
How it may do on new data: the 98.1% validation score was higher than the 93.8% test score, and accuracy varied by person, so a new person could get a result noticeably different from the average. Only 9 test people were used.

FEATURED MISTAKE
The network was 100.0% sure that a window from person 20 was "Walking upstairs", but the person was doing "Walking downstairs". Possible reasons are untested guesses: this person may walk differently from the 21 training people, and the network may be overconfident. Ideas that might help: more varied training volunteers, rescaling each person's data, or training on the raw signal.

EXHIBIT NOTES
The predictions shown on the page were computed ahead of time in Python and saved. The page does not run the network on new input. The gallery of 206 windows holds all 86 test mistakes plus 120 randomly chosen correct windows, so mistakes appear far more often there than in real use.

LIMITATIONS
Only 30 volunteers wore one phone model at the waist, doing scripted activities. Results are from one training run with one random seed, so small differences between models could be chance. The network has not been tested on other phones, other body positions, older adults or people with walking difficulties. The guide does not cover anything outside this exhibit.
