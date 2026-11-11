# Predicting Machine Faults with Spark ML and MLflow

Machines usually give warning signs before they fail. This project uses readings from vibration sensors to predict whether a machine has a fault, so it can be repaired before it breaks down.

I built it on Databricks as part of my MSc in Data Science at the University of Salford, using Spark ML for the model and MLflow to track every experiment.

## Results

| Model | Test accuracy |
| --- | --- |
| Decision tree, default settings | 95.2% |
| Decision tree, tuned with grid search | **96.1%** |

The best settings were entropy as the split measure, a maximum depth of 7 and 64 bins.

## The data

- 9,292 readings from vibration sensors, each with 20 numeric features
- Each reading is labelled `fault_detected` (1) or not (0)
- The classes are perfectly balanced, 4,646 of each, so accuracy is a fair way to judge the model

## How I built it

1. **Loaded and cleaned the data** in PySpark, dropping incomplete rows and checking the class balance.
2. **Prepared the features** with `RFormula`, which turns all the sensor columns into a single feature vector and the fault flag into the label.
3. **Split the data** 70% for training and 30% for testing, with a fixed seed so the results can be reproduced.
4. **Trained a baseline** decision tree to see how far a simple model gets.
5. **Tuned it** with `TrainValidationSplit` and a grid of 18 combinations:
   - impurity: gini or entropy
   - maximum depth: 2, 5 or 7
   - maximum bins: 16, 32 or 64
6. **Tracked every run in MLflow** using autologging, so each model's settings and scores were saved without extra code.
7. **Loaded the best model back from MLflow** and used it to make predictions, the same way a production system would pick up a trained model.

## What I learned

- A well-tuned simple model can be the right answer. The decision tree was already strong, and tuning added another point without making it harder to explain.
- Experiment tracking matters even on a small project. With MLflow I could compare runs side by side instead of relying on notes.
- Loading the model from MLflow rather than keeping it in memory is a small step, but it's the step that turns a notebook experiment into something you could deploy.

## What I'd do next

- Look at precision and recall, not just accuracy. Missing a real fault usually costs far more than a false alarm.
- Try a random forest or gradient-boosted trees and compare them against this baseline.
- Check which sensor features drive the predictions, so engineers know what to monitor.

## Tech

Python · PySpark · Spark ML · MLflow · Databricks · Matplotlib

## Running it

The notebook was written for Databricks. Upload `FaultDataset.csv` to `/FileStore/tables/`, attach the notebook to a cluster with MLflow available, and run the cells in order.

---

I'm Yinka, an AI Solutions Engineer and data scientist in Manchester. More of my work is on my [GitHub profile](https://github.com/yinks0067), and you can find me on [LinkedIn](https://www.linkedin.com/in/yinka-sunmola/).
