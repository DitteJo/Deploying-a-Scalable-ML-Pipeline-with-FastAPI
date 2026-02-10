# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model uses RandomForestClassifier.
Contains the following pipeline components:
    - process_data()
    - train_model()
    - inference()
    - compute_model_metrics()
    - performance_on_catagorical_slice()

## Intended Use
This model is designed to predict whether an individual earns more or less than 50K per year based on demographic and employement features such as age, marital status, occupation and education.

## Training Data
This model is trained with Census Income Dataset from UCI
Dataset: https://archive.ics.uci.edu/dataset/20/census+income 

## Evaluation Data
This model is evaluated using a training set that consists of 80% of the data and a test set that consists of the other 20% of the data.

## Metrics
Precision: 0.7229, Recall: 0.6039, F1: 0.6580

## Ethical Considerations
This model uses the census dataset from 1994 so it may not reflect modern demographics or employment. If the dataset contains biases, so may this model.

## Caveats and Recommendations
This model should not be used for real-world decision-making, only for learning.