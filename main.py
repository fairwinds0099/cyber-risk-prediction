import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,roc_curve,recall_score,roc_auc_score,accuracy_score,precision_score,classification_report,confusion_matrix
from DataAnalyzeActions import DataAnalyzeActions
from DataPrepActions import DataPrepActions

dataPrepActions = DataPrepActions()
dataAnalyzeActions = DataAnalyzeActions()
randomOverSampler = RandomOverSampler(random_state=0)
randomForestClassifier = RandomForestClassifier(random_state=0)

if __name__ == '__main__':

    X, y = dataPrepActions.fetchAndCleanDataframe()

    # splitting raw dataset to three: train validation test split and printing sizes
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    xTrainVldtn, xVldtn, yTrainVldtn, yVldtn = train_test_split(xTrain, yTrain, test_size=0.25, random_state=1, stratify=yTrain)
    print('=================  data size =========================')
    print('x (raw) size:', len(X), 'y (raw) size:', len(y))
    print('xTrain size:', len(xTrain), 'yTrain size:', len(yTrain))
    print('xTest size:', len(xTest), 'yTest size:', len(yTest))
    print('xTrainVldtn size:', len(xTrainVldtn), 'yTrainVldtnSize:', len(yTrainVldtn))

    randomForestClassifier = RandomForestClassifier(random_state=0)
    randomForestClassifier.fit(xTrainVldtn, yTrainVldtn)
    predictions = randomForestClassifier.predict(xVldtn)
    print('Classification report:]\n', classification_report(yVldtn, predictions))
    print('Confusion matrix:\n', confusion_matrix(yVldtn, predictions))
    del randomForestClassifier

    randomForestClassifier = RandomForestClassifier(random_state=0)
    xTrainVldtnOverSampled, yTrainValidationOverSampled = randomOverSampler.fit_sample(xTrainVldtn, yTrainVldtn)
    randomForestClassifier.fit(xTrainVldtnOverSampled, yTrainValidationOverSampled)
    predictions = randomForestClassifier.predict(xVldtn)
    print('Classification report:]\n', classification_report(yVldtn, predictions))
    print('Confusion matrix:\n', confusion_matrix(yVldtn, predictions))

    probabilities = randomForestClassifier.predict_proba(xVldtn)[:, 1]
    print(probabilities)

    falsePositiveRates, truePositiveRates, thresholds  = roc_curve(yVldtn, probabilities) #Get the ROC Curve
    roc_auc_score(yVldtn, probabilities)
    dataAnalyzeActions.calculate_scores(randomForestClassifier, 'Random Over Sampling', xVldtn, yVldtn)
    dataAnalyzeActions.precision_recall_threshold(yVldtn,falsePositiveRates, truePositiveRates, thresholds, probabilities)
    plt.figure(figsize=(8, 5))
    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(falsePositiveRates, truePositiveRates)
    plt.xlabel('False Positive Rate = 1 - Specificity Score')
    plt.ylabel('True Positive Rate  = Recall Score')
    plt.title('ROC Curve for Random Under Sampling')
    plt.show()





