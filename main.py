import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,roc_curve,recall_score,roc_auc_score,accuracy_score,precision_score,classification_report,confusion_matrix

from DataAnalyzeActions import DataAnalyzeActions
from DataPrepActions import DataPrepActions

dataPrepActions = DataPrepActions();
dataAnalyzeActions = DataAnalyzeActions();

if __name__ == '__main__':

    X, y = dataPrepActions.fetchAndCleanDataframe()
    ros = RandomOverSampler(random_state=0)
    X_ros, y_ros = ros.fit_sample(X, y)

    rfc = RandomForestClassifier(random_state=0)
    X_train, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, stratify=y_train)
    rfc.fit(X_train_val, y_train_val)
    predictions=rfc.predict(X_val)
    print('Classification report:]\n', classification_report(y_val, predictions))
    #conf_mat = confusion_matrix(y)
    #print('Confusion matrix:\n', conf_mat)

    rf_ros = RandomForestClassifier()
    X_train_val_ros, y_train_val_ros = ros.fit_sample(X_train_val, y_train_val)
    rf_ros.fit(X_train_val_ros, y_train_val_ros)
    prediction = rf_ros.predict(X_val)
    print('Classifcation report:\n', classification_report(y_val, prediction))
    conf_mat = confusion_matrix(y_val, prediction)
    print('Confusion matrix:\n', conf_mat)
    prob = rf_ros.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds  = roc_curve(y_val, prob) #Get the ROC Curve
    roc_auc_score(y_val, prob)
    dataAnalyzeActions.scores_val(rf_ros, 'Random Over Sampling')
    dataAnalyzeActions.precision_recall_threshold(fpr, tpr, thresholds, t=0.2424)
    plt.figure(figsize=(8, 5))
    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate = 1 - Specificity Score')
    plt.ylabel('True Positive Rate  = Recall Score')
    plt.title('ROC Curve for Random Under Sampling')
    plt.show()





