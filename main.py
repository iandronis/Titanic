# Titanic: Machine Learning from disaster
# author: John John
import csv
import matplotlib.pyplot as plt
import numpy as np

from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, precision_recall_curve
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def read_train_data(train_file_path):
    """
    Reads the data from the train file (according the 'Titanic: Machine Learning from disaster' train data file format).

    :param train_file_path: The path for the train data file.
    :return: The columns' names (list(str)), the train instances (data, list of instances-vectors) and
    the training labels (list of labels-integers).
    """
    train_data = []
    train_labels = []
    with open(train_file_path, 'rb') as f:
        tmp_reader = csv.reader(f)
        train_column_names = tmp_reader.next()
        for row in tmp_reader:
            train_labels.append(int(float(row[1])))
            tmp_instance = list()
            tmp_instance.append(row[0])
            tmp_instance += row[2:]
            train_data.append(tmp_instance)
    return train_column_names, train_data, train_labels


def read_test_data(test_file_path):
    """
    Reads the data from the test file (according the 'Titanic: Machine Learning from disaster' test data file format).

    :param test_file_path: The path for the test data file.
    :return: The columns' names (list(str)) and the test instances (data, list of instances-vectors).
    """
    test_data = []
    with open(test_file_path, 'rb') as f:
        tmp_reader = csv.reader(f)
        test_column_names = tmp_reader.next()
        for row in tmp_reader:
            test_data.append(row)
    return test_column_names, test_data


def feature_extraction(data):
    """
    Takes the data and extract features from it.

    :param data: A list of instances, where each instance is a vector (size of vector = instance's dimension).
    :return: A list of instances, where each instance is a vector of the extracted features.
    """
    final_data = list()

    name_titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.']
    sex_titles = ['male', 'female']
    age_limits = [16, 39, 55]
    fare_limits = [30, 94]
    cabin_titles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
    port_titles = ['C', 'Q', 'S']
    for tmp_instance in data:
        new_tmp_instance = list()
        # PassengerId --
        # tmp_instance[0]

        # Pclass --
        new_tmp_instance.append(int(tmp_instance[1].strip()))

        # Name --
        tmp_instance_name = tmp_instance[2].strip()

        name_title_feature = 0
        flag_title_found = False
        for nameTitle in name_titles:
            if nameTitle in tmp_instance_name:
                flag_title_found = True
                name_title_feature = name_titles.index(nameTitle)
                break
        if not flag_title_found:
            name_title_feature = len(name_titles)
        new_tmp_instance.append(name_title_feature)

        # Sex --
        tmp_instance_sex = tmp_instance[3].strip()
        sex_feature = sex_titles.index(tmp_instance_sex)
        new_tmp_instance.append(sex_feature)

        # Age --
        tmp_instance_age = tmp_instance[4].strip()
        if tmp_instance_age.strip() == '':
            age_feature = 4
        else:
            tmp_instance_age = float(tmp_instance_age)
            if tmp_instance_age <= age_limits[0]:
                age_feature = 0
            elif tmp_instance_age <= age_limits[1]:
                age_feature = 1
            elif tmp_instance_age <= age_limits[2]:
                age_feature = 2
            else:
                age_feature = 3
        new_tmp_instance.append(age_feature)

        # SibSp --
        new_tmp_instance.append(float(tmp_instance[5].strip()))

        # Parch --
        new_tmp_instance.append(float(tmp_instance[6].strip()))

        # Ticket --
        # tmp_instance[7]

        # Fare --
        tmp_instance_fare = tmp_instance[8].strip()
        if tmp_instance_fare.strip() == '':
            fare_feature = 3
        else:
            tmp_instance_fare = float(tmp_instance_fare)
            if tmp_instance_fare <= fare_limits[0]:
                fare_feature = 0
            elif tmp_instance_fare <= fare_limits[1]:
                fare_feature = 1
            else:
                fare_feature = 2
        new_tmp_instance.append(fare_feature)

        # Cabin --
        tmp_instance_cabin = tmp_instance[9].strip()
        if tmp_instance_cabin == '':
            cabin_feature = 8
        else:
            tmp_instance_cabin = tmp_instance_cabin[0]
            if tmp_instance_cabin in cabin_titles:
                cabin_feature = cabin_titles.index(tmp_instance_cabin)
            else:
                cabin_feature = 9
        new_tmp_instance.append(cabin_feature)

        # Embarked --
        tmp_instance_port = tmp_instance[10].strip()
        if tmp_instance_port in port_titles:
            port_feature = port_titles.index(tmp_instance_port)
        else:
            port_feature = 3
        new_tmp_instance.append(port_feature)

        final_data.append(new_tmp_instance)
    return final_data


def classify(classification_method, classification_method_params, train_data, train_labels, valid_data):
    """
    Train a model using a classification method and the training data and make predictions on the validation data.

    :param classification_method: The classification method to be used.
    :param classification_method_params: The classification method's parameters.
    :param train_data: The training data (list of instances-vectors).
    :param train_labels: The training labels (list of labels-integers).
    :param valid_data: The validation data (list of instances-vectors).
    :return: Returns the probabilities (score, array-like) and the predicted labels (array).
    """
    if classification_method == KNeighborsClassifier:
        classification_method = classification_method(n_neighbors=classification_method_params[0])
    elif classification_method == LogisticRegression:
        classification_method = classification_method(C=classification_method_params[0])
    else:
        classification_method = classification_method()
    classification_method.fit(train_data, train_labels)
    y_score = classification_method.predict_proba(valid_data)
    predicted_labels = classification_method.predict(valid_data)
    return y_score, predicted_labels


def evaluate(classifier_name, project_names, true_labels, predicted_score, predicted_labels):
    """
    Evaluate, print and plot the results.
    To be more specific, it computes accuracy score, precision-recall curve and the area under the (precision-recall)
    curve.

    :param classifier_name: The classifier's name (str), which has been used.
    :param project_names: A list of the classes' names (list(str)).
    :param true_labels: A list of the true labels (list(int)).
    :param predicted_score: A list of the predicted scores (list(float)).
    :param predicted_labels: A list of the predicted labels (list(int)).
    :return: None
    """
    print "Classifier: " + classifier_name
    print "Accuracy: " + str(accuracy_score(true_labels, predicted_labels))
    print classification_report(true_labels, predicted_labels, digits=4, target_names=project_names)
    # Compute Precision-Recall curve and area under it and plot it.
    # plt.figure()
    # plt.grid()
    true_labels = np.asarray(true_labels)
    tmp_precision, tmp_recall, _ = precision_recall_curve(true_labels, predicted_score[:, 1], pos_label=1)
    prrec_auc = auc(tmp_recall, tmp_precision)
    print "Area under the Precision-Recall curve ({0}): {1:0.3f}".format(classifier_name, prrec_auc)
    plt.plot(tmp_recall, tmp_precision,
             label="Area ({0})= {1:0.3f})".format(classifier_name, prrec_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision - Recall curve')
    plt.legend(loc="lower left")
    # plt.show()


def plot_learning_curve(classification_method, classification_method_params, data, labels):
    """
    Plot the learning curve using a classifier method (the 'learning_curve' method of sklearn library
    uses its own cross-validation).

    :param classification_method: The classification method.
    :param classification_method_params: The classification's parameters.
    :param data: The data (list of instances-vectors).
    :param labels: The labels (list of labels-integers).
    :return: None
    """
    if classification_method == KNeighborsClassifier:
        train_sizes, train_scores, valid_scores = learning_curve(
            classification_method(n_neighbors=classification_method_params[0]), data, labels,
            train_sizes=np.linspace(0.1, 1.0, 25))
    elif classification_method == LogisticRegression:
        train_sizes, train_scores, valid_scores = learning_curve(
            classification_method(C=classification_method_params[0]), data, labels,
            train_sizes=np.linspace(0.1, 1.0, 25))
    else:
        train_sizes, train_scores, valid_scores = learning_curve(
            classification_method(), data, labels, train_sizes=np.linspace(0.1, 1.0, 25))
    plt.figure()
    plt.title("Learning Curve ({0})".format(classification_method.__name__))
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    # Compute the mean and std to also plot them on the learning curve.
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()


def compare_classifiers(classifiers_list, classifiers_params_list, project_names, data, labels):
    """
    Train, classify and evaluate each classifier from a list.

    :param classifiers_list: A list of classifier methods.
    :param classifiers_params_list: A list of parameters for each classifier (list(list)).
    :param project_names: The classes' names.
    :param data: The data to be used for this purpose (list of instances-vectors).
    :param labels: The labels (list of labels-integers).
    :return: None
    """
    # Use cross-validation.
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.35, random_state=0)
    # Normalization.
    x_train_normalized = preprocessing.normalize(x_train, norm='l2')
    x_test_normalized = preprocessing.normalize(x_test, norm='l2')

    plt.figure()  # plot the results (precision-recall curve) on the same plot so it is easy to compare the classifiers.
    plt.grid()
    for j in range(len(classifiers_list)):  # for each classifier ...
        tmp_cls = classifiers_list[j]
        tmp_cls_params = classifiers_params_list[j]

        # Classify.
        tmp_yscore, tmp_predicted_labels = classify(tmp_cls, tmp_cls_params, x_train_normalized, y_train,
                                                    x_test_normalized)
        # Evaluate.
        evaluate(tmp_cls.__name__, project_names, y_test, tmp_yscore, tmp_predicted_labels)
    plt.show()

    # Plot learning curve for each classifier.
    for j in range(len(classifiers_list)):  # for each classifier ...
        plot_learning_curve(classifiers_list[j], classifiers_params_list[j], data, labels)


def final_classification(classification_method, classification_method_params, train_data, train_labels, test_data,
                         test_data_features, pred_labels_file_path):
    """
    Train and classify on test dataset and finally write the results (predicted labels) on csv file
    (according the 'Titanic: Machine Learning from disaster' test labels file format).

    :param classification_method: The final classification method.
    :param classification_method_params: The final classification's method parameters.
    :param train_data: The train data (list of instances-vectors).
    :param train_labels: The training labels (list of labels-integers).
    :param test_data: The test data (list of instances-vectors).
    :param test_data_features: The test data features (list of instances-vectors).
    :param pred_labels_file_path: The path for the predicted labels.
    :return: None
    """
    # Normalization.
    train_data_normalized = preprocessing.normalize(train_data, norm='l2')
    test_data_normalized = preprocessing.normalize(test_data_features, norm='l2')

    # Classify.
    _, predicted_labels = classify(classification_method, classification_method_params, train_data_normalized,
                                   train_labels, test_data_normalized)

    # Write the file for predicted labels on test dataset.
    with open(pred_labels_file_path, 'wb') as pred_labels_file:
        writer = csv.writer(pred_labels_file, delimiter=',')
        writer.writerow(["PassengerId", "Survived"])
        for i in range(len(test_data)):
            tmp_test_instance = test_data[i]
            tmp_passenger_id = int(tmp_test_instance[0])
            tmp_predicted_label = predicted_labels[i]
            writer.writerow([tmp_passenger_id, tmp_predicted_label.astype(int)])


# Declare the paths for train and test files' paths.
trainFilePath = "./data/train.csv"
testFilePath = "./data/test.csv"

# Read data.
# Read data from train dataset.
_, trainData, trainLabels = read_train_data(trainFilePath)

# Read data from test dataset.
_, testData = read_test_data(testFilePath)

# Extract features.
trainDataFeatures = feature_extraction(trainData)
testDataFeatures = feature_extraction(testData)

# # Compare a list of classifiers and find the best choice.
# classifiersList = [DummyClassifier, KNeighborsClassifier, BernoulliNB, MultinomialNB, GaussianNB, LogisticRegression]
# classifiersParamsList = [[], [5], [], [], [], [1e5]]
# projectNames = ["Died", "Survived"]
# compare_classifiers(classifiersList, classifiersParamsList, projectNames, trainDataFeatures, trainLabels)

# Final Classification.
classifierFinal = LogisticRegression  # Score = 78.47%
classifierParamsFinal = [1e5]
predLabelsFilePath = "./predictLabels.csv"
final_classification(classifierFinal, classifierParamsFinal, trainDataFeatures, trainLabels, testData,
                     testDataFeatures, predLabelsFilePath)
