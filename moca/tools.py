from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_train_model_list():
    model_list = ["Decision Tree Classifier", "Random Forest Classifier", "Gradient Boosting Classifier",
                  "Support Vector Classifier", "K-Nearest Neighbour Classifier", "Logistic Regression",
                  "Multi-Layer Perceptron Classifier"]
    return model_list


def get_fsel_model_list():
    model_list = ["Decision Tree Classifier", "Random Forest Classifier", "Gradient Boosting Classifier",
                  "Logistic Regression"]
    return model_list


def get_model_dict():
    model_dict = {
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42), "Random Forest Classifier": RandomForestClassifier(random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42), "Support Vector Classifier": SVC(random_state=42, probability=True),
        "K-Nearest Neighbour Classifier": KNeighborsClassifier(), "Logistic Regression": LogisticRegression(random_state=42), "Multi-Layer Perceptron Classifier": MLPClassifier(random_state=42)
    }
    return model_dict


def get_model_code_dict():
    code_dict = {
        "Decision Tree Classifier": "CART", "Random Forest Classifier": "RF", "Gradient Boosting Classifier": "GBDT",
        "Support Vector Classifier": "SVC", "K-Nearest Neighbour Classifier": "KNN", "Logistic Regression": "LOGREG",
        "Multi-Layer Perceptron Classifier": "MLPC"
    }
    return code_dict


def get_cv_dict():
    cv_dict = {"3-Fold Cross-Validation": KFold(n_splits=3, shuffle=True, random_state=42), "5-Fold Cross-Validation": KFold(n_splits=5, shuffle=True, random_state=42),
               "10-Fold Cross-Validation": KFold(n_splits=10, shuffle=True, random_state=42), "Leave-One-Out Cross-Validation": LeaveOneOut()
    }
    return cv_dict


def get_xls_file_format_list():
    xls_list = ["xls", "xlsx"]
    return xls_list


def get_txt_file_format_list():
    txt_list = ["csv", "txt"]
    return txt_list

