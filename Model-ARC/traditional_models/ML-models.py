from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_models():
    models = {
        "Naive Bayes": GaussianNB(var_smoothing=1e-8),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            max_features='log2',
            bootstrap=True,
            random_state=42
        ),
        "Support Vector Machine": SVC(
            kernel='rbf',
            C=10.0,
            gamma='auto',
            decision_function_shape='ovr',
            random_state=42
        )
    }
    return models
