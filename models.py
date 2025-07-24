# Import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

def run_model(model_name, lr_solver, lr_penalty):
    """Run different prediction models"""

    if model_name == 'logistic_regression':
        return LogisticRegression(penalty=lr_penalty, solver=lr_solver)
    elif model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100)
    elif model_name == 'gbc':
        return GradientBoostingClassifier()
    elif model_name == 'gaussian_nb':
        return GaussianNB()
    else:
        raise ValueError(f"Unknown model: {model_name}")