# Import libraries
from sklearn.model_selection import train_test_split
import pandas as pd


def load_dataset(dataset_name, chosen_window, test_size=0.25, random_state=1, risk_window_end=365):
    """Loads and splits the dataset"""

    # Pick a dataset and outcome file
    if dataset_name == 'diabetes':
        data = pd.read_csv('cov_merged.csv')
        target = pd.read_csv('outcomes.csv')
    elif dataset_name == 'kb_diabetes':
        data = pd.read_csv('covariates.csv')
        target = pd.read_csv('outcomes.csv')
    elif dataset_name == 'copd':
        data = pd.read_csv('covariates_copd.csv')
        target = pd.read_csv('outcomes_copd.csv')
    elif dataset_name == 'kb_copd':
        data = pd.read_csv('covariates_copd.csv')
        target = pd.read_csv('outcomes_copd.csv')

    else:
        # if dataset does not exists raise value error
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Get the concept and analysis id from the covariate id
    data_concepts = []
    windows = []
    for i in data['covariateId']:
        concept = i // 1000
        window = i % 1000
        data_concepts.append(concept)
        windows.append(window)

    # Create windows to represent analysis ids
    analysisId_to_window = {101: "all", 102: '365d', 103: '180d', 104: '030d', 105: 'all',
                            106: '365d', 107: '180d', 108: '030d'}

    # Create observations data file from data file
    observation = pd.DataFrame()
    observation['conceptId'] = data_concepts
    observation['analysisId'] = windows
    observation['window'] = observation['analysisId'].apply(lambda x: analysisId_to_window.get(x, 'Unknown'))
    observation['covariateValue'] = 1
    observation['patient'] = data['rowId']

    # Pick time window
    # observation = observation[observation['window'] == chosen_window]
    observation = observation.drop_duplicates(subset=['conceptId', 'patient'])

    # Creating outcomes list
    outcomes = []
    has_outcome = 0
    no_outcome = 0
    for i in observation['patient']:
        if i in target['rowId'].values:
            if target.loc[target['rowId'] == i, 'daysToEvent'].values[0] < risk_window_end:
                outcomes.append(1)
                has_outcome += 1
            else:
                outcomes.append(0)
                no_outcome += 1
        else:
            outcomes.append(0)
            no_outcome += 1

    print(f'Has outcome {has_outcome}')
    print(f'No outcome {no_outcome}')

    if dataset_name != 'kb_diabetes' and dataset_name != 'kb_diabetes_10_years' and dataset_name != 'kb_copd':
        # From long to wide format
        observation = observation.pivot(index=['patient'], columns=['conceptId'], values=['covariateValue'])
        # observation = observation['covariateValue']

        # Replace NaN values with 0
        for i in observation.columns:
            observation[i] = observation[i].fillna(0)

        print('Number of covariates :', len(observation.columns))

        # Creating outcomes list
        outcomes = []
        has_outcome = 0
        no_outcome = 0
        for i in observation.index:
            if i in target['rowId'].values:
                if target.loc[target['rowId'] == i, 'daysToEvent'].values[0] < risk_window_end:
                    outcomes.append(1)
                    has_outcome += 1
                else:
                    outcomes.append(0)
                    no_outcome += 1
            else:
                outcomes.append(0)
                no_outcome += 1

        print(f'Has outcome {has_outcome}')
        print(f'No outcome {no_outcome}')
    print(observation)


    # Split the data in train and test set
    X_train, X_test, y_train, y_test = train_test_split(observation, outcomes, stratify=outcomes, random_state=random_state, test_size=test_size)

    return X_train, X_test, y_train, y_test, target