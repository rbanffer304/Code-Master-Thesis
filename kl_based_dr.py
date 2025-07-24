# Import libraries
import pandas as pd

# def run_kl_based_dr_level(dataset_name, X_train, X_test, chosen_level, risk_window_end=365):
#     """Apply level knowledge based dimension reduction"""
#
#     # Load ancestor and outcomes data
#     ancestry = pd.read_csv("relations_ancestor_total.csv")
#     # Load datasets
#     if dataset_name == 'kb_diabetes':
#         target = pd.read_csv('outcomes.csv')
#     elif dataset_name == 'kb_copd':
#         target = pd.read_csv('outcomes_copd.csv')
#
#     # Filter relations only above a certain chosen level
#     filtered_relations = ancestry[
#         (ancestry['ANCESTOR_CONCEPT_ID'] == 441840) &
#         (ancestry['MIN_LEVELS_OF_SEPARATION'] <= chosen_level )
#         ]
#     print(filtered_relations)
#
#     # # Merge filtered relations with X train
#     # covariates_train = X_train.merge(filtered_relations, left_on="conceptId", right_on="DESCENDANT_CONCEPT_ID")
#     # covariates_train = covariates_train[['conceptId', 'analysisId', 'covariateValue', 'patient', 'covariateId']].drop_duplicates()
#     # X_train = covariates_train
#     #
#     # # Merge filtered relations with X test
#     # covariates_test = X_test.merge(filtered_relations, left_on="conceptId", right_on="DESCENDANT_CONCEPT_ID")
#     # covariates_test = covariates_test[['conceptId', 'analysisId', 'covariateValue', 'patient', 'covariateId']].drop_duplicates()
#     # X_test = covariates_test
#
#     # Select only observed concepts
#     observed = X_train['conceptId']
#     # Select observed descendants
#     observed_descendants = ancestry[ancestry['DESCENDANT_CONCEPT_ID'].isin(observed)]
#     # Select concepts with observed descendants
#     concepts_with_observed_descendants = observed_descendants['ANCESTOR_CONCEPT_ID']
#     # Selected concepts by knowledge dr
#     selected_concepts_list = filtered_relations['DESCENDANT_CONCEPT_ID']
#     # Select only concepts that are selected with knowledge based dimension reduction technique
#     selected_concepts_df = X_train[X_train['conceptId'].isin(selected_concepts_list)]
#
#     patient = []
#     concept = []
#     covariateId = []
#     covariateValue = []
#     analysisId = []
#
#     for _, row in X_train.iterrows():
#         count = 0
#
#         for i in set(selected_concepts_list):
#             count += 1
#
#         for j in set(concepts_with_observed_descendants):
#             if j in set(selected_concepts_list):
#                 count += 1
#
#
#         if row['conceptId'] in set(selected_concepts_list) or row['conceptId'] in set(
#             concepts_with_observed_descendants):
#             patient.append(row['patient'])
#             concept.append(row['conceptId'])
#             covariateId.append(row['covariateId'])
#             analysisId.append(row['analysisId'])
#             covariateValue.append(count)
#         else:
#             patient.append(row['patient'])
#             concept.append(row['conceptId'])
#             covariateId.append(row['covariateId'])
#             analysisId.append(row['analysisId'])
#             covariateValue.append(0)
#
#         print(count)
#
#
#     df = pd.DataFrame()
#     df['conceptId'] = concept
#     df['analysisId'] = analysisId
#     df['covariateValue'] = covariateValue
#     df['patient'] = patient
#     df['covariateId'] = covariateId
#
#     X_train = df
#
#     observation = X_train.pivot(index=['patient'], columns=['covariateId'], values=['covariateValue'])
#     observation = observation['covariateValue']
#
#     # Replace NaN values with 0
#     for i in observation.columns:
#         observation[i] = observation[i].fillna(0)
#
#     print('Number of covariates :', len(observation.columns))
#     X_train = observation
#     print(X_test)
#
#     observation = X_test.pivot(index=['patient'], columns=['covariateId'], values=['covariateValue'])
#     observation = observation['covariateValue']
#
#     # Replace NaN values with 0
#     for i in observation.columns:
#         observation[i] = observation[i].fillna(0)
#
#     # print('Number of covariates :', len(observation.columns))
#     X_test = observation
#     print(X_test)
#
#     X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
#
#     # Create outcomes list
#     outcomes = []
#     has_outcome = 0
#     no_outcome = 0
#     for i in X_train.index:
#         if i in target['rowId'].values:
#             # Check if event occurs within the risk window
#             if target.loc[target['rowId'] == i, 'daysToEvent'].values[0] < risk_window_end:
#                 outcomes.append(1)
#                 has_outcome += 1
#             else:
#                 outcomes.append(0)
#                 no_outcome += 1
#         else:
#             outcomes.append(0)
#             no_outcome += 1
#
#     y_train = outcomes
#
#     outcomes = []
#     has_outcome = 0
#     no_outcome = 0
#     for i in X_test.index:
#         if i in target['rowId'].values:
#             # Check if event occurs within the risk window
#             if target.loc[target['rowId'] == i, 'daysToEvent'].values[0] < risk_window_end:
#                 outcomes.append(1)
#                 has_outcome += 1
#             else:
#                 outcomes.append(0)
#                 no_outcome += 1
#         else:
#             outcomes.append(0)
#             no_outcome += 1
#
#     y_test = outcomes
#
#
#     return X_train, X_test, y_train, y_test

def run_kl_based_dr_level(dataset_name, X_train, X_test, chosen_level, risk_window_end=365):
    # Load ancestor and outcomes data
    ancestry = pd.read_csv("relations_ancestor_total.csv")
    # Load datasets
    if dataset_name == 'kb_diabetes':
        target = pd.read_csv('outcomes.csv')
    elif dataset_name == 'kb_copd':
        target = pd.read_csv('outcomes_copd.csv')

    # Filter relations above chosen level
    filtered_relations = ancestry[
        (ancestry['ANCESTOR_CONCEPT_ID'] == 441840) &
        (ancestry['MIN_LEVELS_OF_SEPARATION'] < chosen_level)
    ]

    print(filtered_relations)


    selected_concepts_set = set(filtered_relations['DESCENDANT_CONCEPT_ID'])
    observed_concepts_set = set(X_train['conceptId'])
    observed_descendants = ancestry[ancestry['DESCENDANT_CONCEPT_ID'].isin(observed_concepts_set)]
    concepts_with_observed_descendants_set = set(observed_descendants['ANCESTOR_CONCEPT_ID'])

    # Count how many times each concept appears as an ancestor
    concept_to_score = observed_descendants['ANCESTOR_CONCEPT_ID'].value_counts().to_dict()

    # Function to compute cumulative score for each row
    def compute_cumulative_score(concept_id):
        score = 0
        if concept_id in selected_concepts_set:
            score += 1
            score += concept_to_score.get(concept_id, 0)
        return score

    # Apply to training and test sets
    X_train['covariateValue'] = X_train['conceptId'].apply(compute_cumulative_score)
    print(X_train)


    def pivot_covariates(df):
        observation = df.pivot(index='patient', columns='covariateId', values='covariateValue')
        return observation.fillna(0)

    X_train = pivot_covariates(X_train)
    X_test = pivot_covariates(X_test)

    print(X_train)
    print(X_test)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    def create_outcomes(indexes):
        outcomes = []
        for i in indexes:
            if i in target['rowId'].values:
                days_to_event = target.loc[target['rowId'] == i, 'daysToEvent'].values[0]
                outcomes.append(1 if days_to_event < risk_window_end else 0)
            else:
                outcomes.append(0)
        return outcomes

    y_train = create_outcomes(X_train.index)
    y_test = create_outcomes(X_test.index)

    return X_train, X_test, y_train, y_test



def run_kl_based_dr_weight(dataset_name, X_train, X_test, parent_weight, child_weight, risk_window_end=365):
    """Apply weight knowledge based dimension reduction"""

    # Load datasets
    if dataset_name == 'kb_diabetes':
        target = pd.read_csv('outcomes.csv')
    elif dataset_name == 'kb_copd':
        target = pd.read_csv('outcomes_copd.csv')

    # Load ancestor data
    ancestry = pd.read_csv("relations_ancestor_total.csv")
    parentage = ancestry[ancestry['MIN_LEVELS_OF_SEPARATION'] == 1]


    # Split covariateId to get conceptId and analysisId
    observed_covariates_train = X_train['covariateId']
    observed_concepts_train = []
    windows = []
    for i in observed_covariates_train:
        concept = i // 1000
        window = i % 1000
        observed_concepts_train.append(concept)
        windows.append(window)

    X_train['conceptId'] = observed_concepts_train
    X_train['analysisId'] = windows

    print(len(set(observed_concepts_train)))

    # Split covariateId to get conceptId and analysisId
    observed_covariates_test = X_test['covariateId']
    observed_concepts_test = []
    windows = []
    for i in observed_covariates_test:
        concept = i // 1000
        window = i % 1000
        observed_concepts_test.append(concept)
        windows.append(window)

    X_test['conceptId'] = observed_concepts_test
    X_test['analysisId'] = windows
    print(len(set(observed_concepts_test)))

    # Identify unique ancestor concepts related to observed concepts
    relevant_concept_train = ancestry[ancestry['DESCENDANT_CONCEPT_ID'].isin(observed_concepts_train)][
        'ANCESTOR_CONCEPT_ID'].unique()

    # Filter ancestry dataframe to include only relevant ancestor descendant relationships
    relevant_ancestry_train = ancestry[
        ancestry['ANCESTOR_CONCEPT_ID'].isin(relevant_concept_train) | ancestry['DESCENDANT_CONCEPT_ID'].isin(
            relevant_concept_train)]
    relevant_ancestry_train = relevant_ancestry_train.rename(
        columns={'ANCESTOR_CONCEPT_ID': 'ancestor', 'DESCENDANT_CONCEPT_ID': 'descendent',
                 'MIN_LEVELS_OF_SEPARATION': 'minimum separation', 'MAX_LEVELS_OF_SEPARATION': 'maximum separation'}) \
        .drop(columns=['Unnamed: 0', 'CTID'])

    # Filter parentage dataframe to include only relevant parent-child relationships
    relevant_parentage_train = parentage[
        parentage['ANCESTOR_CONCEPT_ID'].isin(relevant_concept_train) | parentage['DESCENDANT_CONCEPT_ID'].isin(
            relevant_concept_train)]
    relevant_parentage_train = relevant_parentage_train.rename(columns={'ANCESTOR_CONCEPT_ID': 'parent', 'DESCENDANT_CONCEPT_ID': 'child'})\
        .drop(columns=['Unnamed: 0', 'MIN_LEVELS_OF_SEPARATION', 'MAX_LEVELS_OF_SEPARATION', 'CTID'])

    children_train = set(relevant_parentage_train['child'])
    parents_train = set(relevant_parentage_train['parent'])

    # Identify leaf concepts
    leaf_concept_train = children_train - parents_train
    leaf_concepts_train = pd.DataFrame({"id": list(leaf_concept_train)})

    # Create windows to represent analysis ids
    analysisId_to_window = {101: "all", 102: '365d', 103: '180d', 104: '030d', 105: 'all',
                            106: '365d', 107: '180d', 108: '030d'}

    # Create observations dataframe
    observations = pd.DataFrame()
    observations['patient'] = X_train['patient'].astype(int)
    observations['window'] = X_train['analysisId'].apply(lambda x: analysisId_to_window.get(x, 'Unknown'))
    observations['conceptId'] = X_train['conceptId']
    observations['covariateId'] = X_train['covariateId']

    # Group by patient and count cumulative concept count
    observed_concepts_per_patient = observations.groupby(["patient", "window"]).agg(conceptcount=("conceptId", "nunique")).reset_index()

    # Count the number of unique patients per window
    train_patientcount = observations.groupby("window")['patient'].nunique().reset_index()
    train_patientcount = train_patientcount.rename(columns={'patient': 'count'})

    # Merge observations with the compute unique concept count per patient
    train_observation_ancestor = observations.merge(observed_concepts_per_patient, on=['patient', 'window'])
    train_observation_ancestor = train_observation_ancestor.merge(relevant_ancestry_train, left_on='conceptId', right_on='descendent')\
        .drop(columns=['descendent', 'minimum separation', 'maximum separation'])

    # Compute the weight of each concept by inverting the unique concept count
    train_observation_ancestor['weight'] = 1.0 / train_observation_ancestor['conceptcount']
    train_observation_ancestor = train_observation_ancestor.rename(columns={'conceptId': 'observed concept'})
    train_observation_ancestor = train_observation_ancestor.drop(columns=['conceptcount'])

    # Aggregate weighted concept values per patient
    train_per_patient_weighted_concept = train_observation_ancestor.groupby(['patient', 'window', 'ancestor']).agg(weight=('weight', 'sum')).reset_index()
    train_per_patient_weighted_concept = train_per_patient_weighted_concept.rename(columns={'ancestor': 'conceptId'})

    # Merge weighted concept values with patient count per window
    df_train_weighted_concept = train_per_patient_weighted_concept.merge(train_patientcount, on="window")

    # Group by window and concept and aggregate the weight
    train_weighted_concept = df_train_weighted_concept.groupby(["window", "conceptId"], as_index=False).agg(weight=("weight", "sum"))

    # Normalize weights by dividing by patient count
    train_weighted_concept["weight"] = train_weighted_concept["weight"] / df_train_weighted_concept.groupby(["window", "conceptId"])["count"].first().values

    # Rename columns for clarity in parent child relationships
    wcfp = train_weighted_concept.rename(columns={"conceptId": "parent", "weight": "parent weight"})
    wcfc = train_weighted_concept.rename(columns={"conceptId": "child", "weight": "child weight"})

    # Merge with parent child relationships
    df_wcfp = relevant_parentage_train.merge(wcfp, on="parent")
    df_wcfc = df_wcfp.merge(wcfc, on=["child", "window"])

    # Assign a weight of zero to leaf nodes
    leaf_df = leaf_concepts_train.merge(wcfp, left_on='id', right_on='parent').drop(columns=['parent'])
    leaf_df["child"] = None
    leaf_df["child weight"] = 0.0
    leaf_df = leaf_df.rename(columns={"id": "parent"})

    # Combine hierarchical relationships and leaf concepts
    train_weighted_parentage = pd.concat([df_wcfc, leaf_df], ignore_index=True)

    # Select concepts where parent weight is >= chosen parent weight and child weight < chosen child weight
    concept_selection = train_weighted_parentage[(train_weighted_parentage['parent weight'] >= parent_weight) & (child_weight > train_weighted_parentage['child weight'])]

    # Select only observed concepts
    observed = observations['conceptId']
    # Select observed descendants
    observed_descendants = relevant_ancestry_train[relevant_ancestry_train['descendent'].isin(observed)]
    # Select concepts with observed descendants
    concepts_with_observed_descendants = observed_descendants['ancestor']
    # Selected concepts by knowledge dr
    selected_concepts_list = concept_selection['child']
    # Select only concepts that are selected with knowledge based dimension reduction technique
    selected_concepts_df = X_train[X_train['conceptId'].isin(selected_concepts_list)]

    patient = []
    concept = []
    covariateId = []
    covariateValue = []
    analysisId = []

    for _, row in X_train.iterrows():

        if row['conceptId'] in set(selected_concepts_list) and row['conceptId'] in set(concepts_with_observed_descendants):
            patient.append(row['patient'])
            concept.append(row['conceptId'])
            covariateId.append(row['covariateId'])
            analysisId.append(row['analysisId'])
            covariateValue.append(1)
        else:
            patient.append(row['patient'])
            concept.append(row['conceptId'])
            covariateId.append(row['covariateId'])
            analysisId.append(row['analysisId'])
            covariateValue.append(0)
            # pass

    df = pd.DataFrame()
    df['conceptId'] = concept
    df['analysisId'] = analysisId
    df['covariateValue'] = covariateValue
    df['patient'] = patient
    df['covariateId'] = covariateId

    X_train = df
    # X_train.columns = X_train.columns.astype(str)



    # observation = X_train.pivot_table(index='patient', columns='conceptId', values='covariateValue', aggfunc='sum', fill_value=0)
    observation = X_train.pivot(index=['patient'], columns=['covariateId'], values=['covariateValue'])
    print(observation)
    observation = observation['covariateValue']
    print(observation)


    # Replace NaN values with 0
    for i in observation.columns:
        observation[i] = observation[i].fillna(0)

    print('Number of covariates :', len(observation.columns))
    X_train = observation
    print(X_train)

    # observation = X_test.pivot_table(index='patient', columns='conceptId', values='covariateValue', aggfunc='sum', fill_value=0)
    # observation = observation['covariateValue']
    observation = X_test.pivot(index=['patient'], columns=['covariateId'], values=['covariateValue'])
    observation = observation['covariateValue']

    # Replace NaN values with 0
    for i in observation.columns:
        observation[i] = observation[i].fillna(0)

    X_test = observation
    print(X_test)

    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Create outcomes list
    outcomes = []
    has_outcome = 0
    no_outcome = 0
    for i in X_train.index:
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

    y_train = outcomes

    # Create outcomes list
    outcomes = []
    has_outcome = 0
    no_outcome = 0
    for i in X_test.index:
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

    y_test = outcomes

    return X_train, X_test, y_train, y_test














