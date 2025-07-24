
import pandas as pd
data = pd.read_csv('covariates.csv')

data_concepts = []
windows = []
for i in data['covariateId']:
    concept = i // 1000
    window = i % 1000
    data_concepts.append(concept)
    windows.append(window)

# Create observations data file from data file
observation = pd.DataFrame()
observation['conceptId'] = data_concepts
observation['analysisId'] = windows
observation['covariateValue'] = 1
observation['patient'] = data['rowId']
observation['covariateId'] = data['covariateId']

print(observation)

concept_counts = observation['conceptId'].value_counts()
print(concept_counts)

concept_counts = observation.groupby('patient')['conceptId'].value_counts()
print(concept_counts)
