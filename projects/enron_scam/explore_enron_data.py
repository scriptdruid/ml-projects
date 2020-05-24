""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(
    open(
        "/Users/vipul/Work/Github/ml-projects/projects/final_project/final_project/final_project_dataset.pkl",
        "rb",
    )
)

print(f" Total number of people = {len(enron_data.keys())}")
# Total number of people = 146

print(f"Total features available : {len(enron_data['METTS MARK'].values())}")
# Total features available : 21

fir

print(enron_data['METTS MARK']['poi'])