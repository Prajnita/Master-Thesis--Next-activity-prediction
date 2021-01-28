from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from collections import OrderedDict
import pandas as pd
import numpy as np
import sys

# get event log
# log_csv = csv_import_adapter.import_dataframe_from_path(os.path.join(self.path_dir), sep=";")

log_csv = pd.read_csv("data\\bpi2019_converted_no_context_sample.csv", sep=';')
log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)

log_csv = log_csv.rename(
    columns={'case': 'case:concept:name', 'time': 'time:timestamp', 'event': 'concept:name'})
log_csv['concept:name'] = log_csv['concept:name'].astype(dtype=str)

param = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}

event_log = log_converter.apply(log_csv, parameters=param)

# get the path of activities and find the final columns
from pm4py.algo.filtering.log.paths import paths_filter

paths = paths_filter.get_paths_from_log(event_log, attribute_key="concept:name")
threshold = []
for key, value in paths.items():
    threshold.append(value)
max(threshold)
final_threshold = 1 * max(threshold) // 100
x = [i for i in threshold if i > final_threshold]
print(x)
final_columns = []
for i in x:
    for key, value in paths.items():
        if value == i:
            final_columns.append(key)
        else:
            pass
new_dict = dict(zip(final_columns, x))

path_columns = []
path_rows = []
for key, value in new_dict.items():
    # temp = [key, value]
    #final_columns.append(key)
    key = list(key.split(","))
    path_columns.append(key[1])

    path_rows.append(key[0])
path_columns = list(map(int, path_columns))
path_rows = list(map(int, path_rows))

j=0
for col in path_columns:
    i = 0
    for rows in path_rows:
        if col == rows:
            add_path = path_columns[i]
            final_columns[j]
            path_comb = final_columns[j] + "," + str(add_path)
            final_columns.append(path_comb)
            i = i + 1
        else:
            i = i + 1
    j = j+1
print(final_columns)
