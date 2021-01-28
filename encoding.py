import os
# from pm4py.objects.log.adapters.pandas import csv_import_adapter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from collections import OrderedDict
import pandas as pd
import numpy as np
import sys
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
import pm4py.visualization.petrinet.visualizer as pn_vis_factory
from pm4py.visualization.common import gview
from pm4py.visualization.common import save as gsave

np.set_printoptions(threshold=sys.maxsize)


class Encoding():

    def __init__(self, path_dir):
        self.path_dir = path_dir

    def get_event_log(self):
        # get event log
        # log_csv = csv_import_adapter.import_dataframe_from_path(os.path.join(self.path_dir), sep=";")

        log_csv = pd.read_csv(self.path_dir, sep=';')
        log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)

        log_csv = log_csv.rename(
            columns={'case': 'case:concept:name', 'time': 'time:timestamp', 'event': 'concept:name'})
        log_csv['concept:name'] = log_csv['concept:name'].astype(dtype=str)

        param = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}

        event_log = log_converter.apply(log_csv, parameters=param)
        return event_log, log_csv

    def get_columns(self, event_log):
        net, initial_marking, final_marking = inductive_miner.apply(event_log)
        gviz = pn_vis_factory.apply(net, initial_marking, final_marking)
        gsave.save(gviz,"data/helpdesk.png")


        place = []
        transitions = []
        i = 0
        # Extract places
        for p in net.places:
            # place.append(p)
            if p.name == 'source' or p.name == "sink":
                col_p = p.name
            else:
                col_p = "p_" + str(i)
                i += 1
            place.append(col_p)

        # Extract transitions
        j = 0
        k = 0
        l = 0
        m = 0
        for t in net.transitions:
            col_t = str(t)
            if "skip" in col_t:
                col_t_final = "skip_" + str(j)
                j += 1
            elif "init_loop" in col_t:
                col_t_final = "init_loop_" + str(k)
                k += 1
            elif "tauJoin" in col_t:
                col_t_final = "tauJoin_" + str(l)
                l += 1
            elif "tauSplit" in col_t:
                col_t_final = "tauSplit_" + str(m)
                m += 1
            else:
                col_t_final = str(t)

            transitions.append(col_t_final)

        # Combine together to get model based attributes and length of places
        model_base_attribute = place + transitions

        place_count = len(place)
        print(model_base_attribute)

        return model_base_attribute, place_count
