from __future__ import division
import csv
import numpy
import copy
import utils as utils
from sklearn.model_selection import KFold, ShuffleSplit
from encoding import Encoding
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petrinet import visualizer as pn_vis_factory


class Preprocessor(object):
    data_structure = {
        'support': {
            'num_folds': 1,
            'data_dir': "data/data_converted.csv",
            'ascii_offset': 161,
            'data_format': "%d.%m.%Y-%H:%M:%S",
            'train_index_per_fold': [],
            'test_index_per_fold': [],
            'iteration_cross_validation': 0,
            'elements_per_fold': 0,
            'event_labels': [],
            'event_types': [],
            'map_event_label_to_event_id': [],
            'map_event_id_to_event_label': [],
            'map_event_type_to_event_id': [],
            'map_event_id_to_event_type': [],
            'end_process_instance': '!'
        },

        'meta': {
            'num_features': 0,
            'num_event_ids': 0,
            'max_length_process_instance': 0,
            'num_attributes_context': 0,
            'num_attributes_control_flow': 3,  # process instance id, event id and timestamp
            'num_process_instances': 0,
            'num_process_model_path_instance': 0
        },

        'data': {
            'process_instances': [],
            'ids_process_instances': [],
            'context_attributes_process_instances': [],
            'process_model_path_instance': [],

            'train': {
                'features_data': numpy.array([]),
                'labels': numpy.ndarray([])
            },

            'test': {
                'process_instances': [],
                'context_attributes': [],
                'process_model path_instance': [],
                'event_ids': []
            }
        }
    }

    def __init__(self, args):
        utils.llprint("Initialization ... \n")
        self.data_structure['support']['num_folds'] = args.num_folds
        self.data_structure['support']['data_dir'] = args.data_dir + args.data_set

        self.get_sequences_from_eventlog()

        self.data_structure['support']['elements_per_fold'] = \
            int(round(
                self.data_structure['meta']['num_process_instances'] / self.data_structure['support']['num_folds']))

        self.data_structure['data']['process_instances'] = list(
            map(lambda x: x + '!', self.data_structure['data']['process_instances']))
        self.data_structure['meta']['max_length_process_instance'] = max(
            map(lambda x: len(x), self.data_structure['data']['process_instances']))
        self.data_structure['support']['event_labels'] = list(
            map(lambda x: set(x), self.data_structure['data']['process_instances']))
        self.data_structure['support']['event_labels'] = list(
            set().union(*self.data_structure['support']['event_labels']))
        self.data_structure['support']['event_labels'].sort()
        self.data_structure['support']['event_types'] = copy.copy(self.data_structure['support']['event_labels'])

        self.data_structure['support']['map_event_label_to_event_id'] = dict(
            (c, i) for i, c in enumerate(self.data_structure['support']['event_labels']))
        self.data_structure['support']['map_event_id_to_event_label'] = dict(
            (i, c) for i, c in enumerate(self.data_structure['support']['event_labels']))
        self.data_structure['support']['map_event_type_to_event_id'] = dict(
            (c, i) for i, c in enumerate(self.data_structure['support']['event_types']))
        self.data_structure['support']['map_event_id_to_event_type'] = dict(
            (i, c) for i, c in enumerate(self.data_structure['support']['event_types']))

        self.data_structure['meta']['num_event_ids'] = len(self.data_structure['support']['event_labels'])

        self.data_structure['meta']['num_features'] = self.data_structure['meta']['num_event_ids'] + \
                                                      self.data_structure['meta']['num_attributes_context'] + \
                                                      self.data_structure['meta']['num_process_model_path_instance']

        if args.cross_validation:
            self.set_indices_k_fold_validation()
        else:
            self.set_indices_split_validation(args)

    def get_sequences_from_eventlog(self):

        id_latest_process_instance = ''
        process_instance = ''
        first_event_of_process_instance = True

        context_attributes_process_instance = []
        process_model_path_instance = []
        output = True

        eventlog_csvfile = open(self.data_structure['support']['data_dir'], 'r')
        eventlog_reader = csv.reader(eventlog_csvfile, delimiter=';', quotechar='|')
        final_columns, log_csv, eventlog, place_count = self.add_process_model_data()
        self.data_structure['meta']['num_process_model_path_instance'] = len(final_columns)

        next(eventlog_reader, None)
        row_count = 0

        for event in eventlog_reader:

            id_current_process_instance = event[0]

            if output:
                self.check_for_context_attributes(event)
                output = False

            if id_current_process_instance != id_latest_process_instance:
                self.add_data_to_data_structure(id_current_process_instance, 'ids_process_instances')
                id_latest_process_instance = id_current_process_instance
                count = 0

                if not first_event_of_process_instance:
                    self.add_data_to_data_structure(process_instance, 'process_instances')
                    self.add_data_to_data_structure(process_model_path_instance,
                                                    'process_model_path_instance')

                    if self.data_structure['meta']['num_attributes_context'] > 0:
                        self.add_data_to_data_structure(context_attributes_process_instance,
                                                        'context_attributes_process_instances')

                process_instance = ''
                process_model_path_instance = []

                if self.data_structure['meta']['num_attributes_context'] > 0:
                    context_attributes_process_instance = []

                self.data_structure['meta']['num_process_instances'] += 1

            if self.data_structure['meta']['num_attributes_context'] > 0:
                context_attributes_event = self.get_context_attributes_of_event(event)
                context_attributes_process_instance.append(context_attributes_event)

            process_instance = self.add_event_to_process_instance(event, process_instance)
            event_pm = eventlog[int(event[0])]
            process_model_path = self.add_process_model_path(row_count, event_pm, log_csv, final_columns, count,
                                                             place_count)
            process_model_path_instance.append(process_model_path)
            count += 1
            row_count += 1
            first_event_of_process_instance = False

        self.add_data_to_data_structure(process_instance, 'process_instances')
        self.add_data_to_data_structure(process_model_path_instance, 'process_model_path_instance')

        if self.data_structure['meta']['num_attributes_context'] > 0:
            self.add_data_to_data_structure(context_attributes_process_instance, 'context_attributes_process_instances')

        self.data_structure['meta']['num_process_instances'] += 1

    def add_process_model_path(self, row_count, event_pm, log_csv, final_columns, count, place_count):
        # Filter event log based on case number
        case_no = event_pm.attributes['concept:name']
        new_log_csv = log_csv.loc[log_csv['case:concept:name'] == case_no]
        param = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}
        # Get respective rows for pertinent extraction
        new_event_log = log_converter.apply(new_log_csv.loc[0:row_count], parameters=param)
        process_model_path = numpy.zeros((len(final_columns)), dtype=int)
        process_model_path = list(map(str, process_model_path))
        # Extract net using inductive miner package
        new_net, new_initial_marking, new_final_marking = inductive_miner.apply(new_event_log)
        # Map places
        if count == 0:
            for place in new_net.places:
                for c in range(len(final_columns)):
                    # if final_columns[c].name == str(place):
                    if str(place) == "sink":
                        break
                    elif final_columns[c] == str(place):
                        process_model_path[c] = '1'
                        # print("place column number is", c)
                        break
                    else:
                        pass
        elif count < (len(new_log_csv) - 2):
            for place in new_net.places:
                for c in range(len(final_columns)):
                    # if final_columns[c].name == str(place):
                    if str(place) == "sink":
                        break
                    elif final_columns[c] == str(place):
                        process_model_path[c] = '1'
                        # print("place column number is", c)
                        break
                    else:
                        pass
        elif (count + 1) == len(new_log_csv):
            for place in new_net.places:
                for c in range(len(final_columns)):
                    if final_columns[c] == str(place):
                        process_model_path[c] = '1'
                        break
        else:
            pass
        # Map Transitions
        for transition in new_net.transitions:
            for c in range(len(final_columns)):
                if c < place_count:
                    pass
                else:
                    if final_columns[c] == str(transition) or final_columns[c] == str(transition):
                        process_model_path[c] = '1'
                        break

        return process_model_path

    def set_training_set(self):
        utils.llprint("Get training instances ... \n")
        process_instances_train, context_attributes_train, process_model_path_instance_train, _ = \
            self.get_instances_of_fold('train')

        utils.llprint("Create cropped training instances ... \n")
        cropped_process_instances, cropped_context_attributes, cropped_process_model_path_instance, next_events = \
            self.get_cropped_instances(
                process_instances_train,
                context_attributes_train,
                process_model_path_instance_train)

        utils.llprint("Create training set data as tensor ... \n")
        features_data = self.get_data_tensor(cropped_process_instances,
                                             cropped_context_attributes,
                                             cropped_process_model_path_instance,
                                             'train')

        utils.llprint("Create training set label as tensor ... \n")
        labels = self.get_label_tensor(cropped_process_instances,
                                       next_events)

        self.data_structure['data']['train']['features_data'] = features_data
        self.data_structure['data']['train']['labels'] = labels

    def get_event_type(self, predictions):
        max_prediction = 0
        event_type = ''
        index = 0

        for prediction in predictions:
            if prediction >= max_prediction:
                max_prediction = prediction
                event_type = self.data_structure['support']['map_event_id_to_event_type'][index]
            index += 1

        return event_type

    def check_for_context_attributes(self, event):
        if len(event) == self.data_structure['meta']['num_attributes_control_flow']:
            utils.llprint("No context attributes found ...\n")
        else:
            self.data_structure['meta']['num_attributes_context'] = len(event) - self.data_structure['meta'][
                'num_attributes_control_flow']
            utils.llprint("%d context attribute(s) found ...\n" % self.data_structure['meta']['num_attributes_context'])

    def add_data_to_data_structure(self, values, structure):
        self.data_structure['data'][structure].append(values)

    def get_context_attributes_of_event(self, event):
        context_attributes_event = []
        for attribute_index in range(self.data_structure['meta']['num_attributes_control_flow'],
                                     self.data_structure['meta']['num_attributes_control_flow'] +
                                     self.data_structure['meta']['num_attributes_context']):
            context_attributes_event.append(event[attribute_index])

        return context_attributes_event

    def add_process_model_data(self):
        enc = Encoding(self.data_structure['support']['data_dir'])
        eventlog, log_csv = enc.get_event_log()
        final_columns, place_count = enc.get_columns(eventlog)

        return final_columns, log_csv, eventlog, place_count

    def add_event_to_process_instance(self, event, process_instance):
        return process_instance + chr(int(event[1]) + self.data_structure['support']['ascii_offset'])

    # Produces indicies for each fold of a k-fold cross-validation.

    def set_indices_k_fold_validation(self):

        kFold = KFold(n_splits=self.data_structure['support']['num_folds'], random_state=1, shuffle=True)

        for train_indices, test_indices in kFold.split(self.data_structure['data']['process_instances']):
            self.data_structure['support']['train_index_per_fold'].append(train_indices)
            self.data_structure['support']['test_index_per_fold'].append(test_indices)

    '''
    Produces indices for train and test set of a split-validation.
    '''

    def set_indices_split_validation(self, args):

        shuffle_split = ShuffleSplit(n_splits=1, test_size=args.split_rate_test, random_state=0)

        for train_indices, test_indices in shuffle_split.split(self.data_structure['data']['process_instances']):
            self.data_structure['support']['train_index_per_fold'].append(train_indices)
            self.data_structure['support']['test_index_per_fold'].append(test_indices)

    '''
    Retrieves instances of a fold.
    '''

    def get_instances_of_fold(self, mode):

        process_instances_of_fold = []
        context_attributes_of_fold = []
        path_instance_of_fold = []
        event_ids_of_fold = []

        for index, value in enumerate(self.data_structure['support'][mode + '_index_per_fold'][
                                          self.data_structure['support']['iteration_cross_validation']]):

            process_instances_of_fold.append(self.data_structure['data']['process_instances'][value])
            path_instance_of_fold.append(self.data_structure['data']['process_model_path_instance'][value])

            event_ids_of_fold.append(self.data_structure['data']['ids_process_instances'][value])

            if self.data_structure['meta']['num_attributes_context'] > 0:
                context_attributes_of_fold.append(
                    self.data_structure['data']['context_attributes_process_instances'][value])

        if mode == 'test':
            self.data_structure['data']['test']['process_instances'] = process_instances_of_fold
            self.data_structure['data']['test']['context_attributes'] = context_attributes_of_fold
            self.data_structure['data']['test']['process_model path_instance'] = path_instance_of_fold
            self.data_structure['data']['test']['event_ids'] = event_ids_of_fold
            return

        return process_instances_of_fold, context_attributes_of_fold, path_instance_of_fold, event_ids_of_fold

    '''
    Crops prefixes out of instances.
    '''

    def get_cropped_instances(self, process_instances, context_attributes_process_instances,
                              process_model_path_instance):

        cropped_process_instances = []
        cropped_context_attributes = []
        cropped_path_instance = []
        next_events = []

        if self.data_structure['meta']['num_attributes_context'] > 0:

            for process_instance, context_attributes_process_instance, process_model_path_instance in zip(
                    process_instances,
                    context_attributes_process_instances, process_model_path_instance):
                for i in range(0, len(process_instance)):
                    if i == 0:
                        continue

                    # 0:i -> get 0 up to n-1 events of a process instance, since n is the label
                    cropped_process_instances.append(process_instance[0:i])
                    cropped_context_attributes.append(context_attributes_process_instance[0:i])
                    cropped_path_instance.append(process_model_path_instance[0:i])

                    # label
                    next_events.append(process_instance[i])

        else:
            for process_instance, process_model_path_instance in zip(
                    process_instances, process_model_path_instance):
                for i in range(0, len(process_instance)):

                    if i == 0:
                        continue
                    cropped_process_instances.append(process_instance[0:i])
                    # cropped_path_instance.append(process_instance[0:i])
                    cropped_path_instance.append(process_model_path_instance[0:i])
                    # label
                    next_events.append(process_instance[i])

        return cropped_process_instances, cropped_context_attributes, cropped_path_instance, next_events

    '''
    Crops prefixes out of a single process instance.
    '''

    def get_cropped_instance(self, prefix_size, index, process_instance):
        cropped_process_instance = ''.join(process_instance[:prefix_size])
        cropped_process_model_path_instance = self.data_structure['data']['test']['process_model path_instance'][index][
                                              :prefix_size]
        if self.data_structure['meta']['num_attributes_context'] > 0:
            cropped_context_attributes = self.data_structure['data']['test']['context_attributes'][index][:prefix_size]

        else:
            cropped_context_attributes = []

        return cropped_process_instance, cropped_context_attributes, cropped_process_model_path_instance

    '''
    Produces a vector-oriented representation of data as 3-dimensional tensor.
    '''

    def get_data_tensor(self, cropped_process_instances, cropped_context_attributes_process_instance,
                        cropped_process_model_path_instance, mode):
        if mode == 'train':
            data_set = numpy.zeros((
                len(cropped_process_instances),
                self.data_structure['meta']['max_length_process_instance'],
                self.data_structure['meta']['num_features']), dtype=numpy.uint8)

        else:
            data_set = numpy.zeros((
                1,
                self.data_structure['meta']['max_length_process_instance'],
                self.data_structure['meta']['num_features']), dtype=numpy.float32)

        for index, cropped_process_instance in enumerate(cropped_process_instances):

            cropped_path_instance = cropped_process_model_path_instance[index]

            if self.data_structure['meta']['num_attributes_context'] > 0:
                cropped_context_attributes = cropped_context_attributes_process_instance[index]

            for t, event in enumerate(cropped_process_instance):
                for event_label in self.data_structure['support']['event_labels']:
                    if event_label == event:
                        data_set[
                            index, t, self.data_structure['support']['map_event_label_to_event_id'][event_label]] = 1.0

                for x in range(0, self.data_structure['meta']['num_process_model_path_instance']):
                    data_set[index, t, self.data_structure['meta']['num_event_ids'] + x] = cropped_path_instance[t][x]

                if self.data_structure['meta']['num_attributes_context'] > 0:
                    for y in range(0, self.data_structure['meta']['num_attributes_context']):
                        data_set[index, t, self.data_structure['meta']['num_event_ids'] +
                                 self.data_structure['meta']['num_process_model_path_instance'] + y] = \
                            cropped_context_attributes[t][y]

        return data_set

    def get_data_tensor_for_single_prediction(self, args, cropped_process_instance, cropped_context_attributes,
                                              cropped_process_model_path_instance):
        data_set = self.get_data_tensor(
            [cropped_process_instance],
            [cropped_context_attributes],
            [cropped_process_model_path_instance],
            'test')

        return data_set

    '''
    Produces a vector-oriented representation of label as 2-dimensional tensor.
    '''

    def get_label_tensor(self, cropped_process_instances, next_events):
        label = numpy.zeros((len(cropped_process_instances), len(self.data_structure['support']['event_types'])),
                            dtype=numpy.float64)

        for index, cropped_process_instance in enumerate(cropped_process_instances):

            for event_type in self.data_structure['support']['event_types']:

                if event_type == next_events[index]:
                    label[index, self.data_structure['support']['map_event_type_to_event_id'][event_type]] = 1
                else:
                    label[index, self.data_structure['support']['map_event_type_to_event_id'][event_type]] = 0

        return label
