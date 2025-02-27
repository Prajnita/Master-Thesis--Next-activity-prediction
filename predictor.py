from __future__ import division
from tensorflow.keras.models import load_model
import csv
import distance
try:
    from itertools import izip as zip
except ImportError:
    pass
from jellyfish._jellyfish import damerau_levenshtein_distance
import utils as utils


def test(args, preprocessor):
    preprocessor.get_instances_of_fold('test')

    model = load_model(
        '%smodel_%s.h5' % (args.checkpoint_dir, preprocessor.data_structure['support']['iteration_cross_validation']))

    prediction_size = 1

    # set options for result output
    data_set_name = args.data_set.split('.csv')[0]
    result_dir_generic = args.result_dir + data_set_name + "__" + args.task
    result_dir_fold = result_dir_generic + "_%d%s" % (
        preprocessor.data_structure['support']['iteration_cross_validation'], ".csv")

    # start prediction
    with open(result_dir_fold, 'w') as result_file_fold:
        result_writer = csv.writer(result_file_fold, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(
            ["CaseID", "Prefix length", "Ground truth", "Predicted", "Levenshtein", "Damerau", "Jaccard"])

        # for each prefix_size
        for prefix_size in range(2, preprocessor.data_structure['meta']['max_length_process_instance']):
            utils.llprint("Prefix size: %d\n" % prefix_size)

            index = 0
            for process_instance, event_id in zip(preprocessor.data_structure['data']['test']['process_instances'],
                                                  preprocessor.data_structure['data']['test']['event_ids']):

                cropped_process_instance, cropped_context_attributes, cropped_process_model_path_instance = preprocessor.get_cropped_instance(
                    prefix_size,
                    index,
                    process_instance)
                index = index + 1

                # make no prediction for this case, since this case has already finished
                if preprocessor.data_structure['support']['end_process_instance'] in cropped_process_instance:
                    continue

                ground_truth = ''.join(process_instance[prefix_size:prefix_size + prediction_size])
                prediction = ''

                # predict only next activity (i = 1)
                for i in range(prediction_size):

                    if len(ground_truth) <= i:
                        continue

                    test_data = preprocessor.get_data_tensor_for_single_prediction(
                        args,
                        cropped_process_instance,
                        cropped_context_attributes,
                        cropped_process_model_path_instance)

                    y = model.predict(test_data)
                    y_char = y[0][:]

                    predicted_event = preprocessor.get_event_type(y_char)

                    cropped_process_instance += predicted_event
                    prediction += predicted_event

                    if predicted_event == preprocessor.data_structure['support']['end_process_instance']:
                        print('! predicted, end of process instance ... \n')
                        break

                output = []
                if len(ground_truth) > 0:
                    output.append(event_id)
                    output.append(prefix_size)
                    output.append(str(ground_truth).encode("utf-8"))
                    output.append(str(prediction).encode("utf-8"))
                    output.append(1 - distance.nlevenshtein(prediction, ground_truth))
                    dls = 1 - (damerau_levenshtein_distance(str(prediction), str(ground_truth)) / max(len(prediction),
                                                                                                      len(
                                                                                                          ground_truth)))
                    output.append(dls)
                    output.append(1 - distance.jaccard(prediction, ground_truth))

                    result_writer.writerow(output)
