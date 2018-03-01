"""
Basic Node Attribute Inference (NAI) scheduler.

It implements a basic pipeline for performing inference for missing node labels by first splitting data to train, test,
and validation sets, performing dimensionality reduction and metric learning and finally inference. Each stage of the
pipeline has options that must be specified such as how many samples per class in the train set, the size of the test
set, the number of output dimensions for dimensionality reduction, the metric learning method and corresponding
parameters, and finally, the classification algorithm to use along with its related parameters.

"""
from utils.nai_epgm_utils import *
import argparse
import shutil
import copy
import os
from utils.nai_pipeline import NAIPipeline, PluginError


def display_results(results):
    for r in results:
        print(r)


def best_parameters(results):
    '''
    Finds and returns the best set of parameters, in terms of highest accuracy, for a series of experiments the
    results stored in the 'results' list of dictionaries given
    :param results: List of dictionaries with parameter and accuracy values
    :return: The dictionary in results with highest accuracy entry.
    '''
    best_result = None
    highest_accuracy = -1.0
    for r in results:
        if r['accuracy']['acc_test'] > highest_accuracy:
            highest_accuracy = r['accuracy']['acc_test']
            best_result = copy.deepcopy(r)

    return best_result


def parse_args():
    """
    Parses the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Metric Learning for node attribute classification in graph analysis.")

    parser.add_argument('--dataset-name', dest='dataset_name', nargs='?', default='cora',
                        help='Name of dataset.')

    parser.add_argument('--target-node-type', dest='node_type', nargs='?', default=None,
                        help='Type of nodes whose attributes are to be inferred.')

    parser.add_argument('--target-attribute', dest='target_attribute', nargs='?', default='',
                        help='Name of the attribute to infer.')

    parser.add_argument('--attributes-to-ignore', dest='attributes_to_ignore', nargs='*', default=[],
                        help='Names of attribute to ignore as predictors.')

    parser.add_argument('--input-dir', dest='input_dir', nargs='?',
                        default='/Users/eli024/Projects/data/cora/cora.epgm/',
                        help='Input directory where graph in EPGM format can be found.')

    parser.add_argument('--temp-dir', dest='temp_dir', nargs='?',
                        default='~/temp/',
                        help='Directory for storing temporary files')

    parser.add_argument('--output-dir', dest='output_dir',
                        nargs='?', default='pred/',
                        help='Directory to write graph with predicted node labels in EPGM format.')

    parser.add_argument('--pipeline', dest='pipeline_filename',
                        nargs='?', default='',
                        help='JSON formatted file specifying the NAI pipeline and corresponding plugin parameters.')

    parser.add_argument('--convert-epgm', dest='convert_epgm', default=False, action='store_true',
                        help='Extract edge list, .lab, and .att files from EPGM graph format input. Default is True.')

    return parser.parse_args()


def prepare_parameters_dict():
    parameters = {}

    # if wanting to tune parameters use code similar to below. For each parameter, give a list of values
    # to try
    #  parameters['representation'] = {"p": [0.5, 1.0, 2.0], "q": [0.5, 1.0, 2.0]}
    #  parameters['metric'] = {"method": ['lfda', 'lmnn'], "with_pca": [True], "pca_dim": [32, 16, 8]}
    #  parameters['inference'] = {"method": ['logistic', 'rforest']}
    #
    parameters['representation'] = {"p": [1.0], "q": [1.0]}
    parameters['metric'] = {"metric": ['lfda'], "with_pca": [True], "pca_dim": [8, 16], "dim": [8]}
    parameters['inference'] = {"method": ['logistic', 'rforest']}

    return parameters


if __name__ == '__main__':

    use_fixed_pipeline = False

    args = parse_args()

    input_epgm = os.path.expanduser(args.input_dir)
    dataset_name = args.dataset_name  # 'cora'
    tmp_directory = os.path.expanduser(args.temp_dir)

    # check if the tmp directory exists and if not, create it
    if not os.path.exists(tmp_directory):
        print("Creating temp directory {:s}".format(tmp_directory))
        os.mkdir(tmp_directory)
    else:
        # remove all files (if any) in the temp directory
        print("Deleting files in temp directory {:s}".format(tmp_directory))
        for fname in os.listdir(tmp_directory):
            full_path_fname = os.path.join(tmp_directory, fname)
            try:
                if os.path.isfile(full_path_fname):
                    os.unlink(full_path_fname)
            except Exception as e:
                print(e)

    #
    nai_pipeline = NAIPipeline()

    G_epgm = None
    if args.convert_epgm:
        # remember the epgm graph and use for output later
        G_epgm, v_map, iv_map, unique_vertex_labels, *_ = convert_from_EPGM(input_epgm, dataset_name, tmp_directory,
                                                                        node_type=args.node_type, target_attribute=args.target_attribute,
                                                                        attributes_to_ignore=args.attributes_to_ignore)
    else:
        # copy the EPGM files from input_epgm directory to tmp_directory
        the_files = os.listdir(args.input_dir)
        for epgm_file in the_files:
            if epgm_file.endswith('.json'):
                shutil.copy2(args.input_dir+epgm_file, tmp_directory)

    dataset_dir = tmp_directory

    if args.pipeline_filename == '':
        use_fixed_pipeline = True
        print("** Using fixed NAI pipeline **")
    else:
        ml_pipeline, parameters = nai_pipeline.load_pipeline_from_file(args.pipeline_filename)
        print("** Using NAI pipeline from {:s}".format(args.pipeline_filename))
        # set the target-attribute and node-type values in plugin parameters to the values send to the
        # scheduler via the command line.
        for plugin in ml_pipeline:
            if 'target_attribute' in plugin['parameters'].keys():
                plugin['parameters']['target_attribute'] = [args.target_attribute]
            if 'node_type' in plugin['parameters'].keys():
                plugin['parameters']['node_type'] = [args.node_type]
            if 'attributes_to_ignore' in plugin['parameters'].keys():
                plugin['parameters']['attributes_to_ignore'] = [args.attributes_to_ignore]

    try:
        if use_fixed_pipeline:
            parameters = prepare_parameters_dict()
            all_results = nai_pipeline.run_fixed_pipeline(dataset_dir=dataset_dir,
                                                          dataset_name=dataset_name, parameters=parameters)
        else:
            all_results = nai_pipeline.run_pipeline(dataset_dir=dataset_dir,
                                                    dataset_name=dataset_name, plugin_parameters=ml_pipeline)
    except PluginError as plugin_error:
        print("***********************")
        print("PluginError raised")
        print(plugin_error.result)
        print("***********************")

    plugin_names = [p['name'] for p in ml_pipeline]


    write_predictions_to_epgm = bool(set(plugin_names).intersection(set(["inference", "gcn"])))

    if "inference" in plugin_names:
        display_results(all_results)
        params = best_parameters(results=all_results)

        print("\n------------------------------------------\n")
        print("Best set of parameters: ", params)
        print("\n------------------------------------------\n")

        # Now the biggest hack of all
        # Write the predicted labels to the epgm vertices file.
        # Assume that only one file with extension *.pred in /temp/ directory, read it, and use the inverse vertex map
        # to update the vertices G_epgm before writing back to disk.
        write_to_epgm(input_epgm, tmp_directory, args.output_dir, G_epgm, iv_map, unique_vertex_labels, args.target_attribute)
    elif "gcn" in plugin_names:
        G_epgm, v_map, iv_map, unique_vertex_labels = convert_from_EPGM(input_epgm, dataset_name, tmp_directory,
                                                                        node_type=args.node_type,
                                                                        target_attribute=args.target_attribute,
                                                                        attributes_to_ignore=args.attributes_to_ignore,
                                                                        write_to_disk=False)

        write_to_epgm(input_epgm,
                      tmp_directory+'predictions/',
                      args.output_dir,
                      G_epgm,
                      None,
                      None,
                      target_attribute=args.target_attribute)

    # if write_predictions_to_epgm:
    #     if G_epgm is None:
    #         # this is necessary to make write_to_epgm work with results from the GCN plugin. There has to be a
    #         # better way to do this.
    #         G_epgm, v_map, iv_map, unique_vertex_labels = convert_from_EPGM(input_epgm, dataset_name, tmp_directory,
    #                                                                         node_type=args.node_type,
    #                                                                         target_attribute=args.target_attribute,
    #                                                                         attributes_to_ignore=args.attributes_to_ignore,
    #                                                                         write_to_disk=False)
    #     # write_to_epgm(input_epgm, tmp_directory, args.output_dir, G_epgm, iv_map, unique_vertex_labels, args.target_attribute)


    print("Scheduler Finished!")
