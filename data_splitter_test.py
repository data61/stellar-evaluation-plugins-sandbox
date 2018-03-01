import os
from utils.data_splitter import DataSplitter


def run_with_cora_epgm():
    input_dir = os.path.expanduser('../tests/resources/data/cora/cora.epgm')
    output_dir = os.path.expanduser('../tests/resources/data_splitter/cora.epgm.out')

    ds = DataSplitter()

    y = ds.load_data(input_dir, dataset_name='cora', target_attribute='subject', node_type='paper')

    y_train, y_val, y_test = ds.split_data(y, nc=20, test_size=100)

    ds.write_data(output_dir=output_dir, dataset_name='cora', y_train=y_train, y_test=y_test, y_val=y_val)

    print("Done")


def run_with_yelp_epgm():
    input_dir = os.path.expanduser('../tests/resources/data/yelp/yelp.epgm')
    output_dir = os.path.expanduser('../tests/resources/data_splitter/yelp.epgm.out')
    dataset_name = 'small_yelp_example'
    ds = DataSplitter()

    y = ds.load_data(input_dir, dataset_name=dataset_name, target_attribute='elite', node_type='user')

    y_train, y_val, y_test, y_unlabeled = ds.split_data(y, nc=20, test_size=100)

    ds.write_data(output_dir=output_dir, dataset_name=dataset_name,
                  y_train=y_train, y_test=y_test, y_val=y_val, y_unlabeled=y_unlabeled)

    print("Done")


def run_with_yelp_lab():
    input_dir = os.path.expanduser('../tests/resources/data_splitter/yelp.epgm.out/small_yelp_example.lab')
    output_dir = os.path.expanduser('../tests/resources/data_splitter/yelp.epgm.out')
    dataset_name = 'small_yelp_example'
    ds = DataSplitter()

    y = ds.load_data(input_dir, dataset_name=dataset_name, target_attribute='elite', node_type='user')

    y_train, y_val, y_test, y_unlabeled = ds.split_data(y, nc=20, test_size=100)

    ds.write_data(output_dir=output_dir, dataset_name=dataset_name,
                  y_train=y_train, y_test=y_test, y_val=y_val, y_unlabeled=y_unlabeled)

    print("Done")


if __name__ == '__main__':
    run_with_yelp_lab()

