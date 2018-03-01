import os
from utils.nai_epgm_utils import *


def run_with_yelp():
    input_dir = os.path.expanduser('../tests/resources/data/yelp/yelp.epgm')
    output_dir = os.path.expanduser('../tests/resources/data_splitter/yelp.epgm.out')
    dataset_name = 'small_yelp_example'
    attributes_to_ignore = ['yelpId', 'name']


    convert_from_EPGM(source_directory=input_dir,
                      output_directory=output_dir,
                      dataset_name=dataset_name,
                      target_attribute="elite",
                      node_type="user",
                      write_to_disk=True,
                      attributes_to_ignore=attributes_to_ignore)

    print("Done")


if __name__ == '__main__':
    run_with_yelp()

