import os
import yaml
import pandas as pd
import argparse

#get data

def read_params(config_path):
    project_root = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(project_root, config_path)) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    data_path = config["data_source"]["s3_source"]
    project_root = os.path.dirname(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(project_root, data_path), sep=",", encoding='utf-8')
    return df

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--config",default="params.yaml")
    parsed_args = arg.parse_args()
    get_data(config_path=parsed_args.config)

