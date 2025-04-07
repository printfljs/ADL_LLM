import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from openai import OpenAI
from activity_recognition import (
    preprocess_data, segment_by_labeled_activity, save_results,
    run_pipeline
)
from data_load.load_data import load_ordoneza_dataset_nochange
from data_load.load_data import load_aruba_dataset
from utils.tools import load_dataframes_from_files
from evaluation.evaluator import evaluate_predictions_split_activities
import json

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main(config_file='config.json', api_key=OPENAI_API_KEY, model_name="gpt-4o-mini-2024-07-18", temperature=0):
    """
    Main function to run the activity recognition pipeline using configurations from a JSON file.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Read dataset configurations
    dataset_name = config['dataset']['dataset_name']
    dataset_part = config['dataset']['dataset_part']
    data_dir = config['dataset']['data_dir']

    # Read list configurations
    LOCATIONS = config['lists']['locations']
    DEVICES = config['lists']['devices']
    ACTIVITIES = config['lists']['activities']

    # Read prompt configurations
    DESCRIPTION_PROMPT = config['prompts']['description_prompt']
    CLASSIFICATION_SYSTEM_PROMPT_TEMPLATE = config['prompts']['classification_system_prompt_template']

    WINDOWS_DIR = f'{data_dir}/windows'
    TRUTH_LABELS_FILE = f'{data_dir}/truth_labels.txt'
    DESC_OUTPUT_PATH = f'{data_dir}/descriptions_{dataset_part}.txt'
    LABEL_OUTPUT_PATH = f'{data_dir}/predictions_{dataset_part}.txt'
    METRICS_OUTPUT_PATH = f'{data_dir}/metrics_{dataset_part}.txt'
    EVAL_FILE = f'{data_dir}/evaluation.txt'

    def get_classify_system_prompt(allow_multiple_activities=False):
        locations_str = ', '.join(f"'{loc}'" for loc in LOCATIONS)
        devices_str = ', '.join(DEVICES)
        activities_str = ','.join(ACTIVITIES)
        response_format = (
            "Your answer should be one or more of these activities using the following format: ACTIVITY=(activity name) or ACTIVITY=(activity name,activity name)"
            if allow_multiple_activities
            else "Your answer should be only one activity using the following format: ACTIVITY=(activity name)"
        )
        return CLASSIFICATION_SYSTEM_PROMPT_TEMPLATE.format(
            locations_str=locations_str,
            devices_str=devices_str,
            activities_str=activities_str,
            response_format=response_format
        )

    CLASSIFICATION_SYSTEM_PROMPT = get_classify_system_prompt(allow_multiple_activities=False)

    client = OpenAI(api_key=api_key)

    if dataset_name.lower() == 'ordoneza':
        dataset = load_ordoneza_dataset_nochange(Path("../dataset/UCI_ADL_Binary"), part=dataset_part)
    # elif dataset_name.lower() == 'aruba':
    #     dataset = load_aruba_dataset(Path("../dataset/casas/Aruba"))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    state_df = preprocess_data(dataset, dataset_part)
    windows, activities = segment_by_labeled_activity(state_df, dataset['activities'])
    save_results(windows, activities, WINDOWS_DIR, TRUTH_LABELS_FILE)

    loaded_dataframes = load_dataframes_from_files(WINDOWS_DIR)
    if loaded_dataframes:
        run_pipeline(
            windows=loaded_dataframes,
            desc_output_path=DESC_OUTPUT_PATH,
            label_output_path=LABEL_OUTPUT_PATH,
            metrics_output_path=METRICS_OUTPUT_PATH,
            description_prompt=DESCRIPTION_PROMPT,
            classification_prompt=CLASSIFICATION_SYSTEM_PROMPT,
            client=client,
            model_name=model_name,
            temperature=temperature
        )
        evaluate_predictions_split_activities(data_dir, dataset_part, TRUTH_LABELS_FILE, LABEL_OUTPUT_PATH, EVAL_FILE)

if __name__ == "__main__":
    main()