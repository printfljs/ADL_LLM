import json
import os
from pathlib import Path
from openai import OpenAI
from activity_recognition import (
    preprocess_data, read_csv_like_txt_to_df, merge_same_activities, align_data_with_events,
    extract_label_to_txt, process_sensor_data_to_windows, save_results, run_pipeline,
    load_dataframes_from_files, evaluate_predictions_split_activities, load_ordoneza_dataset_nochange
)

def main(config_path='config.json'):
    # Load configuration from JSON file
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Dataset Configuration
    dataset_name = config['dataset']['dataset_name']
    dataset_part = config['dataset']['dataset_part']
    data_dir = config['dataset']['data_dir']
    window_size = 15

    # File Paths
    ACTIVITY_EDGES_FILE = f"{data_dir}/activity_edges.txt"
    ALIGNED_DATA_FILE = f"{data_dir}/aligned_data.csv"
    INITIAL_LABELS_FILE = f"{data_dir}/initial_labels.txt"
    WINDOWS_DIR = f"{data_dir}/windows"
    TRUTH_LABELS_FILE = f"{data_dir}/truth_labels.txt"
    INITIAL_EVAL_FILE = f"{data_dir}/initial_evaluation.txt"
    DESC_OUTPUT_PATH = f'{data_dir}/descriptions_{dataset_part}.txt'
    LABEL_OUTPUT_PATH = f'{data_dir}/predictions_{dataset_part}.txt'
    METRICS_OUTPUT_PATH = f'{data_dir}/metrics_{dataset_part}.txt'
    EVAL_FILE = f"{data_dir}/evaluation.txt"

    # Model Configuration
    API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=API_KEY)
    MODEL_NAME = "gpt-4o-mini-2024-07-18"
    TEMPERATURE = 0

    # Prompts and Lists
    DESCRIPTION_PROMPT = config['prompts']['description_prompt']
    locations = config['lists']['locations']
    devices = config['lists']['devices']
    activities = config['lists']['activities']
    edge_activities = config['lists']['edge_activities']
    ACTIVITY_EDGES_PROMPT_TEMPLATE = config['prompts']['activity_edges_prompt']  # 保留占位符的模板
    CLASSIFICATION_SYSTEM_PROMPT = config['prompts']['classification_system_prompt_template'].format(
        locations_str=', '.join(f"'{loc}'" for loc in locations),
        devices_str=', '.join(devices),
        activities_str=','.join(activities)
    )

    # Load Dataset
    if dataset_name.lower() == 'ordoneza':
        dataset = load_ordoneza_dataset_nochange(Path("../dataset/UCI_ADL_Binary"), part=dataset_part)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Pipeline Execution
    state_df = preprocess_data(
        dataset, 
        dataset_part, 
        data_dir, 
        window_size, 
        ACTIVITY_EDGES_PROMPT_TEMPLATE,  # 传递模板
        edge_activities,                 # 传递 edge_activities 列表
        client, 
        MODEL_NAME, 
        TEMPERATURE
    )
    df = read_csv_like_txt_to_df(ACTIVITY_EDGES_FILE)
    if df is None:
        return
    
    df = merge_same_activities(df)
    result_df = align_data_with_events(df, dataset['activities'], state_df)
    result_df.to_csv(ALIGNED_DATA_FILE, index=False)
    extract_label_to_txt(result_df, INITIAL_LABELS_FILE)
    
    windows_list, activities_list = process_sensor_data_to_windows(result_df)
    save_results(windows_list, activities_list, WINDOWS_DIR, TRUTH_LABELS_FILE)
    
    evaluate_predictions_split_activities(data_dir, dataset_part, TRUTH_LABELS_FILE, INITIAL_LABELS_FILE, INITIAL_EVAL_FILE)
    
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
            model_name=MODEL_NAME,
            temperature=TEMPERATURE
        )
        evaluate_predictions_split_activities(data_dir, dataset_part, TRUTH_LABELS_FILE, LABEL_OUTPUT_PATH, EVAL_FILE)

if __name__ == "__main__":
    main()