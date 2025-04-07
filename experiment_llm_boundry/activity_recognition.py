import os
import pandas as pd
import numpy as np
import sys
import re
import csv
import ast
import time
import matplotlib.pyplot as plt
from datetime import timedelta
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Append parent directory to sys.path for module imports
sys.path.append("..")
load_dotenv()

# Constants and Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Column Names
ACTIVITY = 'activity'
DEVICE = 'device'
START_TIME = 'start_time'
END_TIME = 'end_time'
TIME = 'time'
VALUE = 'value'
NAME = 'name'

# Load Dataset
from data_load.load_data import load_ordoneza_dataset_nochange

# Data Processing Functions
from data_process.preprocess import convert_events_to_states
from utils.tools import load_dataframes_from_files
from evaluation.evaluator import evaluate_predictions_split_activities

def identify_activity_edges_from_raw_data(df, activity_edges_prompt, client, model_name, temperature, print_prompts=True):
    """
    根据传感器原始数据识别活动边缘（开始时间和结束时间）。
    """
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])

    data_str = df.to_string(index=False)
    user_prompt = f"Raw sensor data:\n{data_str}"

    # 输出提示
    if print_prompts:
        print("System Prompt for Activity Edges Identification:")
        print(activity_edges_prompt)
        print("\nUser Prompt:")
        print(user_prompt)
        print("\nResult:")

    try:
        response = client.chat.completions.create(
            model=model_name, 
            messages=[
                {"role": "system", "content": activity_edges_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=temperature
        )
        result = response.choices[0].message.content.strip()
        if print_prompts:
            print(result)
            print("\n" + "="*50 + "\n")
        return result
    except Exception as e:
        error_msg = f"Error identifying activity edges: {str(e)}"
        if print_prompts:
            print(error_msg)
            print("\n" + "="*50 + "\n")
        return error_msg

def run_pipeline_identify_activity_edges(df, window_size, activity_output_path, activity_edges_prompt, client, model_name, temperature):
    num_rows = len(df)
    
    with open(activity_output_path, "w", encoding="utf-8") as activity_file:
        for i in range(0, num_rows, window_size):
            window = df.iloc[i:i + window_size]
            
            if not window.empty:
                activity_edges = identify_activity_edges_from_raw_data(window, activity_edges_prompt, client, model_name, temperature)
                activity_file.write(f"{activity_edges}\n\n")
    
    print(f"Activity edges saved to: {activity_output_path}")

def preprocess_data(dataset, dataset_part, data_dir, window_size, activity_edges_prompt_template, edge_activities, client, model_name, temperature):
    state_df = convert_events_to_states(dataset['devices'], dataset_part)
    # 在这里填充 edge_activities_str
    activity_edges_prompt = activity_edges_prompt_template.format(edge_activities_str=','.join(edge_activities))
    run_pipeline_identify_activity_edges(dataset['devices'], window_size, f"{data_dir}/activity_edges.txt", activity_edges_prompt, client, model_name, temperature)
    return state_df

def read_csv_like_txt_to_df(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            data = [line.split(',') for line in lines]
        return pd.DataFrame(data, columns=['start_time', 'end_time', 'initial_label'])
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def merge_same_activities(df):
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    i = 0
    while i < len(df) - 1:
        if df.at[i, 'initial_label'] == df.at[i + 1, 'initial_label']:
            df.at[i, 'end_time'] = df.at[i + 1, 'end_time']
            df = df.drop(i + 1).reset_index(drop=True)
        else:
            i += 1
    return df

def align_data_with_events(df, activities_df, devices_df):
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    activities_df['start_time'] = pd.to_datetime(activities_df['start_time'])
    activities_df['end_time'] = pd.to_datetime(activities_df['end_time'])
    devices_df['start_time'] = pd.to_datetime(devices_df['start_time'])
    devices_df['end_time'] = pd.to_datetime(devices_df['end_time'])

    def align_row(row):
        start, end = row['start_time'], row['end_time']
        overlapping_devices = devices_df[
            ((devices_df['start_time'] >= start) & (devices_df['start_time'] <= end)) |
            ((devices_df['end_time'] >= start) & (devices_df['end_time'] <= end)) |
            ((devices_df['start_time'] <= start) & (devices_df['end_time'] >= end))
        ]
        overlapping_activities = activities_df[
            ((activities_df['start_time'] >= start) & (activities_df['start_time'] <= end)) |
            ((activities_df['end_time'] >= start) & (activities_df['end_time'] <= end)) |
            ((activities_df['start_time'] <= start) & (activities_df['end_time'] >= end))
        ]
        return pd.Series({
            'sensor_events': overlapping_devices.to_dict('records'),
            'activities': overlapping_activities['activity'].tolist()
        })

    aligned_data = df.apply(lambda row: align_row(row), axis=1)
    result_df = pd.concat([df, aligned_data], axis=1)
    return result_df[result_df['activities'].str.len() > 0]

def extract_label_to_txt(df, output_file_path):
    try:
        with open(output_file_path, 'w') as f:
            for label in df['initial_label'].tolist():
                f.write(str(label) + '\n')
        print(f"Labels saved to {output_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_sensor_data_to_windows(df):
    df.columns = ['start_time', 'end_time', 'label', 'sensor_events', 'activities']
    windows_list, activities_list = [], []
    
    for _, row in df.iterrows():
        events = ast.literal_eval(row['sensor_events']) if isinstance(row['sensor_events'], str) else row['sensor_events']
        activities = ast.literal_eval(row['activities']) if isinstance(row['activities'], str) else row['activities']
        activities_list.append(','.join(activities))
        
        window = [
            {
                'state': event['state'],
                'start_time': pd.to_datetime(event['start_time']),
                'end_time': pd.to_datetime(event['end_time']),
                'location': event['location'],
                'type': event['type'],
                'place': event['place'],
                'device': event['device'],
            } for event in events
        ]
        windows_list.append(window)
    
    return windows_list, activities_list

def save_results(windows_list, activities_list, windows_dir, truth_labels_file):
    os.makedirs(windows_dir, exist_ok=True)
    with open(truth_labels_file, 'w') as f:
        for activity in activities_list:
            f.write(f"{activity}\n")
    
    fieldnames = ['state', 'start_time', 'end_time', 'location', 'type', 'place', 'device']
    for i, window in enumerate(windows_list, 1):
        with open(os.path.join(windows_dir, f'window_{i}.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(window)

def generate_natural_language_description(df, description_prompt, client, model_name, temperature, print_prompts=True):
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    window_start = df['start_time'].min()
    window_time_str = window_start.strftime('%I:%M %p').lstrip('0')
    df = df.sort_values('start_time')
    
    events = [
        {
            'state': row['state'].split('(')[1].split(',')[0] if '(' in row['state'] else row['state'],
            'place': row['place'].lower(),
            'duration': max(int((row['end_time'] - row['start_time']).total_seconds()), 1),
            'start_time': row['start_time'].strftime('%I:%M %p').lstrip('0'),
            'end_time': row['end_time'].strftime('%I:%M %p').lstrip('0')
        } for _, row in df.iterrows()
    ]
    
    prompt = description_prompt.format(window_time_str=window_time_str, events=events)
    system_prompt = "You are a helpful intelligent assistant tasked with converting sensor data into smooth, natural English descriptions"

    if print_prompts:
        print("System Prompt for Description Generation:")
        print(system_prompt)
        print("\nUser Prompt:")
        print(prompt)
        print("\nResult:")

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=temperature
        )
        response_time = time.time() - start_time
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
        result = response.choices[0].message.content.strip()
        if print_prompts:
            print(result)
            print("\n" + "="*50 + "\n")
        return {
            'description': result,
            'response_time': response_time,
            'tokens_used': tokens_used
        }
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if print_prompts:
            print(error_msg)
            print("\n" + "="*50 + "\n")
        return {'description': error_msg, 'response_time': 0, 'tokens_used': 0}

def classify_single_activity(description, classification_prompt, client, model_name, temperature, print_prompts=True):
    if print_prompts:
        print("System Prompt for Activity Classification:")
        print(classification_prompt)
        print("\nUser Prompt:")
        print(description)
        print("\nResult:")

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": classification_prompt},
                {"role": "user", "content": description}
            ],
            temperature=temperature
        )
        response_time = time.time() - start_time
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
        activity_str = response.choices[0].message.content.strip()
        match = re.search(r"ACTIVITY=(.*)", activity_str)
        result = match.group(1).strip() if match else "Unknown"
        if print_prompts:
            print(result)
            print("\n" + "="*50 + "\n")
        return {
            'activity': result,
            'response_time': response_time,
            'tokens_used': tokens_used
        }
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if print_prompts:
            print(error_msg)
            print("\n" + "="*50 + "\n")
        return {'activity': error_msg, 'response_time': 0, 'tokens_used': 0}

def run_pipeline(windows, desc_output_path, label_output_path, metrics_output_path, description_prompt, classification_prompt, client, model_name, temperature):
    os.makedirs(os.path.dirname(desc_output_path), exist_ok=True)
    response_times, total_tokens, all_tokens = [], 0, []

    with open(desc_output_path, 'w', encoding='utf-8') as desc_file, \
         open(label_output_path, 'w', encoding='utf-8') as label_file, \
         open(metrics_output_path, 'w', encoding='utf-8') as metrics_file:
        
        for idx, window in enumerate(windows):
            desc_result = generate_natural_language_description(window, description_prompt, client, model_name, temperature)
            class_result = classify_single_activity(desc_result['description'], classification_prompt, client, model_name, temperature)
            
            response_times.extend([desc_result['response_time'], class_result['response_time']])
            total_tokens += desc_result['tokens_used'] + class_result['tokens_used']
            all_tokens.extend([desc_result['tokens_used'], class_result['tokens_used']])

            print(f"Window {idx + 1}:")
            print(f"Description Time: {desc_result['response_time']:.2f}s, Tokens: {desc_result['tokens_used']}")
            print(f"Classification Time: {class_result['response_time']:.2f}s, Tokens: {class_result['tokens_used']}\n")
            
            desc_file.write(f"{desc_result['description']}\n")
            label_file.write(f"{class_result['activity']}\n")
            metrics_file.write(f"Window {idx + 1}:\nDescription Time: {desc_result['response_time']:.2f}s, Tokens: {desc_result['tokens_used']}\nClassification Time: {class_result['response_time']:.2f}s, Tokens: {class_result['tokens_used']}\n\n")

        print(f"Total windows processed: {len(windows)}, Total tokens used: {total_tokens}")
        print(f"Descriptions saved to: {desc_output_path}, Labels saved to: {label_output_path}, Metrics saved to: {metrics_output_path}")

    plot_ecdf(response_times, os.path.dirname(desc_output_path))

def plot_ecdf(response_times, output_dir):
    sorted_times = np.sort(response_times)
    y = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_times, y, marker='.', linestyle='none')
    plt.xlabel('Response Time (seconds)')
    plt.ylabel('ECDF')
    plt.title('Empirical Cumulative Distribution Function of Response Times')
    plt.grid(True)
    plot_path = f'{output_dir}/response_time_ecdf.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.show()
    print(f"ECDF plot saved to: {plot_path}")