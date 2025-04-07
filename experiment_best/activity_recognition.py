import pandas as pd
import numpy as np
import sys
import os
import re
import csv
import time
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

sys.path.append("..")
load_dotenv()

# Constants and Configuration
DATASET = 'A'
DATA_DIR = 'LLM_segment_A'

# Column Names
ACTIVITY = 'activity'
DEVICE = 'device'
START_TIME = 'start_time'
END_TIME = 'end_time'
TIME = 'time'
VALUE = 'value'
NAME = 'name'

# Load Dataset
from data_process.preprocess import convert_events_to_states

def preprocess_data(dataset, dataset_part):
    state_df = convert_events_to_states(dataset['devices'], dataset_part)
    return state_df

def segment_by_labeled_activity(sensor_df, activity_df):
    windows_list, activities_list = [], []
    required_columns = ['state', 'start_time', 'end_time', 'location', 'type', 'place', 'device']
    assert all(col in sensor_df.columns for col in required_columns), "Sensor DataFrame missing required columns"

    for _, activity_row in activity_df.iterrows():
        window = sensor_df[
            (sensor_df['start_time'] <= activity_row['end_time']) &
            (sensor_df['end_time'] >= activity_row['start_time'])
        ].copy()

        if not window.empty:
            window_events = window.apply(
                lambda row: {
                    'state': row['state'],
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'location': row['location'],
                    'type': row['type'],
                    'place': row['place'],
                    'device': row['device']
                },
                axis=1
            ).tolist()
            windows_list.append(window_events)
            activities_list.append(activity_row['activity'])

    return windows_list, activities_list

def save_results(windows, activities, windows_dir, truth_labels_file):
    os.makedirs(windows_dir, exist_ok=True)
    with open(truth_labels_file, 'w') as f:
        for activity in activities:
            f.write(f"{activity}\n")

    fieldnames = ['state', 'start_time', 'end_time', 'location', 'type', 'place', 'device']
    for i, window in enumerate(windows, 1):
        with open(os.path.join(windows_dir, f'window_{i}.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(window)

def generate_natural_language_description(df, description_prompt, client, model_name, temperature):
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    window_start = df['start_time'].min()
    window_time_str = window_start.strftime('%I:%M %p').lstrip('0')
    df = df.sort_values('start_time')

    events = [
        {
            'state': row['state'].split('(')[1].split(',')[0] if '(' in row['state'] else row['state'],
            'place': row['place'].lower(),
            'duration': max(int((row['end_time'] - row['start_time']).total_seconds()), 1)
        } for _, row in df.iterrows()
    ]

    prompt = description_prompt.format(window_time_str=window_time_str, events=events)
    print("--- Description Prompt ---")
    print(prompt)

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful intelligent assistant tasked with converting sensor data into smooth, natural English descriptions"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=temperature
        )
        response_time = time.time() - start_time
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
        description = response.choices[0].message.content.strip()
        print("--- Description Result ---")
        print(description)
        return {
            'description': description,
            'response_time': response_time,
            'tokens_used': tokens_used
        }
    except Exception as e:
        error_message = f"Error generating description: {str(e)}"
        print("--- Description Result ---")
        print(error_message)
        return {'description': error_message, 'response_time': 0, 'tokens_used': 0}

def classify_single_activity(description, classification_prompt, client, model_name, temperature):
    print("--- Classification System Prompt ---")
    print(classification_prompt)
    print("--- Classification User Prompt ---")
    print(description)
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
        activity_label = match.group(1).strip() if match else "Unknown"
        print("--- Classification Result ---")
        print(activity_label)
        return {
            'activity': activity_label,
            'response_time': response_time,
            'tokens_used': tokens_used
        }
    except Exception as e:
        error_message = f"Error classifying activity: {str(e)}"
        print("--- Classification Result ---")
        print(error_message)
        return {'activity': error_message, 'response_time': 0, 'tokens_used': 0}

def run_pipeline(windows, desc_output_path, label_output_path, metrics_output_path, description_prompt, classification_prompt, client, model_name, temperature):
    os.makedirs(os.path.dirname(desc_output_path), exist_ok=True)
    response_times, total_tokens, all_tokens = [], 0, []

    with open(desc_output_path, 'w', encoding='utf-8') as desc_file, \
            open(label_output_path, 'w', encoding='utf-8') as label_file, \
            open(metrics_output_path, 'w', encoding='utf-8') as metrics_file:

        for idx, window in enumerate(windows):
            print(f"\n--- Processing Window {idx + 1} ---")
            desc_result = generate_natural_language_description(window, description_prompt, client, model_name, temperature)
            description = desc_result['description']
            desc_time = desc_result['response_time']
            desc_tokens = desc_result['tokens_used']

            class_result = classify_single_activity(description, classification_prompt, client, model_name, temperature)
            activity_label = class_result['activity']
            class_time = class_result['response_time']
            class_tokens = class_result['tokens_used']

            response_times.extend([desc_time, class_time])
            total_tokens += desc_tokens + class_tokens
            all_tokens.extend([desc_tokens, class_tokens])

            desc_file.write(f"{description}\n")
            label_file.write(f"{activity_label}\n")
            metrics_file.write(f"Window {idx + 1}:\n")
            metrics_file.write(f"Description Time: {desc_time:.2f}s, Tokens: {desc_tokens}\n")
            metrics_file.write(f"Classification Time: {class_time:.2f}s, Tokens: {class_tokens}\n\n")

        print(f"\n--- Pipeline Summary ---")
        print(f"Total windows processed: {len(windows)}")
        print(f"Total tokens used: {total_tokens}")
        print(f"Descriptions saved to: {desc_output_path}")
        print(f"Activity labels saved to: {label_output_path}")
        print(f"Metrics saved to: {metrics_output_path}")

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