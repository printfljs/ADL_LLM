import pandas as pd
from datetime import timedelta
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import re
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

prompts_path = os.path.join(SCRIPT_DIR, 'prompts.json')
print(f"Loading prompts from: {prompts_path}")
with open(prompts_path, 'r', encoding='utf-8') as f:
    prompts = json.load(f)
print("Prompts loaded successfully")

# 修改后的 LLM 调用函数，添加 dataset 参数
def call_llm(client, model_name: str, dataset: str, prompt_key: str, user_prompt: str, max_tokens: int = 500):
    print(f"Calling LLM with model: {model_name}, dataset: {dataset}, prompt_key: {prompt_key}")
    try:
        start_time = time.time()
        print(f"System prompt: {prompts[dataset][prompt_key]['system_prompt']}")
        print(f"User prompt: {user_prompt}")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompts[dataset][prompt_key]['system_prompt']},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        end_time = time.time()
        response_time = end_time - start_time
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
        print(f"LLM call completed in {response_time:.2f}s")
        return {
            'response': response.choices[0].message.content.strip(),
            'response_time': response_time,
            'tokens_used': tokens_used
        }
    except Exception as e:
        print(f"LLM call failed: {str(e)}")
        return {
            'response': f"Error: {str(e)}",
            'response_time': 0,
            'tokens_used': 0
        }

# 修改后的生成自然语言描述函数
def generate_description_activity_based(client, model_name: str, df, dataset: str = "A"):
    print("Starting generate_natural_language_description")
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    window_start = df['start_time'].min()
    window_time_str = window_start.strftime('%I:%M %p').lstrip('0')
    
    df = df.sort_values('start_time')
    events = [
        {
            'state': row['state'].split('(')[1].split(',')[0] if '(' in row['state'] else row['state'],
            'place': row['place'].lower(),
            'location': row['location'].lower(),
            'type': row['type'].lower(),
            'device': row['device'].lower(),
            'duration': int((row['end_time'] - row['start_time']).total_seconds()) if (row['end_time'] - row['start_time']).total_seconds() > 0 else 1,
            'start_time': row['start_time'].strftime('%I:%M %p').lstrip('0'),
            'end_time': row['end_time'].strftime('%I:%M %p').lstrip('0')
        }
        for _, row in df.iterrows()
    ]
    
    user_prompt = prompts[dataset]['generate_description']['user_prompt'].format(
        window_time_str=window_time_str,
        events=events
    )
    
    llm_result = call_llm(
        client,
        model_name,
        dataset,
        'generate_description',
        user_prompt
    )
    print("Finished generate_natural_language_description")
    return {
        'description': llm_result['response'],
        'response_time': llm_result['response_time'],
        'tokens_used': llm_result['tokens_used']
    }

def generate_description_time_based(client, model_name: str, df, dataset: str = "A"):
    print("Starting generate_natural_language_description (time-based)")
    # TODO: 实现基于时间的自然语言描述逻辑
    print("Finished generate_natural_language_description (time-based)")
    return {
        'description': "Time-based description not implemented yet.",
        'response_time': 0,
        'tokens_used': 0
    }

# 修改后的活动分类函数
def classify_activity(client, model_name: str, description: str, multi: bool = False, dataset: str = "A"):
    print("Starting classify_activity")
    prompt_key = 'classify_activity_multi' if multi else 'classify_activity_single'
    llm_result = call_llm(
        client,
        model_name,
        dataset,
        prompt_key,
        description
    )
    activity_str = llm_result['response']
    activity_label = re.search(r"ACTIVITY=(.*)", activity_str)
    print("Finished classify_activity")
    return {
        'activity': activity_label.group(1).strip() if activity_label else "Unknown",
        'response_time': llm_result['response_time'],
        'tokens_used': llm_result['tokens_used']
    }

# 修改后的 identify_activity_edges_from_raw_data 函数
def identify_activity_edges_from_raw_data(client, model_name: str, df, batch_size: int = 15, 
                                        output_path: str = 'activity_edges.txt', dataset: str = "A"):
    print("Starting identify_activity_edges_from_raw_data")
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    all_results = []
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'a', encoding='utf-8') as activity_file:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_df)} rows)")
            data_str = batch_df.to_string(index=False)
            
            llm_result = call_llm(
                client,
                model_name,
                dataset,
                'identify_activity_edges',
                f"Raw sensor data (batch {batch_idx + 1}/{num_batches}):\n{data_str}",
                max_tokens=1000
            )
            
            batch_result = {
                'batch': batch_idx + 1,
                'response': llm_result['response'],
                'rows_processed': len(batch_df),
                'response_time': llm_result['response_time'],
                'tokens_used': llm_result['tokens_used']
            }
            all_results.append(batch_result)
            
            batch_text = (
                f"{batch_size * (batch_result['batch']-1) + batch_result['rows_processed']}):\n"
                f"{batch_result['response']}\n\n"
            )
            activity_file.write(batch_text)
            activity_file.flush()
            print(f"Batch {batch_idx + 1} results written to: {output_path}")
    
        combined_response = "\n\n".join(
            f"Batch {result['batch']} (rows {batch_size * (result['batch']-1) + 1}-"
            f"{batch_size * (result['batch']-1) + result['rows_processed']}):\n{result['response']}"
            for result in all_results
        )
    
    print(f"Finished identify_activity_edges_from_raw_data. Processed {total_rows} rows in {num_batches} batches")
    return {
        'combined_response': combined_response,
        'batch_results': all_results,
        'total_rows': total_rows,
        'total_batches': num_batches,
        'total_response_time': sum(result['response_time'] for result in all_results),
        'total_tokens': sum(result['tokens_used'] for result in all_results)
    }

# 修改后的 run_pipeline 函数
def run_pipeline(client, model_name: str, df_or_windows, desc_output_path: str = 'descriptions.txt',
                 label_output_path: str = 'predictions.txt', metrics_output_path: str = 'metrics.txt',
                 multi: bool = False, dataset: str = "A", description_type: str = "activity"):
    print("Starting run_pipeline")
    
    os.makedirs(os.path.dirname(desc_output_path), exist_ok=True)
    response_times = []
    total_tokens = [0]
    
    with open(desc_output_path, 'w', encoding='utf-8') as desc_file, \
         open(label_output_path, 'w', encoding='utf-8') as label_file, \
         open(metrics_output_path, 'w', encoding='utf-8') as metrics_file:
        
        print("Processing input data")
        if isinstance(df_or_windows, pd.DataFrame):
            df = df_or_windows
            for idx, row in df.iterrows():
                single_row_df = pd.DataFrame([row])
                print(f"Processing row {idx + 1}")
                process_window(client, model_name, single_row_df, idx + 1, 
                               desc_file, label_file, metrics_file, response_times, total_tokens, multi, dataset, description_type)
        else:
            for idx, window in enumerate(df_or_windows):
                if not window.empty:
                    single_row_df = pd.DataFrame([window.iloc[0]]) if len(window) > 0 else window
                    print(f"Processing window {idx + 1}")
                    process_window(client, model_name, single_row_df, idx + 1, 
                                   desc_file, label_file, metrics_file, response_times, total_tokens, multi, dataset, description_type)
        
        total_rows = len(df_or_windows) if isinstance(df_or_windows, pd.DataFrame) else len(df_or_windows)
        print(f"\nTotal rows processed: {total_rows}")
        print(f"Total tokens used: {total_tokens[0]}")
        print(f"Descriptions saved to: {desc_output_path}")
        print(f"Activity labels saved to: {label_output_path}")
        print(f"Metrics saved to: {metrics_output_path}")
    
    print("Generating ECDF plot")
    plot_ecdf(response_times, desc_output_path)
    print("Pipeline completed")

# 修改后的 process_window 函数
def process_window(client, model_name: str, window, window_num, desc_file, label_file, metrics_file, 
                    response_times, total_tokens, multi: bool, dataset: str, description_type: str):
    print(f"Processing window {window_num}: Starting")
    
    print(f"Window {window_num}: Generating description")
    if description_type == "activity":
        desc_result = generate_description_activity_based(client, model_name, window, dataset)
    elif description_type == "time":
        desc_result = generate_description_time_based(client, model_name, window, dataset)
    else:
        raise ValueError(f"Invalid description_type: {description_type}. Must be 'activity' or 'time'.")
    
    description = desc_result['description']
    
    print(f"Window {window_num}: Classifying activity")
    class_result = classify_activity(client, model_name, description, multi, dataset)
    activity_label = class_result['activity']
    
    response_times.extend([desc_result['response_time'], class_result['response_time']])
    total_tokens[0] += desc_result['tokens_used'] + class_result['tokens_used']
    
    print(f"Window {window_num}:")
    print(f"Description: {description}")
    print(f"Classified Activity: {activity_label}")
    print(f"Description Time: {desc_result['response_time']:.2f}s, Tokens: {desc_result['tokens_used']}")
    print(f"Classification Time: {class_result['response_time']:.2f}s, Tokens: {class_result['tokens_used']}\n")
    
    desc_file.write(f"{description}\n")
    label_file.write(f"{activity_label}\n")
    metrics_file.write(f"Window {window_num}:\n")
    metrics_file.write(f"Description Time: {desc_result['response_time']:.2f}s, Tokens: {desc_result['tokens_used']}\n")
    metrics_file.write(f"Classification Time: {class_result['response_time']:.2f}s, Tokens: {class_result['tokens_used']}\n\n")
    print(f"Window {window_num}: Processing completed")

def plot_ecdf(response_times, desc_output_path: str):
    print("Starting plot_ecdf")
    sorted_times = np.sort(response_times)
    y = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_times, y, marker='.', linestyle='none')
    plt.xlabel('Response Time (seconds)')
    plt.ylabel('ECDF')
    plt.title('Empirical Cumulative Distribution Function of Response Times')
    plt.grid(True)
    
    plot_path = os.path.join(os.path.dirname(desc_output_path), 'response_time_ecdf.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"ECDF plot saved to: {plot_path}")
    print("Finished plot_ecdf")

if __name__ == "__main__":
    from openai import OpenAI
    
    # 请替换为您的 OpenAI API 密钥
    client = OpenAI(api_key="YOUR_API_KEY")
    model_name = "gpt-3.5-turbo" # 请根据您的使用情况选择模型名称

    # 多活动分类
    # run_pipeline(
    #     client=client,
    #     model_name=model_name,
    #     df_or_windows=pd.DataFrame(), #请替换为您需要处理的数据
    #     desc_output_path='output/multi_descriptions.txt',
    #     label_output_path='output/multi_predictions.txt',
    #     metrics_output_path='output/multi_metrics.txt',
    #     multi=True,  # 多活动分类
    #     dataset = "A" #选择对应的数据集A或者B
    # )

    # # 单活动分类
    # run_pipeline(
    #     client=client,
    #     model_name=model_name,
    #     df_or_windows=pd.DataFrame(), #请替换为您需要处理的数据
    #     desc_output_path='output/single_descriptions.txt',
    #     label_output_path='output/single_predictions.txt',
    #     metrics_output_path='output/single_metrics.txt',
    #     multi=False, # 单活动分类
    #     dataset = "A" #选择对应的数据集A或者B
    # )