import pandas as pd
import numpy as np
import pandas as pd

def label_data_uci(activities: pd.DataFrame, events: pd.DataFrame, 
               time_col='time', activity_col='activity', other='other'):
    
    df_acts = activities.copy()
    df_devs = events.copy()

    df_acts['start_time'] = pd.to_datetime(df_acts['start_time'])
    df_acts['end_time'] = pd.to_datetime(df_acts['end_time'])
    df_devs[time_col] = pd.to_datetime(df_devs[time_col])

    eps = pd.Timedelta('1ns')
    mask_act_et_clsn = df_acts['end_time'].isin(df_devs[time_col])
    mask_act_st_clsn = df_acts['start_time'].isin(df_devs[time_col])
    df_acts.loc[df_acts[mask_act_et_clsn].index, 'end_time'] += eps
    df_acts.loc[df_acts[mask_act_st_clsn].index, 'start_time'] -= eps

    df_acts = df_acts.rename(columns={'start_time': time_col})

    # other
    df_other = df_acts[['end_time', activity_col]]
    df_other.loc[:, activity_col] = other
    df_other = df_other.rename(columns={'end_time': time_col})
    df_acts = df_acts.drop(columns='end_time')
    df_acts = pd.concat([df_acts, df_other.iloc[:-1]], ignore_index=True, axis=0) \
        .sort_values(by=time_col) \
        .reset_index(drop=True)

    df_acts['diff'] = df_acts[time_col].shift(-1) - df_acts[time_col]
    mask_invalid_others = (df_acts['diff'] < '1ns') & (df_acts[activity_col] == other)
    df_acts = df_acts[~mask_invalid_others][[time_col, activity_col]]

    df_acts = pd.concat([df_acts, pd.Series({
        time_col: activities.at[activities.index[-1], 'end_time'],
        activity_col: other
    }).to_frame().T]).reset_index(drop=True)

    df_label = df_acts.copy()
    df_label = df_label.sort_values(by=time_col).reset_index(drop=True)
    df_devs = df_devs.sort_values(by=time_col).reset_index(drop=True)

    if df_devs[time_col].iat[0] < df_label[time_col].iat[0]:
        df_label = pd.concat([df_label,
            pd.Series({
                time_col: df_devs.at[0, time_col] - pd.Timedelta('1ms'), 
                activity_col: other
            }).to_frame().T], axis=0, ignore_index=True)

    df_dev_tmp = df_devs.copy()
    df_act_tmp = df_label.copy().sort_values(by=time_col).reset_index(drop=True)

    df_dev_tmp[activity_col] = np.nan
    for col in set(df_devs.columns).difference(df_act_tmp.columns):
        df_act_tmp[col] = np.nan

    df = pd.concat([df_dev_tmp, df_act_tmp], ignore_index=True, axis=0)\
        .sort_values(by=time_col)\
        .reset_index(drop=True)

    df[activity_col] = df[activity_col].ffill()
    df = df.dropna().reset_index(drop=True)
    assert len(df) == len(df_devs)

    other_mask = (df[activity_col] == other)
    df.loc[other_mask, activity_col] = np.nan

    df = df.dropna(subset=[time_col, activity_col]).reset_index(drop=True)

    y= df[[time_col, activity_col]].copy()
    X = df.drop(columns=activity_col).copy()

    return X, y

def get_max_time_interval(df: pd.DataFrame, time_col: str = 'time'):

    df[time_col] = pd.to_datetime(df[time_col])
    time_diff = df[time_col].diff()
    max_time_diff = time_diff.max()

    return max_time_diff

def convert_events_to_states(event_df, dataset='A'):
    """
    将二值传感器事件流转换为传感器状态流，适配特定传感器列表，并包含位置信息。
    通过 dataset 参数控制使用 A 或 B 版本的状态映射。

    参数:
    event_df: DataFrame 包含 Start time, End time, Location, Type, Place 列
    dataset: str, 'A' 或 'B', 选择使用的状态映射版本

    返回:
    DataFrame 包含状态的开始时间、结束时间和状态描述
    """
    states = []

    # 确保时间列是 datetime 类型
    event_df['start_time'] = pd.to_datetime(event_df['start_time'])
    event_df['end_time'] = pd.to_datetime(event_df['end_time'])

    # 清除 location 和 place 列的空格并删除 NaN 行
    event_df['location'] = event_df['location'].str.strip()
    event_df['place'] = event_df['place'].str.strip()
    event_df.dropna(subset=['location', 'place'], inplace=True)

    # 定义传感器类型到状态的映射，根据版本选择不同的映射
    if dataset == 'A':
        state_mappings = {
            'PIR': {
                'Basin': {'Bathroom': 'NearBathroomBasin'},
                'Shower': {'Bathroom': 'NearBathroomShower'},
                'Cooktop': {'Kitchen': 'NearKitchenCooktop'},
            },
            'Magnetic': {
                'Fridge': {'Kitchen': 'FridgeDoorOpen'},
                'Cupboard': {'Kitchen': 'CupboardDoorOpen'},
                'Maindoor': {'Entrance': 'MainDoorOpen'},
                'Cabinet': {'Bathroom': 'CabinetDoorOpen'},
            },
            'Flush': {
                'Toilet': {'Toilet': 'ToiletFlushing'}
            },
            'Pressure': {
                'Seat': {'Living': 'SeatOccupied'},
                'Bed': {'Bedroom': 'BedOccupied'}
            },
            'Electric': {
                'Microwave': {'Kitchen': 'MicrowaveOn'},
                'Toaster': {'Kitchen': 'ToasterOn'},
            }
        }
    elif dataset == 'B':
        state_mappings = {
            'PIR': {
                'Door': {'Living': 'NearLivingArea', 'Kitchen': 'NearKitchenArea', 'Bedroom': 'NearBedroomArea', 'Bathroom': 'NearBathroomArea'},
                'Basin': {'Bathroom': 'NearBathroomBasin'},
                'Shower': {'Bathroom': 'NearBathroomShower'},
            },
            'Magnetic': {
                'Fridge': {'Kitchen': 'FridgeDoorOpen'},
                'Cupboard': {'Kitchen': 'CupboardDoorOpen'},
                'Maindoor': {'Entrance': 'MainDoorOpen'},
            },
            'Flush': {
                'Toilet': {'Toilet': 'ToiletFlushing'}
            },
            'Pressure': {
                'Seat': {'Living': 'SeatOccupied'},
                'Bed': {'Bedroom': 'BedOccupied'}
            },
            'Electric': {
                'Microwave': {'Kitchen': 'MicrowaveOn'}
            }
        }
    else:
        raise ValueError("dataset 参数必须是 'A' 或 'B'")

    # 对每一行进行处理
    for index, row in event_df.iterrows():
        # 获取传感器类型和位置
        sensor_type = row['type']
        location = row['location']
        place = row['place']

        # 根据传感器类型和位置生成状态名称
        if sensor_type in state_mappings and location in state_mappings[sensor_type] and place in state_mappings[sensor_type][location]:
            state_name = state_mappings[sensor_type][location][place]
        else:
            # 默认情况：使用 Location + Type
            state_name = f"{location}{sensor_type}"

        # 创建状态描述，格式为 st(state_name, start_time, end_time)
        state_desc = f"st({place}-{state_name}, {row['start_time'].strftime('%H:%M')}, {row['end_time'].strftime('%H:%M')})"

        # 添加到状态列表
        states.append({
            'state': state_desc,
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'location': row['location'],
            'type': row['type'],
            'place': row['place'],
            'device': row['device']
        })

    # 创建结果 DataFrame
    result_df = pd.DataFrame(states)

    # 按时间排序
    result_df = result_df.sort_values('start_time')

    return result_df