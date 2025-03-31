import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Tuple

from data_process.describe import generate_window_description_hci_A, generate_window_description_hci_B, generate_window_description_hci_A_with_status

def segment_sensor_data_with_descriptions_hci(
    sensor_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    window_size: str = '60s',
    stride: str = '60s',
    time_col: str = 'time',
    sensor_cat_cols: list = ['location', 'type', 'place', 'device'],
    sensor_num_cols: list = ['value'],
    activity_col: str = 'activity',
    description_type: str = 'A' 
) -> Tuple[np.ndarray, np.ndarray, list, dict]:
    """
    返回值新增 descriptions 列表，并根据 description_type 调用不同的描述函数
    """
    # ----------------------------
    # 0. 保留原始数据副本
    # ----------------------------
    sensor_df_raw = sensor_df.copy()
    
    # ----------------------------
    # 1. 预处理编码（原始数据不修改）
    # ----------------------------
    sensor_df_encoded = sensor_df.copy()
    sensor_df_encoded[time_col] = pd.to_datetime(sensor_df_encoded[time_col])
    activity_df[time_col] = pd.to_datetime(activity_df[time_col])
    
    # 编码器初始化
    encoders = {}
    for col in sensor_cat_cols:
        le = LabelEncoder()
        sensor_df_encoded[col] = le.fit_transform(sensor_df_encoded[col])
        encoders[col] = le

    # 编码活动标签
    le_activity = LabelEncoder()
    activity_df[activity_col] = le_activity.fit_transform(activity_df[activity_col])
    encoders['activity'] = le_activity

    # 传感器类别列的映射
    for col, encoder in encoders.items():
        print(f"编码映射 - {col}:")
        for idx, label in enumerate(encoder.classes_):
            print(f"  {idx} -> {label}")

    
    # ----------------------------
    # 2. 窗口分割
    # ----------------------------
    window_starts = pd.date_range(
        start=sensor_df_encoded[time_col].iloc[0],
        end=sensor_df_encoded[time_col].iloc[-1] - pd.Timedelta(window_size),
        freq=stride
    )
    
    # 存储结果
    descriptions = []
    window_data = []
    labels = []
    
    # ----------------------------
    # 3. 处理每个窗口
    # ----------------------------
    for win_start in window_starts:
        win_end = win_start + pd.Timedelta(window_size)
        
        # 从编码数据提取事件（用于构建张量）
        mask_encoded = (sensor_df_encoded[time_col] >= win_start) & (sensor_df_encoded[time_col] < win_end)
        win_sensor_encoded = sensor_df_encoded.loc[mask_encoded].copy()
        
        # 从原始数据提取事件（用于生成描述）
        mask_raw = (sensor_df_raw[time_col] >= win_start) & (sensor_df_raw[time_col] < win_end)
        win_sensor_raw = sensor_df_raw.loc[mask_raw].copy()
        
        if win_sensor_encoded.empty:
            continue
        
        # 生成描述（使用原始数据）
        if description_type == 'A':
            desc = generate_window_description_hci_A(win_sensor_raw)
        elif description_type == 'B':
            desc = generate_window_description_hci_B(win_sensor_raw) # 调用 generate_window_description_hci_B
        else:
            raise ValueError(f"无效的 description_type: {description_type}. 必须是 'A' 或 'B'.")
        descriptions.append(desc)
        
        # 构建时序序列（使用编码数据）
        time_steps = int(pd.Timedelta(window_size).total_seconds())
        sensor_matrix = np.zeros((time_steps, len(sensor_cat_cols)+len(sensor_num_cols)))
        
        for _, row in win_sensor_encoded.iterrows():
            step = int((row[time_col] - win_start).total_seconds())
            if step < time_steps:
                sensor_matrix[step] = [row[col] for col in sensor_cat_cols + sensor_num_cols]
        
        window_data.append(sensor_matrix)
        
        # 处理标签
        mask_act = (activity_df[time_col] >= win_start) & (activity_df[time_col] < win_end)
        if mask_act.any():
            label = activity_df.loc[mask_act, activity_col].iloc[-1]
        else:
            label = -1  # 无标签标记

        labels.append(label)
    
    # ----------------------------
    # 4. 转换为张量
    # ----------------------------
    Xt = np.stack(window_data)
    yt = np.array(labels, dtype=np.int64)
    
    return Xt, yt, descriptions, encoders


def segment_by_time_with_status_hci(X: pd.DataFrame, y: pd.DataFrame = None, window_size: str = '60s', 
                                stride: str = None, time_col: str = 'time', activity_col: str = 'activity',
                                drop_empty_intervals: bool = True) -> tuple:
    # 输入校验
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X 必须是 pandas.DataFrame 类型")
    if y is not None and not isinstance(y, pd.DataFrame):
        raise ValueError("y 如果提供，必须是 pandas.DataFrame 类型")
    if time_col not in X.columns:
        raise ValueError(f"时间列 '{time_col}' 未在 X 中找到")
    if y is not None and (time_col not in y.columns or activity_col not in y.columns):
        raise ValueError(f"y 中必须包含时间列 '{time_col}' 和活动列 '{activity_col}'")

    # 转换为 Timedelta
    window_size = pd.Timedelta(window_size)
    if stride is None:
        stride = window_size  # 如果 stride 为 None，默认使用窗口大小作为步长
    else:
        if isinstance(stride, str) and '*' in stride:
            time_part, multiplier_part = stride.split('*')
            time_delta = pd.Timedelta(time_part)
            multiplier = float(multiplier_part)
            stride = time_delta * multiplier
        else:
            stride = pd.Timedelta(stride)

    # 复制并按时间排序
    df = X.copy().sort_values(by=time_col)
    st = X[time_col].iloc[0] - pd.Timedelta('1s')
    et = X[time_col].iloc[-1] + pd.Timedelta('1s')

    # 生成窗口起始时间
    st_windows = pd.date_range(st, et - window_size, freq=stride)

    # 分割数据
    X_list = []
    y_list = []

    if drop_empty_intervals:
        times = df[time_col].copy()
        win_st = st
        while win_st + window_size <= et:
            win = (win_st, win_st + window_size)
            event_idxs = df[(win[0] <= df[time_col]) & (df[time_col] < win[1])].index
            if not event_idxs.empty:
                X_segment = X.iloc[event_idxs].copy()
                X_list.append(X_segment)
                if y is not None:
                    y_list.append(y.iloc[event_idxs].copy())
                win_st = win_st + stride
            else:
                next_event_time = times[win_st < times].iloc[0]
                win_min_idx_not_containing_ev = (st_windows <= next_event_time - window_size).cumsum().max() - 1
                if win_min_idx_not_containing_ev == len(st_windows) - 1:
                    break
                win_st = st_windows[win_min_idx_not_containing_ev + 1]
    else:
        for win_st in st_windows:
            win = (win_st, win_st + window_size)
            event_idxs = df[(win[0] <= df[time_col]) & (df[time_col] < win[1])].index
            X_segment = X.iloc[event_idxs].copy()
            X_list.append(X_segment)
            if y is not None:
                y_list.append(y.iloc[event_idxs].copy())

    sensor_status_list = []
    for i, X_segment in enumerate(X_list):
        sensor_status = pd.Series(index=X_segment.index)
        prev_segment = X_list[i - 1] if i > 0 else None
        next_segment = X_list[i + 1] if i < len(X_list) - 1 else None

        for sensor in X_segment.columns.difference([time_col]):  # 遍历传感器列
            current_active = X_segment[sensor].any()
            prev_active = prev_segment[sensor].any() if prev_segment is not None else False
            next_active = next_segment[sensor].any() if next_segment is not None else False

            status = 'inactive' 
            if current_active:
                if prev_active and next_active:
                    status = 'keep_on'
                elif prev_active:
                    status = 'already_active'
                elif next_active:
                    status = 'persistent'
                else:
                    status = 'inner'

            sensor_status[X_segment.index] = status  # 为当前窗口的所有行设置状态

        X_segment['sensor_status'] = sensor_status  # 添加传感器状态列
        sensor_status_list.append(sensor_status)

    if y is not None:
        return X_list, y_list
    else:
        return X_list, None