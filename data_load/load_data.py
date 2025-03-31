import pandas as pd
import numpy as np
from pathlib import Path
import json

# 全局常量
ACTIVITY = 'activity'
DEVICE = 'device'
START_TIME = 'start_time'
END_TIME = 'end_time'
TIME = 'time'
VALUE = 'value'
NAME = 'name'


def correct_activities(df_acts: pd.DataFrame, excepts: list = None, retain_corrections: bool = False):
    """
    修正活动数据中的时间重叠问题，优先保留指定的活动。

    Parameters
    ----------
    df_acts : pd.DataFrame
        活动数据，包含 START_TIME, END_TIME, ACTIVITY 列。
    excepts : list, optional
        不被覆盖的活动名称列表，默认为 None。
    retain_corrections : bool, optional
        是否返回修正记录，默认为 False。

    Returns
    -------
    df_acts : pd.DataFrame
        修正后的活动数据。
    corrections : list, optional
        如果 retain_corrections 为 True，返回修正记录。
    """
    df_acts = df_acts.copy().sort_values(START_TIME).reset_index(drop=True)
    excepts = excepts or []
    corrections = []

    for i in range(len(df_acts) - 1):
        current_row = df_acts.iloc[i]
        next_row = df_acts.iloc[i + 1]

        # 检查时间重叠
        if current_row[END_TIME] > next_row[START_TIME]:
            current_activity = current_row[ACTIVITY]
            next_activity = next_row[ACTIVITY]

            # 如果当前活动在 excepts 中，优先保留当前活动
            if current_activity in excepts:
                df_acts.at[i + 1, START_TIME] = current_row[END_TIME]
                corrections.append(f"Adjusted {next_activity} start from {next_row[START_TIME]} to {current_row[END_TIME]} due to overlap with {current_activity}")
            # 如果下一活动在 excepts 中，优先保留下一活动
            elif next_activity in excepts:
                df_acts.at[i, END_TIME] = next_row[START_TIME]
                corrections.append(f"Adjusted {current_activity} end from {current_row[END_TIME]} to {next_row[START_TIME]} due to overlap with {next_activity}")
            # 默认情况下，缩短当前活动
            else:
                df_acts.at[i, END_TIME] = next_row[START_TIME]
                corrections.append(f"Adjusted {current_activity} end from {current_row[END_TIME]} to {next_row[START_TIME]} due to overlap with {next_activity}")

    # 确保结束时间不早于开始时间
    invalid_mask = df_acts[END_TIME] < df_acts[START_TIME]
    df_acts = df_acts[~invalid_mask]
    if invalid_mask.any():
        corrections.append(f"Removed {invalid_mask.sum()} rows where end_time < start_time after correction")

    if retain_corrections:
        return df_acts, corrections
    return df_acts

def load_ordoneza_dataset(data_dir: str, part: str = 'A'):
    """
    加载 Ordonez 数据集（A 或 B 部分），映射活动名称并删除“如厕”活动。

    Parameters
    ----------
    data_dir : str
        数据集目录路径。
    part : str, optional
        数据集部分，'A' 或 'B'，默认为 'A'。

    Returns
    -------
    dict
        包含以下键值的字典：
        - 'activities': 活动数据 DataFrame
        - 'devices': 设备事件 DataFrame
        - 'device_areas': 设备区域映射 DataFrame
        - 'activity_list': 活动名称列表
        - 'device_list': 设备名称列表
    """
    data_dir = Path(data_dir)
    if part not in ['A', 'B']:
        raise ValueError("part 参数必须是 'A' 或 'B'")

    # 文件路径
    act_file = f'Ordonez{part}_ADLs.txt'
    sen_file = f'Ordonez{part}_Sensors.txt'
    corrected_act_file = data_dir / f'Ordonez{part}_ADLs_corr.txt' if part == 'B' else None

    # 对于 OrdonezB，修复活动文件格式
    if part == 'B':
        with open(data_dir / act_file, 'r') as f_in, open(corrected_act_file, 'w') as f_out:
            for i, line in enumerate(f_in):
                if i < 2:  # 保留标题行
                    f_out.write(line)
                    continue
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                new_line = f"{parts[0]} {parts[1]}\t\t{parts[2]} {parts[3]}\t\t{parts[4]}\n"
                f_out.write(new_line)
        act_path = corrected_act_file
    else:
        act_path = data_dir / act_file

    # 处理活动数据
    df_act = pd.read_csv(
        act_path,
        delimiter='\t+',
        skiprows=[0, 1],
        names=[START_TIME, END_TIME, ACTIVITY],
        engine='python'
    )
    df_act[START_TIME] = pd.to_datetime(df_act[START_TIME])
    df_act[END_TIME] = pd.to_datetime(df_act[END_TIME])

    # 修复 OrdonezA 中错误的活动时间
    if part == 'A':
        df_act.iat[69, 1] = pd.Timestamp('2011-12-01 19:29:59')
        df_act.iat[78, 1] = pd.Timestamp('2011-12-02 12:20:59')
        df_act.iat[80, 1] = pd.Timestamp('2011-12-02 12:35:49')
    # 对于 OrdonezB，应用 correct_activities
    elif part == 'B':
        df_act, corrections = correct_activities(df_act, excepts=['Grooming'], retain_corrections=True)
        print("Corrections applied to OrdonezB activities:")
        for corr in corrections:
            print(corr)

    # 映射活动名称
    activity_mapping = {
        'Spare_Time/TV': 'relaxing on couch',
        'Grooming': 'personal care',
        'Toileting': 'toileting',
        'Sleeping': 'sleeping',
        'Breakfast': 'preparing breakfast',
        'Showering': 'showering',
        'Snack': 'snacking',
        'Lunch': 'preparing lunch',
        'Leaving': 'leaving',
        'Dinner': 'preparing dinner'
    }
    df_act[ACTIVITY] = df_act[ACTIVITY].map(activity_mapping)

    # 删除“如厕”活动
    df_act = df_act[df_act[ACTIVITY] != 'toileting']

    # 处理设备数据
    df_sen = pd.read_csv(
        data_dir / sen_file,
        delimiter='\t+',
        skiprows=[0, 1],
        names=[START_TIME, END_TIME, 'location', 'type', 'place'],
        engine='python'
    )
    df_sen[DEVICE] = df_sen['place'] + '_' + df_sen['location'] + '_' + df_sen['type']
    df_sen[START_TIME] = pd.to_datetime(df_sen[START_TIME].str.strip())
    df_sen[END_TIME] = pd.to_datetime(df_sen[END_TIME].str.strip())

    device_areas = df_sen[['place', 'location', 'type', DEVICE]].drop_duplicates()
    df_start = df_sen.drop(columns=END_TIME)
    df_end = df_sen.drop(columns=START_TIME)
    df_start[VALUE] = True
    df_end[VALUE] = False
    df_start.rename(columns={START_TIME: TIME}, inplace=True)
    df_end.rename(columns={END_TIME: TIME}, inplace=True)

    df_events = pd.concat([df_end, df_start]).sort_values(TIME).reset_index(drop=True)

    duplicated_rows = df_events[df_events[TIME].duplicated(keep=False)]

    # 遍历索引，修改奇数索引的 time 列
    for i in range(1, len(duplicated_rows), 2):
        index = duplicated_rows.index[i]
        duplicated_rows.loc[index, TIME] += pd.Timedelta(seconds=1)

    df_events.update(duplicated_rows)

    return {
        'activities': df_act,
        'devices': df_events,
        'device_areas': device_areas,
        'activity_list': df_act[ACTIVITY].unique().tolist(),
        'device_list': df_events[DEVICE].unique().tolist()
    }

def _fix_line(s, i):
    """修复 Kyoto 数据集中的异常行"""
    if i == 2082109 or i == 2082361:  # 移除未正确开始或结束的 Work 活动
        s = s[:-2]
    return s

def _get_devices_df(df):
    """从 Kyoto 数据中提取设备数据"""
    df = df.copy().drop(ACTIVITY, axis=1)
    bin_mask = (df[VALUE] == 'ON') | (df[VALUE] == 'OFF')
    df_binary = df[bin_mask]
    df_binary[VALUE] = (df_binary[VALUE] == 'ON')
    num_mask = pd.to_numeric(df[VALUE], errors='coerce').notnull()
    df_num = df[num_mask]
    df_num[VALUE] = df_num[VALUE].astype(float)
    df_cat = df[~num_mask & ~bin_mask]
    df = pd.concat([df_cat, df_binary, df_num], axis=0, ignore_index=True)
    df.columns = [TIME, DEVICE, VALUE]
    return df.sort_values(by=TIME).reset_index(drop=True)

def _get_activity_df(df):
    """从 Kyoto 数据中提取活动数据"""
    df = df.copy()[~df[ACTIVITY].isnull()][[START_TIME, ACTIVITY]]
    df[ACTIVITY] = df[ACTIVITY].astype(str).apply(lambda x: x.strip())
    act_list = sorted(df[ACTIVITY].unique())
    
    new_df_lst = []
    for i in range(1, len(act_list), 2):
        activity = ' '.join(act_list[i].split(' ')[:-1])
        act_begin = act_list[i-1]
        act_end = act_list[i]
        if activity not in act_begin or activity not in act_end:
            continue
        df_res = df[df[ACTIVITY] == act_begin].reset_index(drop=True)
        df_end = df[df[ACTIVITY] == act_end].reset_index(drop=True)
        df_res[ACTIVITY] = activity
        df_res[END_TIME] = df_end[START_TIME]
        new_df_lst.append(df_res)
    
    res = pd.concat(new_df_lst)
    res = res.reindex(columns=[START_TIME, END_TIME, ACTIVITY])
    return res.sort_values(START_TIME).reset_index(drop=True)

class ActivityDict(dict):
    """活动数据的字典类，键为受试者名称，值为活动 DataFrame"""

    def __init__(self, obj=None):
        if isinstance(obj, pd.DataFrame):
            super().__init__({'subject': obj.copy().reset_index(drop=True)})
        elif isinstance(obj, list):
            if isinstance(obj[0], tuple):
                super().__init__({name: df for (name, df) in obj})
            else:
                super().__init__({f'subject_{i}': df for i, df in enumerate(obj)})
        elif isinstance(obj, dict):
            super().__init__(obj)
        else:
            super().__init__()

    def subjects(self) -> list:
        return list(self.keys())

    def to_json(self, date_unit="ns"):
        return json.dumps({k: df.to_json(date_unit=date_unit) for k, df in self.items()})

    @classmethod
    def read_json(cls, string):
        tmp = json.loads(string)
        return cls({k: pd.read_json(str) for k, str in tmp.items()})

    def nr_acts(self):
        return max([len(df[ACTIVITY].unique()) for df in self.values()])

    def get_activity_union(self):
        return list(set.union(*[set(df[ACTIVITY].unique()) for df in self.values()]))

    def apply(self, func):
        for k, df in self.items():
            self[k] = func(df)
        return self

    def min_starttime(self):
        return min([df[START_TIME].iloc[0] for df in self.values() if not df.empty])

    def max_endtime(self):
        return max([df[END_TIME].iloc[-1] for df in self.values() if not df.empty])

    def concat(self):
        return pd.concat(self.values())

    def copy(self):
        return ActivityDict({k: v.copy() for k, v in self.items()})

def load_kyoto_dataset(data_dir: str):
    """
    加载 Kyoto 2010 数据集。

    Parameters
    ----------
    data_dir : str
        数据集目录路径。

    Returns
    -------
    dict
        包含以下键值的字典：
        - 'activities': ActivityDict，包含两个居民的活动数据
        - 'devices': 设备事件 DataFrame
        - 'activity_list': 活动名称列表
        - 'device_list': 设备名称列表
    """
    data_dir = Path(data_dir)
    raw_path = data_dir / 'data'
    corrected_path = data_dir / 'corrected_data.csv'

    # 预处理原始数据
    with open(raw_path, 'r') as f_o, open(corrected_path, 'w') as f_t:
        delimiter = ';'
        for i, line in enumerate(f_o.readlines()):
            s = [sub.split(' ') for sub in line[:-1].split('\t')]
            s = [subsub for sub in s for subsub in sub if subsub]
            if not s:
                continue
            s = [' '.join([s[0], s[1]])] + s[2:]
            s = _fix_line(s, i)
            new_line = delimiter.join(s[:3])
            if len(s) > 3:
                new_line += delimiter + " ".join(s[3:])
            f_t.write(new_line + "\n")

    # 加载数据
    df = pd.read_csv(
        corrected_path,
        sep=';',
        names=[START_TIME, 'id', VALUE, ACTIVITY],
    )
    df[START_TIME] = pd.to_datetime(df[START_TIME], format='mixed')
    df = df.sort_values(by=START_TIME).drop_duplicates()
    df = df[~df.iloc[:, :3].isna().any(axis=1)].reset_index(drop=True)

    # 分离设备和活动数据
    df_dev = _get_devices_df(df)
    df_act = _get_activity_df(df)

    # 定义居民活动列表
    lst_act_res1 = [
        'R1_Wandering_in_room', 'R1_Sleep', 'R1_Bed_Toilet_Transition', 'R1_Personal_Hygiene',
        'R1_Bathing', 'R1_Work', 'R1_Meal_Preparation', 'R1_Leave_Home', 'R1_Enter_Home',
        'R1_Eating', 'R1_Watch_TV', 'R1_Housekeeping', 'R1_Sleeping_Not_in_Bed'
    ]
    lst_act_res2 = [
        'R2_Wandering_in_room', 'R2_Meal_Preparation', 'R2_Eating', 'R2_Work', 'R2_Bathing',
        'R2_Leave_Home', 'R2_Watch_TV', 'R2_Bed_Toilet_Transition', 'R2_Enter_Home',
        'R2_Sleep', 'R2_Personal_Hygiene', 'R2_Sleeping_Not_in_Bed'
    ]

    # 分割活动数据为两个居民
    dct_act = ActivityDict({
        'resident_1': df_act[df_act[ACTIVITY].isin(lst_act_res1)],
        'resident_2': df_act[df_act[ACTIVITY].isin(lst_act_res2)],
    })

    return {
        'activities': dct_act,
        'devices': df_dev,
        'activity_list': df_act[ACTIVITY].unique().tolist(),
        'device_list': df_dev[DEVICE].unique().tolist()
    }


def load_aruba_dataset(data_path: Path) -> dict:
    """
    处理CASAS数据集的主函数
    
    Args:
        data_path (Path): 数据目录路径
        
    Returns:
        dict: 包含活动和设备数据的字典
    """
    # 定义常量
    START_TIME = 'start_time'
    END_TIME = 'end_time'
    ACTIVITY = 'activity'
    DEVICE = 'device'
    VALUE = 'value'

    DEVICE_MAPPING = {
        'M003': 'M003_Bedroom1_Bed',
        'T002': 'T002_Living_Corner',
        'T003': 'T003_Kitchen_Counter',
        'T004': 'T004_Aisle_near_Bathroom2',
        'T005': 'T005_Office_Desk',
        'T001': 'T001_Bedroom1_Nightstand',
        'M002': 'M002_Bedroom1_Closet',
        'M007': 'M007_Bedroom1_Bedside',
        'M005': 'M005_Bedroom1_Doorway',
        'M004': 'M004_Bathroom1_Doorway',
        'M006': 'M006_Aisle_near_Bedroom1',
        'M008': 'M008_Aisle_near_Bedroom1',
        'M020': 'M020_Living_Center',
        'M010': 'M010_Living_Corner',
        'M011': 'M011_Aisle_near_Bedroom1',
        'M012': 'M012_Living_TV_Area',
        'M013': 'M013_Living_Table',
        'M014': 'M014_Dining_Table',
        'M009': 'M009_Living_Corner',
        'M018': 'M018_Kitchen_Counter',
        'M019': 'M019_Kitchen_Stove',
        'M015': 'M015_Kitchen_Door',
        'M016': 'M016_Backdoor_Entrance',
        'M017': 'M017_Backdoor_Hallway',
        'M021': 'M021_Aisle_near_Bathroom2',
        'M022': 'M022_Aisle_near_Bathroom2',
        'M023': 'M023_Bedroom2_Doorway',
        'M001': 'M001_Bedroom1_Corner',
        'M024': 'M024_Bedroom2_Bedside',
        'D002': 'D002_Backdoor',
        'M031': 'M031_Bathroom2_Doorway',
        'D004': 'D004_Garage_Door',
        'M030': 'M030_Aisle_near_Bathroom2',
        'M029': 'M029_Aisle_near_Bathroom2',
        'M028': 'M028_Office_Desk',
        'D001': 'D001_Frontdoor',
        'M026': 'M026_Office_Corner',
        'M027': 'M027_Office_Chair',
        'M025': 'M025_Office_Bookshelf'
    }

    ACTIVITY_MAPPING = {
        'sleeping': 'sleep',
        'bed_to_toilet': 'bed_to_toilet',
        'meal_preparation': 'cook',
        'relax': 'relax',
        'housekeeping': 'work',
        'eating': 'eat',
        'wash_dishes': 'work',
        'leave_home': 'leave_home',
        'enter_home': 'enter_home',
        'work': 'work',
        'respirate': 'other'
    }

    # 第一步：修正数据格式
    origin_path = data_path / 'data'
    corrected_path = data_path / 'corrected_data.csv'
    
    with open(origin_path, 'r') as f_o, open(corrected_path, 'w') as f_t:
        for line in f_o:
            s = line.strip().split()
            if len(s) < 4:
                continue
            
            if s[2] in ['ENTERHOME', 'LEAVEHOME']:
                continue
            if s[2] == 'c':
                s[2] = 'M014'
            
            device = s[2]
            if device in DEVICE_MAPPING:
                s[2] = DEVICE_MAPPING[device]
            
            value = s[3]
            if 'c' in value:
                value = value.replace('c', '')
            if s[2].startswith('M') and '5' in value:
                value = value.replace('5', '')
            if value in ['ONM026', 'ONM009', 'ONM024']:
                value = 'ON'
            if s[2].startswith('M') and len(value) == 1:
                value = 'ON'
            if s[2].startswith('M') and len(value) == 2 and s[1] == '18:13:47.291404':
                value = 'OFF'
            
            new_line = f"{s[0]} {s[1]},{s[2]},{value}"
            if len(s) >= 5:
                new_line += f",{' '.join(s[4:])}"
            f_t.write(new_line + '\n')

    # 第二步：加载和处理数据
    df = pd.read_csv(
        corrected_path,
        names=[START_TIME, DEVICE, VALUE, ACTIVITY],
        parse_dates=[START_TIME]
    ).sort_values(START_TIME).dropna(subset=[DEVICE, VALUE])

    # 处理设备数据
    df_dev = df.drop(ACTIVITY, axis=1).copy()
    binary_mask = (df_dev[VALUE] == 'ON') | (df_dev[VALUE] == 'OFF')
    df_binary = df_dev[binary_mask].copy()
    df_binary[VALUE] = df_binary[VALUE] == 'ON'
    
    numeric_mask = pd.to_numeric(df_dev[VALUE], errors='coerce').notnull()
    df_numeric = df_dev[numeric_mask].copy()
    df_numeric[VALUE] = df_numeric[VALUE].astype(float)

    temperature_sensors = ['T002_Living_Corner', 'T003_Kitchen_Counter', 'T004_Aisle_near_Bathroom2', 
                         'T005_Office_Desk', 'T001_Bedroom1_Nightstand']
    temperature_mask = df_numeric[DEVICE].isin(temperature_sensors)
    df_numeric.loc[temperature_mask, VALUE] = df_numeric.loc[temperature_mask, VALUE].where(
        df_numeric.loc[temperature_mask, VALUE] <= 200)
    df_numeric = df_numeric.dropna(subset=[VALUE])
    
    df_cat = df_dev[~binary_mask & ~numeric_mask].copy()
    df_dev_final = pd.concat([df_binary, df_numeric, df_cat])
    df_dev_final.columns = [START_TIME, DEVICE, VALUE]
    df_dev_final = df_dev_final.sort_values(START_TIME).reset_index(drop=True)

    # 处理活动数据
    df_act = df[~df[ACTIVITY].isna()][[START_TIME, ACTIVITY]].copy()
    df_act[ACTIVITY] = df_act[ACTIVITY].str.strip()

    result = []
    for activity_name in df_act[ACTIVITY].str.lower().str.replace(r' (begin|end)$', '', regex=True).unique():
        df_activity = df_act[df_act[ACTIVITY].str.lower().str.contains(activity_name.lower())].copy()
        
        df_begin = df_activity[df_activity[ACTIVITY].str.lower().str.endswith('begin')].copy()
        df_end = df_activity[df_activity[ACTIVITY].str.lower().str.endswith('end')].copy()

        if not df_begin.empty and not df_end.empty:
            df_begin = df_begin.sort_values(START_TIME).reset_index(drop=True)
            df_end = df_end.sort_values(START_TIME).reset_index(drop=True)
            
            min_len = min(len(df_begin), len(df_end))
            df_begin = df_begin.iloc[:min_len]
            df_end = df_end.iloc[:min_len]
            
            df_begin[ACTIVITY] = activity_name
            df_begin[END_TIME] = df_end[START_TIME].values
            result.append(df_begin)
        elif not df_begin.empty:
            df_begin[ACTIVITY] = activity_name
            df_begin[END_TIME] = None
            result.append(df_begin)
        elif not df_end.empty:
            df_end[ACTIVITY] = activity_name
            df_end[START_TIME] = None
            result.append(df_end)

    df_act_final=pd.concat(result)[[START_TIME, END_TIME, ACTIVITY]].sort_values(START_TIME).reset_index(drop=True)
    df_act_final[ACTIVITY] = df_act_final[ACTIVITY].map(ACTIVITY_MAPPING)

    df_dev_final = df_dev_final.rename(columns={'start_time': 'time'})

    # 保存结果
    if data_path:
        df_act_final.to_csv(data_path / 'activities.csv', index=False)
        df_dev_final.to_csv(data_path / 'devices.csv', index=False)

    # 返回结果
    return {
        'activities': df_act_final,
        'devices': df_dev_final,
        'activity_list': df_act_final[ACTIVITY].unique().tolist(),
        'device_list': df_dev_final[DEVICE].unique().tolist()
    }

if __name__ == "__main__":
    # 测试 OrdonezA 数据集
    dataset_a = load_ordoneza_dataset(Path("../dataset/UCI_ADL_Binary"), part='A')
    print("OrdonezA Activities:")
    print(dataset_a['activities'].head())
    print("\nOrdonezA Device Events:")
    print(dataset_a['devices'].head())

    # 测试 OrdonezB 数据集（带 correct_activities）
    dataset_b = load_ordoneza_dataset(Path("../dataset/UCI_ADL_Binary"), part='B')
    print("\nOrdonezB Activities:")
    print(dataset_b['activities'].head())
    print("\nOrdonezB Device Events:")
    print(dataset_b['devices'].head())

    # 测试 Kyoto 数据集
    dataset_kyoto = load_kyoto_dataset(Path("../dataset/casas/kyoto2010"))
    print("\nKyoto Activities (Resident 1):")
    print(dataset_kyoto['activities']['resident_1'].head())
    print("\nKyoto Activities (Resident 2):")
    print(dataset_kyoto['activities']['resident_2'].head())
    print("\nKyoto Devices:")
    print(dataset_kyoto['devices'].head())

def load_ordoneza_dataset_nochange(data_dir: str, part: str = 'A'):

    data_dir = Path(data_dir)
    if part not in ['A', 'B']:
        raise ValueError("part 参数必须是 'A' 或 'B'")

    # 文件路径
    act_file = f'Ordonez{part}_ADLs.txt'
    sen_file = f'Ordonez{part}_Sensors.txt'
    corrected_act_file = data_dir / f'Ordonez{part}_ADLs_corr.txt' if part == 'B' else None

    # 对于 OrdonezB，修复活动文件格式
    if part == 'B':
        with open(data_dir / act_file, 'r') as f_in, open(corrected_act_file, 'w') as f_out:
            for i, line in enumerate(f_in):
                if i < 2:  # 保留标题行
                    f_out.write(line)
                    continue
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                new_line = f"{parts[0]} {parts[1]}\t\t{parts[2]} {parts[3]}\t\t{parts[4]}\n"
                f_out.write(new_line)
        act_path = corrected_act_file
    else:
        act_path = data_dir / act_file

    # 处理活动数据
    df_act = pd.read_csv(
        act_path,
        delimiter='\t+',
        skiprows=[0, 1],
        names=[START_TIME, END_TIME, ACTIVITY],
        engine='python'
    )
    df_act[START_TIME] = pd.to_datetime(df_act[START_TIME])
    df_act[END_TIME] = pd.to_datetime(df_act[END_TIME])

    # 修复 OrdonezA 中错误的活动时间
    if part == 'A':
        df_act.iat[69, 1] = pd.Timestamp('2011-12-01 19:29:59')
        df_act.iat[78, 1] = pd.Timestamp('2011-12-02 12:20:59')
        df_act.iat[80, 1] = pd.Timestamp('2011-12-02 12:35:49')
    # 对于 OrdonezB，应用 correct_activities
    elif part == 'B':
        df_act, corrections = correct_activities(df_act, excepts=['Grooming'], retain_corrections=True)
        print("Corrections applied to OrdonezB activities:")
        for corr in corrections:
            print(corr)

    # 映射活动名称
    activity_mapping = {
        'Spare_Time/TV': 'relaxing on couch',
        'Grooming': 'personal care',
        'Toileting': 'toileting',
        'Sleeping': 'sleeping',
        'Breakfast': 'preparing breakfast',
        'Showering': 'showering',
        'Snack': 'snacking',
        'Lunch': 'preparing lunch',
        'Leaving': 'leaving home',
        'Dinner': 'preparing dinner'
    }
    df_act[ACTIVITY] = df_act[ACTIVITY].map(activity_mapping)

    # 删除“如厕”活动
    df_act = df_act[df_act[ACTIVITY] != 'toileting']

    # 处理设备数据
    df_sen = pd.read_csv(
        data_dir / sen_file,
        delimiter='\t+',
        skiprows=[0, 1],
        names=[START_TIME, END_TIME, 'location', 'type', 'place'],
        engine='python'
    )
    df_sen[DEVICE] = df_sen['place'] + '_' + df_sen['location'] + '_' + df_sen['type']
    df_sen[START_TIME] = pd.to_datetime(df_sen[START_TIME].str.strip())
    df_sen[END_TIME] = pd.to_datetime(df_sen[END_TIME].str.strip())

    device_areas = df_sen[['place', 'location', 'type', DEVICE]].drop_duplicates()

    return {
        'activities': df_act,
        'devices': df_sen,
        'device_areas': device_areas,
        'activity_list': df_act[ACTIVITY].unique().tolist(),
        'device_list': df_sen[DEVICE].unique().tolist()
    }



import pandas as pd
from pathlib import Path
import json

def load_milan_dataset(data_dir: str):
    """
    Load and process dataset from a given folder path.
    
    Args:
        folder_path (Path): Path to the folder containing the data file
        
    Returns:
        dict: Dictionary containing processed activities and devices DataFrames,
              along with their unique lists
    """
    # Constants
    ACTIVITY = 'activity'
    DEVICE = 'device'
    START_TIME = 'start_time'
    END_TIME = 'end_time'
    TIME = 'time'
    VALUE = 'value'
    
    # File paths
    fp = data_dir.joinpath("data")
    fp_corr = data_dir.joinpath('corrected_data.csv')
    
    # Fix data inconsistencies
    with open(fp, 'r') as f_o, open(fp_corr, 'w') as f_t:
        for i, line in enumerate(f_o.readlines()):
            s = line[:-1].split('\t')
            
            # Specific fixes for known data issues
            if i == 10285:  # Remove extra tab
                s.remove('')
            if i == 275005:  # Fix ON0 to ON
                s[2] = 'ON'
            if i == 433139:  # Fix O to ON for M019
                s[2] = 'ON'
            if i == 174353:  # Fix ON` to ON for M022
                s[2] = 'ON'
                
            new_line = ",".join(s)
            try:
                s[4]  # Test if activity exists
                new_line += "," + " ".join(s[4:])
            except IndexError:
                pass
                
            assert len(s) in [3, 4]
            f_t.write(new_line + "\n")
    
    # Load corrected data
    df = pd.read_csv(fp_corr,
                    sep=",",
                    na_values=True,
                    names=[START_TIME, 'id', VALUE, ACTIVITY],
                    engine='python')
    df[START_TIME] = pd.to_datetime(df[START_TIME], format='mixed')
    df = df.sort_values(by=START_TIME).reset_index(drop=True)
    
    # Process devices data
    df_dev = df.copy().drop(ACTIVITY, axis=1)
    bin_mask = (df_dev[VALUE] == 'ON') | (df_dev[VALUE] == 'OFF')
    
    # Binary devices processing
    df_binary = df_dev[bin_mask].copy()
    df_binary[VALUE] = (df_binary[VALUE] == 'ON')
    
    # Numeric devices processing
    num_mask = pd.to_numeric(df_dev[VALUE], errors='coerce').notnull()
    df_num = df_dev[num_mask].copy()
    df_num[VALUE] = df_num[VALUE].astype(float)
    
    # Categorical devices processing
    df_cat = df_dev[~num_mask & ~bin_mask]
    
    # Combine and format devices DataFrame
    df_dev = pd.concat([df_cat, df_binary, df_num], axis=0, ignore_index=True)
    df_dev.columns = [TIME, DEVICE, VALUE]
    df_dev = df_dev.sort_values(by=TIME).reset_index(drop=True)
    
    # Process activities data
    df_act = df.copy()[~df[ACTIVITY].isnull()][[START_TIME, ACTIVITY]]
    df_act[ACTIVITY] = df_act[ACTIVITY].astype(str).apply(lambda x: x.strip())
    
    act_list = sorted(list(df_act[ACTIVITY].unique()))
    new_df_lst = []
    
    for i in range(1, len(act_list), 2):
        activity = ' '.join(act_list[i].split(' ')[:-1])
        df_res = df_act[df_act[ACTIVITY] == act_list[i-1]].reset_index(drop=True)
        df_end = df_act[df_act[ACTIVITY] == act_list[i]].reset_index(drop=True)
        
        df_res[ACTIVITY] = activity
        df_res[END_TIME] = df_end[START_TIME]
        new_df_lst.append(df_res)
    
    # Combine and format activities DataFrame
    df_act = pd.concat(new_df_lst)
    df_act = df_act.reindex(columns=[START_TIME, END_TIME, ACTIVITY])
    df_act = df_act.sort_values(START_TIME).reset_index(drop=True)
    
    # Get unique lists
    lst_act = df_act[ACTIVITY].unique()
    lst_dev = df_dev[DEVICE].unique()
    
    return {
        'activities': df_act,
        'devices': df_dev,
        'activity_list': lst_act,
        'device_list': lst_dev
    }

# Example usage:
# folder_path = Path("your/data/folder")
# data = load_dataset(folder_path)