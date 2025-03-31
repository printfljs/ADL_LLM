
import pandas as pd
import numpy as np

def generate_window_description_hci_A(win_sensor_raw: pd.DataFrame) -> str:
    """
    根据原始传感器数据生成自然语言描述（无需编码）
    输入：单个窗口的原始传感器数据（未编码）
    输出：自然语言描述字符串
    """
    descriptions = []
    previous_time = None
    
    for _, row in win_sensor_raw.iterrows():
        time_str = row['time']
        device = row['device']
        value = row['value']
        location = row['location']
        sensor_type = row['type']
        place = row['place']
        
        # 时间处理
        time_obj = pd.to_datetime(time_str)
        current_time = time_obj.strftime("%I:%M %p")
        
        # 生成基础描述
        desc = ""
        if "Seat_Pressure" in device:
            action = "sitting on the seat" if value else "not sitting on the seat"
            desc = f"Here, the subject is {action} in the {place}."
        elif "Bed_Pressure" in device:
            action = "resting on the bed" if value else "not resting on the bed"
            desc = f"Here, the subject is {action} in the {place}."
        elif "Maindoor_Magnetic" in device:
            action = "open" if value else "closed"
            desc = f"The {place} main door is {action}."
        elif "Cooktop_PIR" in device:
            action = "near the cooktop" if value else "away from the cooktop"
            desc = f"Here, the subject is {action} in the {place}."
        elif "Microwave_Electric" in device:
            action = "turns on the microwave" if value else "turns off the microwave"
            desc = f"Here, the subject {action} in the {place}."
        elif "Fridge_Magnetic" in device:
            action = "opens the fridge" if value else "closes the fridge"
            desc = f"Here, the subject {action} in the {place}."
        elif "Toilet_Flush" in device:
            action = "flushed the toilet" if value else "did not flush the toilet"
            desc = f"The subject {action} in the {place}."
        elif "Shower_PIR" in device:
            action = "using the shower" if value else "not using the shower"
            desc = f"Here, the subject is {action} in the {place}."
        elif "Basin_PIR" in device:
            action = "using the basin" if value else "not using the basin"
            desc = f"Here, the subject is {action} in the {place}."
        elif "Cabinet_Magnetic" in device:
            action = "opens the cabinet" if value else "closes the cabinet"
            desc = f"Here, the subject {action} in the {place}."
        elif "Cupboard_Magnetic" in device:
            action = "opens the cupboard" if value else "closes the cupboard"
            desc = f"Here, the subject {action} in the {place}."
        elif "Toaster_Electric" in device:
            action = "turns on the toaster" if value else "turns off the toaster"
            desc = f"Here, the subject {action} in the {place}."
        else:
            desc = f"The {place} {location} {sensor_type} sensor detected {'activity' if value else 'no activity'}."

        
        # 时间差描述
        if previous_time:
            time_diff = (time_obj - previous_time).total_seconds()
            desc = f"After {int(time_diff)} seconds, {desc}"
        else:
            desc = f"At {current_time}, {desc}"
        
        descriptions.append(desc)
        previous_time = time_obj
    
    return " ".join(descriptions)


import pandas as pd

def generate_window_description_hci_A_with_status(win_sensor_raw: pd.DataFrame) -> str:
    """
    根据原始传感器数据生成自然语言描述（无需编码）
    输入：单个窗口的原始传感器数据（未编码）
    输出：自然语言描述字符串
    """
    descriptions = []
    previous_time = None
    
    for _, row in win_sensor_raw.iterrows():
        time_str = row['time']
        device = row['device']
        value = row['value']
        location = row['location']
        sensor_type = row['type']
        place = row['place']
        status = row['sensor_status']
        
        # 时间处理
        time_obj = pd.to_datetime(time_str)
        current_time = time_obj.strftime("%I:%M %p")
        
        # 生成基础描述
        desc = ""
        if "Seat_Pressure" in device:
            action = "sitting on the seat" if value else "left the seat"
            desc = f"Here, the subject {action} in the {place}."
        elif "Bed_Pressure" in device:
            action = "resting on the bed" if value else "left the bed"
            desc = f"Here, the subject {action} in the {place}."
        elif "Maindoor_Magnetic" in device:
            action = "open" if value else "closed"
            desc = f"The {place} main door is {action}."
        elif "Cooktop_PIR" in device:
            action = "near the cooktop" if value else "away from the cooktop"
            desc = f"Here, the subject is {action} in the {place}."
        elif "Microwave_Electric" in device:
            action = "turns on the microwave" if value else "turns off the microwave"
            desc = f"Here, the subject {action} in the {place}."
        elif "Fridge_Magnetic" in device:
            action = "opens the fridge" if value else "closed the fridge"
            desc = f"Here, the subject {action} in the {place}."
        elif "Toilet_Flush" in device:
            action = "flushed the toilet" if value else "finish flushing the toilet"
            desc = f"The subject {action} in the {place}."
        elif "Shower_PIR" in device:
            action = "using the shower" if value else "finished showering"
            desc = f"Here, the subject {action} in the {place}."
        elif "Basin_PIR" in device:
            action = "using the basin" if value else "finished using the basin"
            desc = f"Here, the subject {action} in the {place}."
        elif "Cabinet_Magnetic" in device:
            action = "opens the cabinet" if value else "closed the cabinet"
            desc = f"Here, the subject {action} in the {place}."
        elif "Cupboard_Magnetic" in device:
            action = "opens the cupboard" if value else "closed the cupboard"
            desc = f"Here, the subject {action} in the {place}."
        elif "Toaster_Electric" in device:
            action = "turns on the toaster" if value else "turns off the toaster"
            desc = f"Here, the subject {action} in the {place}."
        else:
            desc = f"The {place} {location} {sensor_type} sensor detected {'activity' if value else 'no activity'}."

        # 根据状态修改描述
        if status == "already_active":
            desc = desc.replace("is ", "was already ")
            if "turns" in desc:
                desc = desc.replace("turns", "had already turned")
            elif "opens" in desc:
                desc = desc.replace("opens", "had already opened")
            elif "closes" in desc:
                desc = desc.replace("closes", "had already closed")
        elif status == "persistent":
            desc = desc.replace("is ", "remains ")
            if "turns" in desc:
                desc = desc.replace("turns", "continues to turn")
            elif "opens" in desc:
                desc = desc.replace("opens", "remains open")
            elif "closes" in desc:
                desc = desc.replace("closes", "remains closed")
        elif status == "keep_on":
            desc = desc.replace("is ", "has been continuously ")
            if "turns" in desc:
                desc = desc.replace("turns", "has been continuously turning")
            elif "opens" in desc:
                desc = desc.replace("opens", "has been continuously open")
            elif "closes" in desc:
                desc = desc.replace("closes", "has been continuously closed")
        elif status == "inner":
            desc = desc.replace("is ", "was briefly ")
            if "turns" in desc:
                desc = desc.replace("turns", "briefly turned")
            elif "opens" in desc:
                desc = desc.replace("opens", "briefly opened")
            elif "closes" in desc:
                desc = desc.replace("closes", "briefly closed")

        # 时间差描述
        if previous_time:
            time_diff = (time_obj - previous_time).total_seconds()
            desc = f"After {int(time_diff)} seconds, {desc}"
        else:
            desc = f"At {current_time}, {desc}"
        
        descriptions.append(desc)
        previous_time = time_obj
    
    return " ".join(descriptions)


def generate_window_description_hci_B(win_sensor_raw: pd.DataFrame) -> str:
    """
    根据原始传感器数据生成自然语言描述（无需编码）
    输入：单个窗口的原始传感器数据（未编码）
    输出：自然语言描述字符串
    """
    descriptions = []
    previous_time = None

    for _, row in win_sensor_raw.iterrows():
        time_str = row['time']
        device = row['device']
        value = row['value']
        location = row['location']
        sensor_type = row['type']
        place = row['place']

        # 时间处理
        time_obj = pd.to_datetime(time_str)
        current_time = time_obj.strftime("%I:%M %p")

        # 生成基础描述
        desc = ""
        if "Seat_Pressure" in device:
            action = "sitting on the seat" if value else "not sitting on the seat"
            desc = f"Here, the subject is {action} in the {place}."
        elif "Bed_Pressure" in device:
            action = "resting on the bed" if value else "not resting on the bed"
            desc = f"Here, the subject is {action} in the {place}."
        elif "Maindoor_Magnetic" in device:
            action = "open" if value else "closed"
            desc = f"The {place} main door is {action}."
        elif "Microwave_Electric" in device:
            action = "turns on the microwave" if value else "turns off the microwave"
            desc = f"Here, the subject {action} in the {place}."
        elif "Fridge_Magnetic" in device:
            action = "opens the fridge" if value else "closes the fridge"
            desc = f"Here, the subject {action} in the {place}."
        elif "Toilet_Flush" in device:
            action = "flushed the toilet" if value else "did not flush the toilet"
            desc = f"The subject {action} in the {place}."
        elif "Shower_PIR" in device:
            action = "using the shower" if value else "not using the shower"
            desc = f"Here, the subject is {action} in the {place}."
        elif "Basin_PIR" in device:
            action = "using the basin" if value else "not using the basin"
            desc = f"Here, the subject is {action} in the {place}."
        elif "Cupboard_Magnetic" in device:
            action = "opens the cupboard" if value else "closes the cupboard"
            desc = f"Here, the subject {action} in the {place}."
        elif "Door Kitchen_PIR" in device:
            action = "passes through the kitchen door" if value else "not passing through the kitchen door"
            desc = f"Here, the subject is {action}."
        elif "Door Bathroom_PIR" in device:
            action = "passes through the bathroom door" if value else "not passing through the bathroom door"
            desc = f"Here, the subject is {action}."
        elif "Door Bedroom_PIR" in device:
            action = "passes through the bedroom door" if value else "not passing through the bedroom door"
            desc = f"Here, the subject is {action}."
        else:
            desc = f"The {place} {location} {sensor_type} sensor detected {'activity' if value else 'no activity'}."

        # 时间差描述
        if previous_time:
            time_diff = (time_obj - previous_time).total_seconds()
            desc = f"After {int(time_diff)} seconds, {desc}"
        else:
            desc = f"At {current_time}, {desc}"

        descriptions.append(desc)
        previous_time = time_obj

    return " ".join(descriptions)


