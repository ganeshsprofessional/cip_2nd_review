import json
import os
import glob
import math
import numpy as np
import pandas as pd

# -----------------------------
# Configuration & Thresholds
# -----------------------------
POS_TOLERANCE_METERS = 2.0  # Allowable GPS noise before flagging as attack
SPD_TOLERANCE_MPS = 1.0     # Allowable speed noise before flagging as attack

def load_ground_truth(folder_path):
    """
    Loads all ground truth files into a dictionary for O(1) lookups.
    Key: (sender_id, sendTime)
    Value: Dictionary of actual kinematic data (pos, spd, acl, hed)
    """
    gt_map = {}
    gt_files = glob.glob(os.path.join(folder_path, "traceGroundTruthJSON-*.json"))
    
    for file_path in gt_files:
        with open(file_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                if data.get("type") == 4:
                    sender = data.get("sender")
                    send_time = data.get("sendTime")
                    gt_map[(sender, send_time)] = data
                    
    print(f"Loaded {len(gt_map)} ground truth records.")
    return gt_map

def check_if_attack(type3_data, gt_map):
    """
    1. Accept missing: Returns 0 if no ground truth is found.
    2. Tolerance threshold: Returns 1 only if difference exceeds noise tolerance.
    """
    sender = type3_data.get("sender")
    send_time = type3_data.get("sendTime")
    
    gt_data = gt_map.get((sender, send_time))
    if not gt_data:
        return 0  # Missing ground truth assumed normal
    
    # Extract positions and speeds
    t3_pos = type3_data.get("pos", [0,0,0])
    gt_pos = gt_data.get("pos", [0,0,0])
    
    t3_spd = type3_data.get("spd", [0,0,0])
    gt_spd = gt_data.get("spd", [0,0,0])
    
    # Calculate Euclidean differences
    pos_diff = math.sqrt(sum((a - b)**2 for a, b in zip(t3_pos, gt_pos)))
    spd_diff = math.sqrt(sum((a - b)**2 for a, b in zip(t3_spd, gt_spd)))
    
    if pos_diff > POS_TOLERANCE_METERS or spd_diff > SPD_TOLERANCE_MPS:
        return 1
    
    return 0

def process_logs_to_dataframe(folder_path, gt_map):
    """
    Reads receiver logs, uses the most recent Type 2 message as ego state,
    and extracts Type 3 messages into flat rows.
    """
    log_files = glob.glob(os.path.join(folder_path, "traceJSON-*.json"))
    rows = []
    
    for file_path in log_files:
        # Extract receiver ID from filename
        filename = os.path.basename(file_path)
        parts = filename.split('-')
        if len(parts) >= 6:
            receiver_id = int(parts[5].split('.')[0])
        else:
            continue

        # Default ego state in case Type 3 arrives before any Type 2
        latest_ego_state = {
            "pos": [0.0, 0.0, 0.0],
            "spd": [0.0, 0.0, 0.0]
        }
        
        with open(file_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                msg_type = data.get("type")
                
                if msg_type == 2:
                    # Update recent ego state
                    latest_ego_state["pos"] = data.get("pos", [0,0,0])
                    latest_ego_state["spd"] = data.get("spd", [0,0,0])
                    
                elif msg_type == 3:
                    # Parse Type 3 and compute relative features
                    t3_pos = data.get("pos", [0,0,0])
                    t3_spd = data.get("spd", [0,0,0])
                    t3_acl = data.get("acl", [0,0,0])
                    t3_hed = data.get("hed", [0,0,0])
                    
                    is_attack = check_if_attack(data, gt_map)
                    
                    row = {
                        "rcvTime": data.get("rcvTime"),
                        "sendTime": data.get("sendTime"),
                        "receiver": receiver_id,
                        "sender": data.get("sender"),
                        "messageID": data.get("messageID"),
                        
                        # Relative features for GNN
                        "rel_pos_x": t3_pos[0] - latest_ego_state["pos"][0],
                        "rel_pos_y": t3_pos[1] - latest_ego_state["pos"][1],
                        "rel_spd_x": t3_spd[0] - latest_ego_state["spd"][0],
                        "rel_spd_y": t3_spd[1] - latest_ego_state["spd"][1],
                        
                        # Raw absolute features
                        "pos_x": t3_pos[0],
                        "pos_y": t3_pos[1],
                        "spd_x": t3_spd[0],
                        "spd_y": t3_spd[1],
                        "acl_x": t3_acl[0],
                        "acl_y": t3_acl[1],
                        "hed_x": t3_hed[0],
                        "hed_y": t3_hed[1],
                        
                        "is_attack": is_attack
                    }
                    rows.append(row)
                    
    return pd.DataFrame(rows)

def calculate_kinematic_features(df):
    """
    Calculates advanced features: distance difference, beacon rate, 
    and rolling neighborhood averages for the DQN reward function.
    """
    # 1. Magnitudes
    df["speed"] = np.sqrt(df["spd_x"]**2 + df["spd_y"]**2)
    df["acceleration"] = np.sqrt(df["acl_x"]**2 + df["acl_y"]**2)
    
    # 2. Distance Difference (\delta_{j,t})
    df = df.sort_values(by=["sender", "rcvTime"])
    group = df.groupby("sender")
    
    df["dt"] = df["rcvTime"] - group["rcvTime"].shift(1)
    df["euclidean_dist"] = np.sqrt(
        (df["pos_x"] - group["pos_x"].shift(1))**2 +
        (df["pos_y"] - group["pos_y"].shift(1))**2
    )
    
    df["kinematic_dist"] = (
        ((group["speed"].shift(1) + df["speed"]) / 2) * df["dt"] +
        0.5 * ((group["acceleration"].shift(1) + df["acceleration"]) / 2) * df["dt"]**2
    )
    df["distance_diff"] = (df["kinematic_dist"] - df["euclidean_dist"]).fillna(0.0)
    
    # 3. Beacon Rate (B_{j,t})
    df['msg_count'] = group.cumcount() + 1
    df['elapsed_time'] = df['rcvTime'] - group['rcvTime'].transform('first')
    df['beacon_rate'] = np.where(df['elapsed_time'] == 0, 0.0, df['msg_count'] / df['elapsed_time'])
    
    # 4. Neighborhood Average Speed (\mu_{i,t}, \sigma_{i,t})
    # Grouped by receiver and a rolling 1-second window
    df['rcvTime_td'] = pd.to_timedelta(df['rcvTime'], unit='s')
    df = df.set_index('rcvTime_td').sort_index()
    
    # Calculate rolling mean and std for each receiver's neighborhood
    rolling_stats = df.groupby('receiver')['speed'].rolling('1s')
    df['avg_speed_1s'] = rolling_stats.mean().reset_index(level=0, drop=True)
    df['stddev_speed_1s'] = rolling_stats.std().fillna(0.0).reset_index(level=0, drop=True)
    
    df = df.reset_index(drop=True)
    
    # Drop intermediate columns used for calculation
    cols_to_drop = ['dt', 'euclidean_dist', 'kinematic_dist', 'msg_count', 'elapsed_time']
    df = df.drop(columns=cols_to_drop)
    
    # Sort back to chronological order for the GNN graph builder
    df = df.sort_values(by=["receiver", "rcvTime"]).reset_index(drop=True)
    
    return df

def main(input_folder, output_csv):
    print(f"Starting preprocessing for folder: {input_folder}")
    
    # Step 1: Load Ground Truth map
    gt_map = load_ground_truth(input_folder)
    
    # Step 2: Parse Type 2 and Type 3 logs into a DataFrame
    df = process_logs_to_dataframe(input_folder, gt_map)
    print(f"Extracted {len(df)} Type 3 messages.")
    
    # Step 3: Calculate temporal and kinematic features
    df = calculate_kinematic_features(df)
    
    # Step 4: Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Successfully saved preprocessed data to {output_csv}")

if __name__ == "__main__":
    # Example usage:
    # Set this to the path containing your traceJSON and traceGroundTruthJSON files
    FOLDER_PATH = "C:\\Users\\ganes\\projects\\cip_test\\cip_2nd_review\\veremi\\VeReMi_25200_28800_2025-11-15_13_57_9\\" 
    OUTPUT_FILE = "gnn_dqn_ready_dataset.csv"
    
    main(FOLDER_PATH, OUTPUT_FILE)