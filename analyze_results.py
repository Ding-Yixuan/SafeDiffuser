import pickle
import pandas as pd
import numpy as np
import os
import glob

def analyze_pickles(folder_path):
    # 1. 寻找该文件夹下所有的 RawData_*.pkl
    pkl_files = glob.glob(os.path.join(folder_path, "RawData_*.pkl"))
    
    if not pkl_files:
        print("❌ 未找到 .pkl 文件，请检查路径！")
        return

    all_data = []
    
    # 2. 读取并合并所有数据
    for pkl_f in pkl_files:
        print(f"📖 读取文件: {os.path.basename(pkl_f)}")
        with open(pkl_f, 'rb') as f:
            data = pickle.load(f)
            # 确保 data 是 list
            if isinstance(data, list):
                all_data.extend(data)
            else:
                print(f"⚠️ 文件 {pkl_f} 格式不对，跳过")

    # 转为 Pandas DataFrame，方便处理
    df = pd.DataFrame(all_data)
    
    print("\n" + "="*50)
    print(f"数据加载完毕，共 {len(df)} 条记录")
    print("包含了以下算法: ", df['algorithm'].unique())
    print("="*50 + "\n")

    # ==================== 指标 1: 是否碰撞 (Is Collision) ====================
    # 按照算法分组，取 list
    # pivot table: 行=Case/Run, 列=Algorithm
    # 但你要求的是 "保存为一个csv"，通常是一列一个算法，或者一行一条记录
    
    # 我们生成一个汇总表：Index 是 (Case, Run), Columns 是算法名, Value 是 is_collision
    df_collision = df.pivot_table(index=['case_idx', 'run_idx'], 
                                  columns='algorithm', 
                                  values='is_collision')
    
    csv_path_1 = os.path.join(folder_path, "Metric1_Collision_Bool.csv")
    df_collision.to_csv(csv_path_1)
    print(f"✅ 指标1 (碰撞0/1) CSV 已保存: {csv_path_1}")

    # ==================== 指标 2: 轨迹碰撞步数 (Collision Steps) ====================
    df_steps = df.pivot_table(index=['case_idx', 'run_idx'], 
                              columns='algorithm', 
                              values='collision_steps')
    
    csv_path_2 = os.path.join(folder_path, "Metric2_Collision_Steps.csv")
    df_steps.to_csv(csv_path_2)
    print(f"✅ 指标2 (碰撞步数) CSV 已保存: {csv_path_2}")
    
    # ==================== 指标 3: 轨迹长度 (Path Length) ====================
    df_length = df.pivot_table(index=['case_idx', 'run_idx'], 
                               columns='algorithm', 
                               values='path_length')
    
    # 你提到 "采集所有轨迹的这个距离值的最大值"
    # 我理解为：算出每条轨迹的长度后，找出最大的一条（Max Path Length）
    # 这里我们先存下所有长度的 CSV
    csv_path_3 = os.path.join(folder_path, "Metric3_Path_Length.csv")
    df_length.to_csv(csv_path_3)
    print(f"✅ 指标3 (轨迹长度) CSV 已保存: {csv_path_3}")

    # ==================== 统计计算 (Sum, Mean, Var) ====================
    print("\n📊 === 最终统计报告 === 📊\n")
    
    stats_summary = []

    for algo in df['algorithm'].unique():
        sub_df = df[df['algorithm'] == algo]
        
        # 1. 碰撞次数 (0/1) 的统计
        # Sum = 总碰撞次数, Mean = 碰撞率
        col_sum = sub_df['is_collision'].sum()
        col_mean = sub_df['is_collision'].mean()
        col_var = sub_df['is_collision'].var()
        
        # 2. 碰撞步数 的统计
        step_sum = sub_df['collision_steps'].sum()
        step_mean = sub_df['collision_steps'].mean()
        step_var = sub_df['collision_steps'].var()
        
        # 3. 轨迹长度 的统计
        # Max = 所有轨迹里最长的那条
        len_max = sub_df['path_length'].max()
        len_mean = sub_df['path_length'].mean()
        len_var = sub_df['path_length'].var()
        
        stats_summary.append({
            'Algorithm': algo,
            'Collision_Count(Sum)': col_sum,
            'Collision_Rate(Mean)': col_mean,
            'Collision_Bool_Var': col_var,
            'Collision_Steps_Sum': step_sum,
            'Collision_Steps_Mean': step_mean,
            'Collision_Steps_Var': step_var,
            'Path_Length_Max': len_max,
            'Path_Length_Mean': len_mean,
            'Path_Length_Var': len_var
        })

    df_stats = pd.DataFrame(stats_summary)
    print(df_stats.T) # 转置打印，方便看
    
    stats_path = os.path.join(folder_path, "Final_Statistics_Report.csv")
    df_stats.to_csv(stats_path, index=False)
    print(f"\n✅ 统计报表已保存: {stats_path}")

# 使用方法：
# 将所有 .pkl 文件放在同一个文件夹下（比如 logs/benchmark_results/）
# 然后运行此函数
if __name__ == "__main__":
    # 假设你的 pkl 文件都在当前目录的 'data' 文件夹下
    # 你可以修改为你实际的保存路径
    target_dir = "/home/lqz27/dyx_ws/SafeDiffuser/logs/maze2d-custom-v1/plans/release_H256_T128_LimitsNormalizer_b1_condFalse/58/" 
    # 或者直接用当前目录 "."
    analyze_pickles(target_dir)