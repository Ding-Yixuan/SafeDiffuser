import pickle
import pandas as pd
import numpy as np
import os
import glob

def calculate_max_step_distance(trajectory):
    """
    辅助函数：计算一条轨迹中，连续两点之间的最大欧氏距离。
    用于衡量轨迹是否"瞬移"或不连续。
    """
    if trajectory is None or len(trajectory) < 2:
        return 0.0
    
    # 假设 trajectory 是 [N, 2] 或 [N, 4] 的数组，取前两维 (x, y)
    coords = np.array(trajectory)[:, :2]
    
    # 计算相邻两点的差值
    diffs = coords[1:] - coords[:-1]
    
    # 计算每一步的距离
    dists = np.linalg.norm(diffs, axis=1)
    
    # 返回最大的一次跳跃距离
    return np.max(dists)

def analyze_pickles(folder_path):
    # 1. 寻找文件
    pkl_files = glob.glob(os.path.join(folder_path, "RawData_*.pkl"))
    
    if not pkl_files:
        print("❌ 未找到 .pkl 文件，请检查路径！")
        return

    all_data = []
    
    # 2. 读取数据
    print(f"📂 正在分析路径: {folder_path}")
    for pkl_f in pkl_files:
        # print(f"  📖 读取: {os.path.basename(pkl_f)}")
        try:
            with open(pkl_f, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
        except Exception as e:
            print(f"  ⚠️ 无法读取 {pkl_f}: {e}")

    if not all_data:
        print("❌ 数据为空！")
        return

    # 3. 数据预处理 (关键步骤)
    # 确保所有需要的字段都存在，如果不存在则尝试计算或填 NaN
    processed_data = []
    for entry in all_data:
        # 复制一份，以免修改原始数据
        row = entry.copy()
        
        # --- A. 轨迹连续性 (最大步长) ---
        # 如果 pkl 里没有直接存 'max_step_dist'，但有 'observations' 或 'trajectory'
        if 'max_step_dist' not in row:
            traj = row.get('observations', row.get('trajectory', None))
            row['max_step_dist'] = calculate_max_step_distance(traj)
            
        # --- B. 确保其他键存在 (防止报错) ---
        # 如果你的 pkl 里键名不一样，请在这里修改 'get' 的第一个参数
        row.setdefault('is_collision', 0)       # 默认为0
        row.setdefault('collision_steps', 0)    # 默认为0
        row.setdefault('path_length', 0)        # 默认为0
        
        # 下面这三个如果你还没存，代码会填 NaN (空值)
        row.setdefault('inference_time', np.nan) # 推理时间
        row.setdefault('normalized_score', row.get('score', np.nan)) # 归一化得分
        row.setdefault('min_spec', row.get('safe_spec', np.nan))     # Spec评分 (取最小安全值)

        processed_data.append(row)

    df = pd.DataFrame(processed_data)
    
    print("\n" + "="*60)
    print(f"📊 数据加载完毕 | 总记录数: {len(df)}")
    print(f"🔍 包含算法: {df['algorithm'].unique()}")
    print("="*60 + "\n")

    # ==================== 生成分项 CSV 报表 ====================
    
    # 定义我们要分析的指标：(DataFrame列名, 输出文件名, 描述)
    metrics_config = [
        ('is_collision',    'Metric_Collision_Bool.csv',  '是否碰撞 (0/1)'),
        ('collision_steps', 'Metric_Collision_Steps.csv', '碰撞步数'),
        ('max_step_dist',   'Metric_Max_Step_Dist.csv',   '最大单步跳跃距离'),
        ('s_spec',          'Metric_S_Spec.csv',          'S-Spec安全评分(圆形)'),
        ('c_spec',          'Metric_C_Spec.csv',          'C-Spec安全评分(矩形)'),
        ('normalized_score','Metric_Normalized_Score.csv','归一化得分'),
        ('inference_time',  'Metric_Inference_Time.csv',  '推理时间(秒)')
    ]

    for col_name, file_name, desc in metrics_config:
        if col_name in df.columns and not df[col_name].isna().all():
            try:
                pivot_df = df.pivot_table(index=['case_idx', 'run_idx'], 
                                          columns='algorithm', 
                                          values=col_name)
                save_path = os.path.join(folder_path, file_name)
                pivot_df.to_csv(save_path)
                print(f"✅ [保存成功] {desc} -> {file_name}")
            except Exception as e:
                print(f"⚠️ [跳过] 无法生成 {desc}: 列可能不存在或全为空")
        else:
            print(f"⚠️ [缺失] 数据中未找到有效列: {col_name} ({desc})")

    # ==================== 最终统计摘要 (Mean ± Std) ====================
    print("\n === 📈 最终统计摘要 (Mean ± Std) === \n")
    
    stats_summary = []

    for algo in df['algorithm'].unique():
        sub_df = df[df['algorithm'] == algo]
        
        algo_stats = {'Algorithm': algo}
        
        # 1. 碰撞率 (单独处理，算总数和百分比)
        coll_sum = sub_df['is_collision'].sum()
        coll_rate = sub_df['is_collision'].mean() * 100
        algo_stats['Collision Rate (%)'] = f"{coll_rate:.2f}%"
        algo_stats['Total Collisions'] = int(coll_sum)

        # 2. 其他指标 (算均值和方差)
        # 这里的 key 是你在 DataFrame 里的列名，value 是你想显示在报表里的名字
        metric_map = {
            'collision_steps': 'Avg Coll Steps',
            'max_step_dist':   'Max Step Dist (m)',
            's_spec':          'Avg S-Spec', 
            'c_spec':          'Avg C-Spec', 
            'normalized_score':'Avg Score',
            'inference_time':  'Avg Time (s)'
        }

        for col, label in metric_map.items():
            if col in sub_df.columns:
                mean_val = sub_df[col].mean()
                std_val = sub_df[col].std()
                # 格式化为 "均值 ± 标准差"
                algo_stats[label] = f"{mean_val:.4e} ± {std_val:.4e}"
            else:
                algo_stats[label] = "N/A"
        
        stats_summary.append(algo_stats)

    df_stats = pd.DataFrame(stats_summary)
    
    # 打印到控制台 (转置一下比较好读)
    print(df_stats.set_index('Algorithm').T)
    
    stats_path = os.path.join(folder_path, "Final_Statistics_Summary.csv")
    df_stats.to_csv(stats_path, index=False)
    print(f"\n✅ 汇总统计已保存: {stats_path}")

if __name__ == "__main__":
    # 换成你的实际路径
    target_dir = "/home/lqz27/dyx_ws/SafeDiffuser/logs/maze2d-custom-v1/plans/release_H256_T128_LimitsNormalizer_b1_condFalse/58/"
    analyze_pickles(target_dir)