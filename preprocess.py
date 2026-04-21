import pandas as pd
import numpy as np

def create_all_summaries():
    print("🚀 【一括処理】事前集計を開始します...")

    # --- 1. 全データの読み込み (一回で済みます) ---
    print("📂 データを読み込み中...")
    df_stores = pd.read_csv("260408店舗別一覧.csv", encoding="cp932", usecols=[0, 9])
    df_stores.columns = ['店舗ID', '都道府県名']

    df_m = pd.read_csv("【260406】機種ダウンロードデータ.csv", encoding="cp932", low_memory=False)
    df_m.columns = df_m.columns.str.strip()
    
    # マスタの列名整理 (PW機種コード=pcode, 区分=D列, 紐付け=E列)
    df_m['pcode'] = df_m.iloc[:, 4].astype(str).str.strip() 
    df_m['ps_kbn'] = pd.to_numeric(df_m.iloc[:, 3], errors='coerce').fillna(1)
    if 'メーカー名' in df_m.columns:
        df_m = df_m.rename(columns={'メーカー名': 'メーカー'})

    # 巨大な設置詳細データ
    df_d = pd.read_csv("P_dai_260408.csv", encoding="cp932", usecols=[0, 1, 2, 3])
    df_d.columns = ['店舗ID', 'pcode', 'amt', '台数']
    df_d['pcode'] = df_d['pcode'].astype(str).str.strip()
    df_d['台数'] = pd.to_numeric(df_d['台数'], errors='coerce').fillna(0)

    # --- 2. データの紐付け ---
    print("🔗 データを紐付け中...")
    df_merged = pd.merge(df_d, df_stores, on='店舗ID', how='left')
    df_merged = pd.merge(df_merged, df_m[['pcode', 'ps_kbn', 'メーカー']], on='pcode', how='left')

    # --- 3. 貸区分の判定 (高速ベクトル演算) ---
    print("📊 区分判定中...")
    amt = pd.to_numeric(df_merged['amt'], errors='coerce').fillna(0)
    kbn = df_merged['ps_kbn']
    conditions = [
        (kbn == 1) & (amt >= 4.0), (kbn == 1) & (amt < 4.0),
        (kbn == 2) & (amt >= 10.0), (kbn == 2) & (amt < 10.0)
    ]
    choices = ["通常", "低貸", "通常", "低貸"]
    df_merged['貸区分'] = np.select(conditions, choices, default="不明")

    # --- 4. 【出力①】メーカー・店舗別集計 (これまでの分) ---
    print("📉 メーカー別集計を保存中...")
    maker_summary = df_merged.groupby(['店舗ID', '貸区分', 'メーカー', 'ps_kbn', 'pcode'])['台数'].sum().reset_index()
    maker_summary.to_parquet("maker_summary_260408.parquet")

    # --- 5. 【出力②】都道府県別集計 (新しい分) ---
    print("📉 都道府県別集計を保存中...")
    pref_summary = df_merged.groupby(['都道府県名', '貸区分'])['台数'].sum().unstack(fill_value=0)
    # 合計列の追加
    pref_summary['合計'] = pref_summary.sum(axis=1)
    pref_summary.to_parquet("pref_summary_260408.parquet")

    print("\n" + "="*30)
    print("✅ すべての処理が完了しました！")
    print("・maker_summary_260408.parquet (メーカー別)")
    print("・pref_summary_260408.parquet (都道府県別)")
    print("="*30)

if __name__ == "__main__":
    create_all_summaries()