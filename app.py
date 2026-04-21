import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pydeck as pdk
import traceback
import os
import requests

# --- Google Drive から大容量ファイルを自動ダウンロード ---
GDRIVE_FILE_ID = "14iNrXAqNIMC5tSCN-Q5PR1tbAI7fqkDd"
LOCAL_PATH = "P_dai_260408.csv"

@st.cache_resource
def download_p_dai():
    if os.path.exists(LOCAL_PATH):
        return  # すでにダウンロード済みならスキップ
    with st.spinner("📥 設置データをダウンロード中... (初回のみ、しばらくお待ちください)"):
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        session = requests.Session()
        response = session.get(url, stream=True)
        # 大容量ファイルの場合、確認トークンが必要
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                params = {"id": GDRIVE_FILE_ID, "confirm": value}
                response = session.get("https://drive.google.com/uc", params=params, stream=True)
                break
        with open(LOCAL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

download_p_dai()

# ページの設定
st.set_page_config(page_title="店舗・機種分析ポータル Pro", layout="wide")

# --- データ読み込み関数 ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("260408店舗別一覧.csv", encoding="cp932")
        df = df.rename(columns={'緯度（世界測地系）': 'latitude', '経度（世界測定系）': 'longitude'})
        df.columns = df.columns.str.strip()
        if len(df.columns) > 9:
            df.rename(columns={df.columns[8]: "都道府県ID", df.columns[9]: "都道府県名"}, inplace=True)
        num_cols = df.select_dtypes(include=['number']).columns
        df[num_cols] = df[num_cols].fillna(0)
        return df
    except Exception as e:
        st.error(f"メインデータの読み込みに失敗: {e}")
        return None

@st.cache_data
def load_machine_master():
    try:
        # 重い処理はpreprocessで行うため、ここではマスタの基本情報のみ読み込み
        df_m = pd.read_csv("【260406】機種ダウンロードデータ.csv", encoding="cp932")
        df_m.columns = df_m.columns.str.strip()
        df_m['pcode'] = df_m['PW機種コード'].astype(str)
        
        # 発売日変換
        df_m['date_dt'] = pd.to_datetime(df_m.iloc[:, 10], format='%Y%m%d', errors='coerce')
        df_m['発売日'] = df_m['date_dt'].dt.strftime('%Y/%m/%d')
        
        # メーカー・区分情報の整理
        df_m['メーカー'] = df_m[df_m.columns[2]]
        df_m['区分'] = df_m['パチンコ/スロット区分']
        
        # スマスロ・スマパチ等の特徴判定（既存ロジック維持）
        if 'コインサイズ' in df_m.columns:
            df_m['スマスロ判定用'] = pd.to_numeric(df_m['コインサイズ'], errors='coerce')
        else:
            df_m['スマスロ判定用'] = pd.to_numeric(df_m.iloc[:, 32], errors='coerce')
        df_m['スマパチ判定用'] = df_m.iloc[:, 48].astype(str).str.strip()
        df_m['デカへそ判定用'] = df_m.iloc[:, 52].astype(str).str.strip()

        return df_m.drop_duplicates(subset=['pcode'])
    except Exception as e:
        st.error(f"機種マスタの読み込みに失敗: {e}")
        return pd.DataFrame()

# --- 【新規】事前集計済みデータの読み込み ---
@st.cache_data
def load_summaries():
    try:
        # preprocess.py で作ったファイルを読み込む
        maker_sum = pd.read_parquet("maker_summary_260408.parquet")
        pref_sum = pd.read_parquet("pref_summary_260408.parquet")
        return maker_sum, pref_sum
    except Exception as e:
        st.warning(f"事前集計データの読み込みに失敗しました。preprocess.pyを実行してください: {e}")
        return None, None

# --- 特定店舗の在庫取得 (高速版) ---
def get_store_inventory(target_id, df_master):
    try:
        # ⚠️ ここは「店舗ごとの全機種」を表示するため、依然として P_dai を使いますが、
        # usecols で必要な列だけに絞り、かつ必要な店舗だけを即座に抽出します。
        df_inv = pd.read_csv("P_dai_260408.csv", encoding="cp932", 
                             usecols=['店舗ID', 'pcode', '貸玉量', '設置台数'])
        df_inv['pcode'] = df_inv['pcode'].astype(str)
        
        res = df_inv[df_inv['店舗ID'] == target_id].copy()
        # マスタと結合
        res = pd.merge(res, df_master, on='pcode', how='left')
        
        # 区分判定
        def judge_kashi(row):
            ps_type = row['区分']
            try: amt = float(row['貸玉量'])
            except: return "不明"
            if ps_type == 1: return "通常P" if 4.0 <= amt <= 5.0 else "低貸P"
            elif ps_type == 2: return "通常S" if 20.0 <= amt < 30.0 else "低貸S"
            return "不明"
            
        res['貸区分'] = res.apply(judge_kashi, axis=1)
        res = res.sort_values(by=['貸玉量', '設置台数'], ascending=[False, False])
        return res
    except Exception as e:
        st.error(f"詳細台数データの取得に失敗: {e}")
        return pd.DataFrame()


@st.cache_data
def load_price_data():
    try:
        df_p = pd.read_csv("260407中古機.csv", encoding="cp932")
        df_p.columns = df_p.columns.str.strip()
        res = df_p.iloc[:, [0, 7]].copy()
        res.columns = ['pcode', '中古価格']
        res['pcode'] = res['pcode'].astype(str)
        res['中古価格'] = pd.to_numeric(res['中古価格'], errors='coerce').fillna(0)
        return res
    except:
        return pd.DataFrame()


def get_store_inventory(target_id, df_master):
    try:
        df_inv = pd.read_csv("P_dai_260408.csv", encoding="cp932", usecols=['店舗ID', 'pcode', '貸玉量', '設置台数'])
        df_inv['pcode'] = df_inv['pcode'].astype(str)
        res = df_inv[df_inv['店舗ID'] == target_id].copy()
        res = pd.merge(res, df_master, on='pcode', how='left')
        
        def judge_kashi(row):
            ps_type = row['区分']
            try: amt = float(row['貸玉量'])
            except: return "不明"
            if ps_type == 1: return "通常P" if 4.0 <= amt <= 5.0 else "低貸P"
            elif ps_type == 2: return "通常S" if 20.0 <= amt < 30.0 else "低貸S"
            return "不明"
            
        res['貸区分'] = res.apply(judge_kashi, axis=1)
        res = res.sort_values(by=['貸玉量', '設置台数'], ascending=[False, False])
        return res
    except Exception as e:
        st.error(f"詳細台数データの取得に失敗: {e}")
        return pd.DataFrame()

# --- 共通ヘルパー関数 ---
def calculate_distance(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2): return 999.0
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = np.sin(np.radians(lat2-lat1)/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(np.radians(lon2-lon1)/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def format_pct(val):
    try:
        f_val = float(val)
        return f"{f_val * 100:.2f}%" if abs(f_val) <= 1.0 and f_val != 0 else f"{f_val:.2f}%"
    except: return "0.00%"

def format_yen(val):
    try: return f"{int(val):,} 円" if val > 0 else "0 円"
    except: return "0 円"

def get_rank(df, col, val):
    return (df[col] > val).sum() + 1

def big_display_eval(label, val, bg_color):
    val_str = str(val).strip()
    f_color = "#FF0000" if val_str in ['S', 'Ｓ'] else "#0000FF" if val_str in ['A', 'Ａ'] else "#333"
    st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #ddd; height: 140px;">
            <p style="margin: 0; font-size: 14px; font-weight: bold;">{label}</p>
            <p style="margin: 5px 0 0 0; font-size: 48px; font-weight: bold; color: {f_color}; line-height: 1;">{val_str}</p>
        </div>
    """, unsafe_allow_html=True)



# --- メイン処理 ---
try:
    # 1. データの読み込み
    df = load_data()
    df_master = load_machine_master()
    df_price = load_price_data()
    maker_summary, pref_summary = load_summaries()

    # 2. モード判定（最上部で行う）
    qp = st.query_params

    if "pcode" in qp:
        st.session_state["main_mode"] = "🎰 機種から探す"
    elif "id" in qp:
        st.session_state["main_mode"] = "🏢 店舗から探す"
    elif "main_mode" not in st.session_state:
        st.session_state["main_mode"] = "🏢 店舗から探す"

    # 3. サイドバーメニュー（1箇所に集約）
    st.sidebar.title("🛠️ 分析メニュー")
    app_mode = st.sidebar.radio(
        "分析対象を選択", 
        ["🏢 店舗から探す", "🎰 機種から探す"], 
        key="main_mode"
    )

    if df is not None:
        # ---------------------------------------------------------
        # 【A】店舗分析モード
        # ---------------------------------------------------------
        if app_mode == "🏢 店舗から探す":
            total_stores = len(df)
            if "id" in qp:
                # --- 詳細画面モード ---
                sid = int(qp["id"])
                store = df[df["店舗ID"] == sid].iloc[0]
                st.header(f"🏢 {store['店舗名']}")
                st.link_button("🔙 検索一覧に戻る", "/")
                
                tabs = st.tabs(["基本・構成・稼働", "🎰 設置機種詳細", "📍 商圏マップ", "⚔️ 競合比較", "📈 スペック分析", "🆕 新台評価", "💰 資産価値", "📱 スマート機","🏢 メーカー別"])

                with tabs[0]:
                    c1, c2, c3 = st.columns([1, 1, 1.2])
                    with c1:
                        st.subheader("📍 基本情報")
                        st.write(f"**都道府県**: {store['都道府県名']} (ID:{int(store['都道府県ID'])})")
                        st.write(f"**グループ**: {store['グループ名']}")
                        st.write(f"**住所**: {store['住所']}")
                        st.write(f"**総台数**: {int(store['総台数'])}台 (P:{int(store['P台数'])} / S:{int(store['S台数'])})")
                    with c2:
                        st.subheader("📊 設置構成比")
                        st.write(f"・P通常: {format_pct(store.get('機種別パチンコ通常設置比率', 0))}")
                        st.write(f"・P低貸: {format_pct(store.get('機種別パチンコ低貸設置比率', 0))}")
                        st.write(f"・S通常: {format_pct(store.get('機種別スロット通常設置比率', 0))}")
                        st.write(f"・S低貸: {format_pct(store.get('機種別スロット低貸設置比率', 0))}")
                    with c3:
                        st.subheader(f"📈 稼働率順位 (全{total_stores}店中)")
                        mk1, mk2 = st.columns(2)
                        v1=store.get("パチンコ通常稼働率",0); r1=get_rank(df,"パチンコ通常稼働率",v1)
                        v3=store.get("スロット通常稼働率",0); r3=get_rank(df,"スロット通常稼働率",v3)
                        mk1.metric("P通常", format_pct(v1), f"{r1}位")
                        mk2.metric("S通常", format_pct(v3), f"{r3}位")
                        mk1.write(f"P低貸: {format_pct(store.get('パチンコ低貸稼働率', 0))}")
                        mk2.write(f"S低貸: {format_pct(store.get('スロット低貸稼働率', 0))}")

                with tabs[1]: # 🎰 設置機種詳細
                    st.subheader("📋 機種別設置詳細")
                    
                    with st.spinner("データを集計中..."):
                        inv = get_store_inventory(sid, df_master)
                    
                    if not inv.empty:
                        # --- 統計データの準備 ---
                        df_pref_reset = pref_summary.reset_index()
                        drop_cols = [c for c in ['都道府県名', '合計', 'index'] if c in df_pref_reset.columns]
                        total_stats = df_pref_reset.drop(columns=drop_cols).sum()

                        pref_name = store['都道府県名']
                        pref_row = df_pref_reset[df_pref_reset['都道府県名'] == pref_name]
                        
                        if not pref_row.empty:
                            pref_stats = pref_row.iloc[0]
                        else:
                            st.warning(f"都道府県 '{pref_name}' の統計データが見つかりません。")
                            pref_stats = pd.Series(0, index=total_stats.index)

                        # --- 表示用ラベル定義 (ここを統一しました) ---
                        # inv内の名称 : pref_summary内の列名
                        category_map = {
                            "通常P": "通常", 
                            "低貸P": "低貸", 
                            "通常S": "通常", 
                            "低貸S": "低貸"
                        }
                        
                        summ = inv.groupby('貸区分')['設置台数'].sum()
                        
                        # --- 上部メトリクス表示 ---
                        m1, m2, m3, m4 = st.columns(4)
                        cols = [m1, m2, m3, m4]
                        
                        for idx, (inv_label, stat_key) in enumerate(category_map.items()):
                            with cols[idx]:
                                store_val = int(summ.get(inv_label, 0))
                                pref_val = int(pref_stats.get(stat_key, 0))
                                total_val = int(total_stats.get(stat_key, 0))
                                
                                st.markdown(f"""
                                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 5px solid #3498db;">
                                    <p style="margin:0; font-size: 12px; color: #666;">{inv_label}</p>
                                    <p style="margin:0; font-size: 20px; font-weight: bold;">{store_val:,} <span style="font-size: 12px;">台</span></p>
                                    <hr style="margin: 5px 0;">
                                    <p style="margin:0; font-size: 11px; color: #888;">県内計: {pref_val:,}台</p>
                                    <p style="margin:0; font-size: 11px; color: #888;">全国計: {total_val:,}台</p>
                                </div>
                                """, unsafe_allow_html=True)

                        st.divider()
                        
                        sub_tabs = st.tabs(["🔴 通常P", "🟢 低貸P", "🔵 通常S", "🟡 低貸S"])
                        
                        display_cols = ['貸玉量','機種名','設置台数','県内台数','全国台数','メーカー','発売日']
                        
                        with st.spinner("機種別統計を計算中..."):
                            # maker_summaryにpcodeがあるか確認し、なければインデックスから復元
                            ms_temp = maker_summary.reset_index()
                            
                            # 1. 全国集計 (機種コード × 貸区分 ごと)
                            # 列名が 'pcode' かどうかチェック
                            p_col = 'pcode' if 'pcode' in ms_temp.columns else ms_temp.columns[0] 
                            
                            total_machine_sum = ms_temp.groupby([p_col, '貸区分'])['台数'].sum().reset_index()
                            total_machine_sum.columns = ['pcode', '統計用区分', '全国台数']
                            
                            # 2. 都道府県集計
                            df_pref_map = df[['店舗ID', '都道府県名']]
                            maker_with_pref = pd.merge(ms_temp, df_pref_map, on='店舗ID', how='left')
                            pref_machine_sum = maker_with_pref[maker_with_pref['都道府県名'] == pref_name].groupby([p_col, '貸区分'])['台数'].sum().reset_index()
                            pref_machine_sum.columns = ['pcode', '統計用区分', '県内台数']

                        # category_map のキー（通常P, 低貸P...）を順番に処理
                        for i, inv_label in enumerate(category_map.keys()):
                            with sub_tabs[i]:
                                # inv側のpcodeと型を合わせる（文字列型に統一）
                                df_target = inv[inv['貸区分'] == inv_label].copy()
                                df_target['pcode'] = df_target['pcode'].astype(str)
                                
                                if not df_target.empty:
                                    stat_key = category_map[inv_label]
                                    
                                    # 統計側のpcodeも文字列型にしてからマージ
                                    t_sum = total_machine_sum[total_machine_sum['統計用区分'] == stat_key].copy()
                                    t_sum['pcode'] = t_sum['pcode'].astype(str)
                                    
                                    p_sum = pref_machine_sum[pref_machine_sum['統計用区分'] == stat_key].copy()
                                    p_sum['pcode'] = p_sum['pcode'].astype(str)
                                    
                                    # マージ実行
                                    df_target = pd.merge(df_target, t_sum[['pcode', '全国台数']], on='pcode', how='left')
                                    df_target = pd.merge(df_target, p_sum[['pcode', '県内台数']], on='pcode', how='left')
                                    
                                    df_target['全国台数'] = df_target['全国台数'].fillna(0).astype(int)
                                    df_target['県内台数'] = df_target['県内台数'].fillna(0).astype(int)

                                    df_target = df_target.sort_values('設置台数', ascending=False)
                                    
                                    # シェア表示
                                    denominator = int(pref_stats.get(stat_key, 1))
                                    share_in_pref = (int(summ.get(inv_label, 0)) / (denominator if denominator > 0 else 1)) * 100
                                    st.caption(f"💡 {pref_name}内の{inv_label}合計に対し、この店舗は約 {share_in_pref:.2f}% を占めています。")
                                    
                                    actual_cols = [c for c in display_cols if c in df_target.columns]
                                    st.dataframe(df_target[actual_cols], use_container_width=True, hide_index=True)
                                else:
                                    st.info(f"{inv_label} の設置データはありません") 

                       
                    else:
                        st.warning("設置データが見つかりませんでした。")


                with tabs[2]: # 📍 商圏マップ
                    st.subheader("📍 半径3.0km圏内マップ")
                    lat, lon = store['latitude'], store['longitude']
                    if not pd.isna(lat):
                        df_dist = df.copy()
                        df_dist['dist'] = df_dist.apply(lambda r: calculate_distance(lat, lon, r['latitude'], r['longitude']), axis=1)
                        nearby = df_dist[df_dist['dist'] <= 3.0].copy()
                        nearby['color'] = nearby.apply(lambda x: [255,0,0,200] if x['店舗ID']==sid else [0,100,255,160], axis=1)
                        st.pydeck_chart(pdk.Deck(initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=13),
                            layers=[pdk.Layer("ScatterplotLayer", nearby, get_position='[longitude, latitude]', get_color='color', get_radius=100, pickable=True)],
                            tooltip={"text": "{店舗名}\n距離: {dist:.2f} km"}))
                        st.dataframe(nearby[nearby['店舗ID']!=sid][['店舗名','グループ名','総台数','dist']].sort_values('dist'), use_container_width=True)

                with tabs[3]: # ⚔️ 競合比較
                    st.subheader("⚔️ 周辺3.0km圏内 競合店比較（距離順）")
                    lat, lon = store['latitude'], store['longitude']
                    
                    if not pd.isna(lat):
                        df_dist = df.copy()
                        df_dist['dist'] = df_dist.apply(lambda r: calculate_distance(lat, lon, r['latitude'], r['longitude']), axis=1)
                        nearby_stores = df_dist[df_dist['dist'] <= 3.0].sort_values('dist').copy()
                        nearby_ids = nearby_stores['店舗ID'].tolist()
                        
                        target_store_name = store['店舗名']
                        other_store_names = [n for n in nearby_stores['店舗名'].tolist() if n != target_store_name]
                        ordered_columns = [target_store_name] + other_store_names

                        with st.spinner("商圏内・全国統計データを集計中..."):
                            # 1. 必要な列に絞って読み込み
                            df_all_inv = pd.read_csv("P_dai_260408.csv", encoding="cp932", 
                                                     usecols=['店舗ID', 'pcode', '貸玉量', '設置台数'])
                            
                            # 2. 商圏内の店舗データ抽出
                            df_comp = df_all_inv[df_all_inv['店舗ID'].isin(nearby_ids)].copy()
                            df_comp['pcode'] = df_comp['pcode'].astype(str)
                            
                            # 3. マスタ・価格・店舗名の紐付け
                            df_comp = pd.merge(df_comp, df_master, on='pcode', how='left')
                            if not df_price.empty:
                                df_comp = pd.merge(df_comp, df_price, on='pcode', how='left')
                            else:
                                df_comp['中古価格'] = 0
                            df_comp = pd.merge(df_comp, nearby_stores[['店舗ID', '店舗名']], on='店舗ID', how='left')

                            # 4. 貸区分判定 (通常/低貸)
                            df_comp['貸区分'] = "不明"
                            kbn = df_comp['区分']
                            amt = pd.to_numeric(df_comp['貸玉量'], errors='coerce').fillna(0)
                            df_comp.loc[(kbn == 1) & (amt >= 4.0), '貸区分'] = "通常P"
                            df_comp.loc[(kbn == 1) & (amt < 4.0), '貸区分'] = "低貸P"
                            df_comp.loc[(kbn == 2) & (amt >= 10.0), '貸区分'] = "通常S"
                            df_comp.loc[(kbn == 2) & (amt < 10.0), '貸区分'] = "低貸S"

                        # 5. 表示タブ
                        comp_tabs = st.tabs(["🔴 通常P比較", "🟢 低貸P比較", "🔵 通常S比較", "🟡 低貸S比較"])
                        labels = ["通常P", "低貸P", "通常S", "低貸S"]
                        
                        for i, label in enumerate(labels):
                            with comp_tabs[i]:
                                df_target = df_comp[df_comp['貸区分'] == label].copy()
                                if not df_target.empty:
                                    # --- 全国・県内統計の紐付け ---
                                    stat_key = category_map[label] # category_mapを利用
                                    
                                    # 全国台数マージ
                                    t_sum = total_machine_sum[total_machine_sum['統計用区分'] == stat_key][['pcode', '全国台数']].copy()
                                    t_sum['pcode'] = t_sum['pcode'].astype(str)
                                    df_target = pd.merge(df_target, t_sum, on='pcode', how='left')
                                    
                                    # 県内台数マージ
                                    p_sum = pref_machine_sum[pref_machine_sum['統計用区分'] == stat_key][['pcode', '県内台数']].copy()
                                    p_sum['pcode'] = p_sum['pcode'].astype(str)
                                    df_target = pd.merge(df_target, p_sum, on='pcode', how='left')

                                    df_target['全国台数'] = df_target['全国台_sum' if '全国台_sum' in df_target.columns else '全国台数'].fillna(0).astype(int)
                                    df_target['県内台数'] = df_target['県内台_sum' if '県内台_sum' in df_target.columns else '県内台数'].fillna(0).astype(int)

                                    # ピボットテーブル作成（統計情報をIndexに含める）
                                    pivot = df_target.pivot_table(
                                        index=['メーカー', '機種名', '発売日', '中古価格', '県内台数', '全国台数'], 
                                        columns='店舗名', 
                                        values='設置台数', 
                                        aggfunc='sum', 
                                        fill_value=0
                                    )
                                    
                                    pivot['商圏合計'] = pivot.sum(axis=1)
                                    
                                    # 列の並び替え
                                    available_cols = [c for c in ordered_columns if c in pivot.columns]
                                    # 商圏合計を一番左、その後に各店舗を並べる
                                    pivot = pivot[['商圏合計'] + available_cols].sort_values('商圏合計', ascending=False)
                                    
                                    st.dataframe(pivot, use_container_width=True)
 

                                   # --- ⚔️ ここから「エリア内未設置機種」の抽出 ---
                                    st.write("---")
                                    # パチンコ(P)かスロット(S)かを判定
                                    is_slot = "S" in label
                                    ptype_label = "スロット" if is_slot else "パチンコ"
                                    st.subheader(f"🔍 {label}：商圏内未設置の有望機種（全国100台以上）")
                                    
                                    # 1. 全国統計から「貸区分」で絞り込み
                                    stat_key = category_map[label]
                                    t_sum_sub = total_machine_sum[total_machine_sum['統計用区分'] == stat_key].copy()
                                    p_sum_sub = pref_machine_sum[pref_machine_sum['統計用区分'] == stat_key].copy()
                                    
                                    # 2. 商圏内の設置済みpcodeを取得
                                    installed_pcodes = df_target['pcode'].unique()
                                    
                                    # 3. 未設置機種を抽出
                                    df_uninstalled = t_sum_sub[~t_sum_sub['pcode'].isin(installed_pcodes)].copy()
                                    
                                    # 4. マスタ情報を紐付け
                                    df_uninstalled = pd.merge(df_uninstalled, df_master, on='pcode', how='left')
                                    
                                    # --- 重要：パチンコ/スロットの区分一致を強制 ---
                                    target_kbn = 2 if is_slot else 1
                                    df_uninstalled = df_uninstalled[df_uninstalled['区分'] == target_kbn]
                                    
                                    # 5. 価格と県内統計の紐付け
                                    if not df_price.empty:
                                        df_uninstalled = pd.merge(df_uninstalled, df_price[['pcode', '中古価格']], on='pcode', how='left')
                                    else:
                                        df_uninstalled['中古価格'] = 0
                                    
                                    df_uninstalled = pd.merge(df_uninstalled, p_sum_sub[['pcode', '県内台数']], on='pcode', how='left')
                                    
                                    # 6. 全国100台以上の条件
                                    df_uninstalled = df_uninstalled[df_uninstalled['全国台数'] >= 100].copy()
                                    
                                    # データ成形
                                    df_uninstalled['全国台数'] = df_uninstalled['全国台数'].fillna(0).astype(int)
                                    df_uninstalled['県内台数'] = df_uninstalled['県内台数'].fillna(0).astype(int)
                                    df_uninstalled['中古価格'] = df_uninstalled['中古価格'].fillna(0).astype(int)

                                    if not df_uninstalled.empty:
                                        # --- 💡 AIの一押し選定ロジック（1万円以上〜10万円前後） ---
                                        # 1. まず10,000円未満の機種を除外（ノイズ除去）
                                        df_recommend = df_uninstalled[df_uninstalled['中古価格'] >= 10000].copy()
                                        
                                        if not df_recommend.empty:
                                            # 2. 価格スコアリング：100,000円に近いほど高評価
                                            # 10万との差分を計算
                                            df_recommend['price_diff'] = abs(df_recommend['中古価格'] - 100000)
                                            max_diff = df_recommend['price_diff'].max() if df_recommend['price_diff'].max() > 0 else 1
                                            # 10万に近いほど高得点（60点満点）
                                            df_recommend['price_score'] = (1 - (df_recommend['price_diff'] / max_diff)) * 60
                                            
                                            # 3. 実績スコアリング：全国台数が多いほど加点（40点満点）
                                            max_nat = df_recommend['全国台数'].max() if df_recommend['全国台数'].max() > 0 else 1
                                            df_recommend['nat_score'] = (df_recommend['全国台数'] / max_nat) * 40
                                            
                                            # 総合スコアでソート
                                            df_recommend['score'] = df_recommend['price_score'] + df_recommend['nat_score']
                                            df_recommend = df_recommend.sort_values('score', ascending=False)
                                            
                                            # 最上位を一押しとして取得
                                            best_pick = df_recommend.iloc[0]
                                            
                                            st.success(f"👑 **AI一押しの導入候補（値頃感重視）： {best_pick['機種名']}**")
                                            st.write(f"（理由：中古価格 ¥{int(best_pick['中古価格']):,} と導入しやすく、全国で {int(best_pick['全国台数']):,}台 稼働している実績機です。エリア内の空白を埋める戦略的な1台として提案します）")
                                        else:
                                            st.info("価格条件（1万円以上）に合致する未設置機種が見つかりませんでした。")

                                        # --- テーブル表示（全体リストは全国台数順など、元の見せ方を維持） ---
                                        st.caption(f"💡 {label}（{ptype_label}）として全国に100台以上設置されているが、このエリアの競合店には導入されていない機種です。")
                                        
                                        target_cols = ['機種名', 'メーカー', '発売日', '全国台数', '県内台数', '中古価格']
                                        actual_show_cols = [c for c in target_cols if c in df_uninstalled.columns]
                                        
                                        st.dataframe(
                                            df_uninstalled[actual_show_cols].sort_values('全国台数', ascending=False), 
                                            use_container_width=True, 
                                            hide_index=True,
                                            column_config={
                                                "中古価格": st.column_config.NumberColumn("中古価格", format="¥%d"),
                                                "全国台数": st.column_config.NumberColumn("全国台数", format="%d 台"),
                                                "県内台数": st.column_config.NumberColumn("県内台数", format="%d 台"),
                                                "発売日": st.column_config.DateColumn("発売日", format="YYYY/MM/DD")
                                            }
                                        )

                                    else:
                                        st.info(f"条件に該当する未設置の{ptype_label}機種はありません。") 

                                else:
                                    st.info(f"{label} の設置データなし")
                    else:
                        st.error("自店舗の緯度・経度情報がないため、競合比較ができません。")


                with tabs[4]: # 📈 スペック分析
                    st.subheader("📈 スペック/タイプ内訳分析")
                    
                    # 貸玉選択
                    view_mode = st.radio("表示対象を選択", ["全体", "通常貸玉", "低貸玉"], horizontal=True)

                    # --- インデックス設定 ---
                    if view_mode == "全体":
                        p_indices = [99, 101, 103, 105, 107, 109, 111, 113, 115, 117]
                        s_indices = [171, 173, 175, 177, 179]
                    elif view_mode == "通常貸玉":
                        p_indices = [119, 121, 123, 125, 127, 129, 131, 133, 135, 137]
                        s_indices = [181, 183, 185, 187, 189]
                    else: # 低貸玉
                        p_indices = [139, 141, 143, 145, 147, 149, 151, 153, 155, 157]
                        s_indices = [191, 193, 195, 197, 199]

                    # 共通のラベル
                    p_labels = ["1/380～", "1/320～", "1/300～", "1/250～", "1/200～", "1/150～", "1/100～", "～1/99", "ハネモノ", "その他"]
                    s_labels = ["ノーマル", "ATのみ", "ボーナス＋ART", "ボーナス＋RT", "その他"]

                    # 1. 自店の円グラフ表示
                    st.markdown(f"#### 🏠 自店の内訳（{view_mode}）")
                    cp, cs = st.columns(2)
                    with cp:
                        p_specs = {k: store.iloc[v] for k, v in zip(p_labels, p_indices)}
                        pdf = pd.DataFrame(list(p_specs.items()), columns=['スペック', '値'])
                        pdf['比率'] = pd.to_numeric(pdf['値'], errors='coerce').fillna(0).apply(lambda x: x*100 if 0 < x <= 1 else x)
                        if pdf['比率'].sum() > 0:
                            st.plotly_chart(px.pie(pdf[pdf['比率']>0], values='比率', names='スペック', hole=0.4, category_orders={"スペック": p_labels}), use_container_width=True)
                    with cs:
                        s_specs = {k: store.iloc[v] for k, v in zip(s_labels, s_indices)}
                        sdf = pd.DataFrame(list(s_specs.items()), columns=['タイプ', '値'])
                        sdf['比率'] = pd.to_numeric(sdf['値'], errors='coerce').fillna(0).apply(lambda x: x*100 if 0 < x <= 1 else x)
                        if sdf['比率'].sum() > 0:
                            st.plotly_chart(px.pie(sdf[sdf['比率']>0], values='比率', names='タイプ', hole=0.4, category_orders={"タイプ": s_labels}), use_container_width=True)

                    st.divider()

                    # 2. 3.0km圏内の一覧表示
                    st.subheader(f"📍 3.0km圏内 店舗別スペック比較（{view_mode}）")
                    
                    try:
                        df_calc = df.copy()
                        df_calc.iloc[:, 16] = pd.to_numeric(df_calc.iloc[:, 16], errors='coerce')
                        df_calc.iloc[:, 17] = pd.to_numeric(df_calc.iloc[:, 17], errors='coerce')
                        df_calc = df_calc.dropna(subset=[df_calc.columns[16], df_calc.columns[17]])
                        
                        base_lat = float(pd.to_numeric(store.iloc[16], errors='coerce'))
                        base_lon = float(pd.to_numeric(store.iloc[17], errors='coerce'))
                        
                        df_calc['距離'] = df_calc.apply(lambda row: calculate_distance(base_lat, base_lon, row.iloc[16], row.iloc[17]), axis=1)
                        area_stores = df_calc[df_calc['距離'] <= 3.0].sort_values('距離').copy()

                        if not area_stores.empty:
                            # --- パチンコ一覧 ---
                            st.markdown("##### 【パチンコ】スペック別構成比率")
                            p_display = area_stores.iloc[:, [1, 4] + p_indices].copy()
                            p_display.insert(0, '距離(km)', area_stores['距離'])
                            p_display.columns = ['距離(km)', '店舗名', 'グループ名'] + p_labels
                            
                            for col in p_labels:
                                p_display[col] = pd.to_numeric(p_display[col], errors='coerce').fillna(0)
                                p_display[col] = p_display[col].apply(lambda x: x*100 if 0 < x <= 1 else x)

                            st.dataframe(
                                p_display, 
                                use_container_width=True, 
                                hide_index=True,
                                column_config={
                                    "距離(km)": st.column_config.NumberColumn("距離", format="%.2f km"),
                                    **{col: st.column_config.NumberColumn(col, format="%.2f %%") for col in p_labels}
                                }
                            )

                            st.write("")
                            # --- スロット一覧 ---
                            st.markdown("##### 【スロット】タイプ別構成比率")
                            s_display = area_stores.iloc[:, [1, 4] + s_indices].copy()
                            s_display.insert(0, '距離(km)', area_stores['距離'])
                            s_display.columns = ['距離(km)', '店舗名', 'グループ名'] + s_labels
                            
                            for col in s_labels:
                                s_display[col] = pd.to_numeric(s_display[col], errors='coerce').fillna(0)
                                s_display[col] = s_display[col].apply(lambda x: x*100 if 0 < x <= 1 else x)

                            st.dataframe(
                                s_display, 
                                use_container_width=True, 
                                hide_index=True,
                                column_config={
                                    "距離(km)": st.column_config.NumberColumn("距離", format="%.2f km"),
                                    **{col: st.column_config.NumberColumn(col, format="%.2f %%") for col in s_labels}
                                }
                            )
                        else:
                            # ここがズレていた可能性があります
                            st.info("3.0km圏内に競合店舗は見つかりませんでした。")
                            
                    except Exception as e:
                        st.error(f"スペック比較データの取得に失敗しました: {e}")

                with tabs[5]: # 🆕 新台評価
                    st.subheader("🆕 新台導入・評価")
                    
                    # --- 上部：自店の評価メトリクス ---
                    e1, e2 = st.columns(2)
                    # BN:65, BO:66
                    with e1: big_display_eval("P新台評価", store.iloc[65], "#FFF0F5")
                    with e2: big_display_eval("S新台評価", store.iloc[66], "#F0F8FF")
                    
                    st.divider()
                    
                    m1, m2, m3 = st.columns(3)
                    # BS:70, BU:72
                    pn = pd.to_numeric(store.iloc[70], errors='coerce') if not pd.isna(store.iloc[70]) else 0
                    sn = pd.to_numeric(store.iloc[72], errors='coerce') if not pd.isna(store.iloc[72]) else 0
                    m1.metric("年間新台導入合計", f"{int(pn+sn)}台")
                    m2.metric("年間P新台導入", f"{int(pn)}台")
                    m3.metric("年間S新台導入", f"{int(sn)}台")

                    # --- 下部：商圏内競合店の評価一覧 ---
                    st.write("")
                    st.subheader("📍 3.0km圏内 店舗別新台評価")
                    
                    try:
                        # 1. 緯度・経度列を数値に変換（Q=16, R=17）
                        df_calc = df.copy()
                        df_calc.iloc[:, 16] = pd.to_numeric(df_calc.iloc[:, 16], errors='coerce')
                        df_calc.iloc[:, 17] = pd.to_numeric(df_calc.iloc[:, 17], errors='coerce')
                        
                        # 2. 座標がない店舗を除外
                        df_calc = df_calc.dropna(subset=[df_calc.columns[16], df_calc.columns[17]])
                        
                        # 3. 対象店舗の座標
                        base_lat = float(pd.to_numeric(store.iloc[16], errors='coerce'))
                        base_lon = float(pd.to_numeric(store.iloc[17], errors='coerce'))
                        
                        # 4. 距離計算
                        df_calc['距離'] = df_calc.apply(
                            lambda row: calculate_distance(base_lat, base_lon, row.iloc[16], row.iloc[17]), 
                            axis=1
                        )
                        
                        # 5. 3.0km以内でフィルタリング
                        area_stores = df_calc[df_calc['距離'] <= 3.0].sort_values('距離').copy()

                        if not area_stores.empty:
                            # 抽出列: B:1, E:4, BN:65, BO:66, BQ:68, BS:70, BU:72, U:20, S:18, T:19
                            display_df = area_stores.iloc[:, [1, 4, 65, 66, 68, 70, 72, 20, 18, 19]].copy()
                            
                            # --- 各種導入率の算出 ---
                            # 全体導入率 (BQ:68 / U:20)
                            num_bq = pd.to_numeric(display_df.iloc[:, 4], errors='coerce').fillna(0)
                            num_u = pd.to_numeric(display_df.iloc[:, 7], errors='coerce').fillna(0)
                            display_df['導入率'] = (num_bq / num_u.replace(0, np.nan)) * 100
                            
                            # P導入率 (BS:70 / S:18)
                            num_bs = pd.to_numeric(display_df.iloc[:, 5], errors='coerce').fillna(0)
                            num_s = pd.to_numeric(display_df.iloc[:, 8], errors='coerce').fillna(0)
                            display_df['P導入率'] = (num_bs / num_s.replace(0, np.nan)) * 100
                            
                            # S導入率 (BU:72 / T:19)
                            num_bu = pd.to_numeric(display_df.iloc[:, 6], errors='coerce').fillna(0)
                            num_t = pd.to_numeric(display_df.iloc[:, 9], errors='coerce').fillna(0)
                            display_df['S導入率'] = (num_bu / num_t.replace(0, np.nan)) * 100
                            
                            # 距離列を挿入
                            display_df.insert(0, '距離(km)', area_stores['距離'])
                            
                            # --- 列の並び替え ---
                            # 距離, 店名, グループ, P評, S評, 合計, 導入率, P導入, P導入率, S導入, S導入率
                            final_cols = [
                                display_df.columns[0], display_df.columns[1], display_df.columns[2],
                                display_df.columns[3], display_df.columns[4], display_df.columns[5],
                                '導入率', display_df.columns[6], 'P導入率', display_df.columns[7], 'S導入率'
                            ]
                            display_df = display_df[final_cols]
                            
                            display_df.columns = [
                                '距離(km)', '店舗名', 'グループ名', 'P評価', 'S評価', 
                                '合計導入', '導入率', 'P導入', 'P導入率', 'S導入', 'S導入率'
                            ]

                            # --- データ型とクレンジング ---
                            # 数値列（整数）
                            for col in ['合計導入', 'P導入', 'S導入']:
                                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0).astype(int)
                            
                            # 数値列（小数）
                            for col in ['導入率', 'P導入率', 'S導入率']:
                                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0)

                            # 文字列列
                            for col in ['P評価', 'S評価', '店舗名', 'グループ名']:
                                display_df[col] = display_df[col].astype(str).replace('nan', '-')

                            # --- スタイリング関数の定義 ---
                            def color_evaluation(val):
                                # 全角・半角どちらでも対応できるように判定
                                if val in ['Ｓ', 'S']:
                                    return 'color: red; font-weight: bold;'
                                elif val in ['Ａ', 'A']:
                                    return 'color: blue;'
                                return ''

                            # 文字列列のクレンジング（再掲）
                            for col in ['P評価', 'S評価', '店舗名', 'グループ名']:
                                display_df[col] = display_df[col].astype(str).replace('nan', '-')

                            # スタイルの適用
                            styled_df = display_df.style.map(color_evaluation, subset=['P評価', 'S評価'])

                            # 6. テーブル表示 (st.dataframe の代わりに styled_df を渡す)
                            st.dataframe(
                                styled_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "距離(km)": st.column_config.NumberColumn("距離", format="%.2f km"),
                                    "P評価": st.column_config.TextColumn("P評"),
                                    "S評価": st.column_config.TextColumn("S評"),
                                    "合計導入": st.column_config.NumberColumn("合計"),
                                    "導入率": st.column_config.NumberColumn("導入率", format="%.2f %%"),
                                    "P導入": st.column_config.NumberColumn("P導入"),
                                    "P導入率": st.column_config.NumberColumn("P導入率", format="%.2f %%"),
                                    "S導入": st.column_config.NumberColumn("S導入"),
                                    "S導入率": st.column_config.NumberColumn("S導入率", format="%.2f %%"),
                                }
                            )

                        else:
                            st.info("3.0km圏内に競合店舗は見つかりませんでした。")
                            
                    except Exception as e:
                        st.error(f"近隣店舗データの取得に失敗しました: {e}")

                with tabs[6]: # 💰 資産価値
                    st.subheader(f"💰 資産価値順位 (全{total_stores}店中)")
                    # --- 上部：自店の資産価値（従来通り、または適宜修正） ---
                    vt=store.iloc[79]; rt=get_rank(df,df.columns[79],vt)
                    st.metric("店舗全体 資産価値", format_yen(vt), f"{rt}位")
                    st.divider()
                    c1, c2 = st.columns(2)
                    with c1:
                        vp=store.iloc[80]; rp=get_rank(df,df.columns[80],vp)
                        st.metric("P合計 資産価値", format_yen(vp), f"{rp}位")
                        st.write(f"・通常: {format_yen(store.iloc[82])} / 低貸: {format_yen(store.iloc[83])}")
                    with c2:
                        vs=store.iloc[81]; rs=get_rank(df,df.columns[81],vs)
                        st.metric("S合計 資産価値", format_yen(vs), f"{rs}位")
                        st.write(f"・通常: {format_yen(store.iloc[84])} / 低貸: {format_yen(store.iloc[85])}")


                    # --- 下部：商圏内競合店の資産価値一覧 ---
                    st.subheader("📍 3.0km圏内 店舗別資産価値一覧")
                    
                    try:
                        # 1. 緯度・経度列を数値に変換（Q=16, R=17）
                        df_calc = df.copy()
                        df_calc.iloc[:, 16] = pd.to_numeric(df_calc.iloc[:, 16], errors='coerce')
                        df_calc.iloc[:, 17] = pd.to_numeric(df_calc.iloc[:, 17], errors='coerce')
                        
                        # 2. 座標がない店舗を除外
                        df_calc = df_calc.dropna(subset=[df_calc.columns[16], df_calc.columns[17]])
                        
                        # 3. 対象店舗の座標
                        base_lat = float(pd.to_numeric(store.iloc[16], errors='coerce'))
                        base_lon = float(pd.to_numeric(store.iloc[17], errors='coerce'))
                        
                        # 4. 距離計算
                        df_calc['距離'] = df_calc.apply(
                            lambda row: calculate_distance(base_lat, base_lon, row.iloc[16], row.iloc[17]), 
                            axis=1
                        )
                        
                        # 5. 3.0km以内でフィルタリング
                        area_stores = df_calc[df_calc['距離'] <= 3.0].sort_values('距離').copy()

                        if not area_stores.empty:
                            # 抽出列: B:1, E:4, CB:79, CC:80, CD:81, CE:82, CF:83, CG:84, CH:85
                            display_df = area_stores.iloc[:, [1, 4, 79, 80, 81, 82, 83, 84, 85]].copy()
                            display_df.insert(0, '距離(km)', area_stores['距離'])
                            
                            display_df.columns = [
                                '距離(km)', '店舗名', 'グループ名', 
                                '店舗全体', 'P全体', 'S全体', 'P通常', 'P低貸', 'S通常', 'S低貸'
                            ]

                            # --- 型の最適化とクレンジング ---
                            # 数値列（すべて1円単位の整数として処理）
                            val_cols = ['店舗全体', 'P全体', 'S全体', 'P通常', 'P低貸', 'S通常', 'S低貸']
                            for col in val_cols:
                                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0).astype(int)
                            
                            # 文字列列
                            display_df['店舗名'] = display_df['店舗名'].astype(str).replace('nan', '')
                            display_df['グループ名'] = display_df['グループ名'].astype(str).replace('nan', '')

                            # 6. テーブル表示
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "距離(km)": st.column_config.NumberColumn("距離", format="%.2f km"),
                                    "店舗全体": st.column_config.NumberColumn("店舗全体", format="%,d"),
                                    "P全体": st.column_config.NumberColumn("P全体", format="%,d"),
                                    "S全体": st.column_config.NumberColumn("S全体", format="%,d"),
                                    "P通常": st.column_config.NumberColumn("P通常", format="%,d"),
                                    "P低貸": st.column_config.NumberColumn("P低貸", format="%,d"),
                                    "S通常": st.column_config.NumberColumn("S通常", format="%,d"),
                                    "S低貸": st.column_config.NumberColumn("S低貸", format="%,d"),
                                }
                            )
                        else:
                            st.info("3.0km圏内に競合店舗は見つかりませんでした。")
                            
                    except Exception as e:
                        st.error(f"資産価値データの取得に失敗しました: {e}")


                with tabs[7]: # 📱 スマート機
                    st.subheader(f"📱 スマート遊技機 順位 (全{total_stores}店中)")
                    # --- 上部：自店のスマート機メトリクス（既存のものを維持） ---
                    c1, c2 = st.columns(2)
                    with c1:
                        v_pp=store.iloc[87]; r_pp=get_rank(df,df.columns[87],v_pp)
                        st.metric("スマパチ設置比率", format_pct(v_pp), f"{r_pp}位")
                        st.write(f"・台数: {int(store.iloc[86])}台 (通常:{int(store.iloc[90])} / 低貸:{int(store.iloc[92])})")
                    with c2:
                        v_sp=store.iloc[89]; r_sp=get_rank(df,df.columns[89],v_sp)
                        st.metric("スマスロ設置比率", format_pct(v_sp), f"{r_sp}位")
                        st.write(f"・台数: {int(store.iloc[88])}台 (通常:{int(store.iloc[94])} / 低貸:{int(store.iloc[96])})")

                    # --- 下部：商圏内競合店のスマート機一覧 ---
                    st.subheader("📍 3.0km圏内 店舗別スマート機詳細")
                    
                    try:
                        # 1. 緯度・経度列を数値に変換（Q=16, R=17）
                        df_calc = df.copy()
                        df_calc.iloc[:, 16] = pd.to_numeric(df_calc.iloc[:, 16], errors='coerce')
                        df_calc.iloc[:, 17] = pd.to_numeric(df_calc.iloc[:, 17], errors='coerce')
                        
                        # 2. 座標がない店舗を除外
                        df_calc = df_calc.dropna(subset=[df_calc.columns[16], df_calc.columns[17]])
                        
                        # 3. 対象店舗の座標
                        base_lat = float(pd.to_numeric(store.iloc[16], errors='coerce'))
                        base_lon = float(pd.to_numeric(store.iloc[17], errors='coerce'))
                        
                        # 4. 距離計算
                        df_calc['距離'] = df_calc.apply(
                            lambda row: calculate_distance(base_lat, base_lon, row.iloc[16], row.iloc[17]), 
                            axis=1
                        )
                        
                        # 5. 3.0km以内でフィルタリング
                        area_stores = df_calc[df_calc['距離'] <= 3.0].sort_values('距離').copy()

                        if not area_stores.empty:
                            # 抽出列: B:1, E:4, CI:86, CJ:87, CM:90, CN:91, CO:92, CP:93, CK:88, CL:89, CQ:94, CR:95, CS:96, CT:97
                            display_df = area_stores.iloc[:, [1, 4, 86, 87, 90, 91, 92, 93, 88, 89, 94, 95, 96, 97]].copy()
                            display_df.insert(0, '距離(km)', area_stores['距離'])
                            
                            display_df.columns = [
                                '距離(km)', '店舗名', 'グループ名', 
                                'スマパチ台数', 'スマパチ比率', '通常P台', '通常P比', '低貸P台', '低貸P比',
                                'スマスロ台数', 'スマスロ比率', '通常S台', '通常S比', '低貸S台', '低貸S比'
                            ]

                            # --- 型の最適化 ---
                            # 台数列（整数・カンマ区切り）
                            int_cols = ['スマパチ台数', '通常P台', '低貸P台', 'スマスロ台数', '通常S台', '低貸S台']
                            for col in int_cols:
                                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0).astype(int)
                            
                            # 比率列（データが0.344などの場合、ここで100倍して34.4にする）
                            ratio_cols = ['スマパチ比率', '通常P比', '低貸P比', 'スマスロ比率', '通常S比', '低貸S比']
                            for col in ratio_cols:
                                # 数値に変換
                                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0)
                                # 🆕 ここで100倍する（もし元データが 0.344 の場合用）
                                display_df[col] = display_df[col] * 100

                            # 文字列列
                            display_df['店舗名'] = display_df['店舗名'].astype(str).replace('nan', '')
                            display_df['グループ名'] = display_df['グループ名'].astype(str).replace('nan', '')

                            # 6. テーブル表示
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "距離(km)": st.column_config.NumberColumn("距離", format="%.2f km"),
                                    "スマパチ台数": st.column_config.NumberColumn("スマパチ", format="%,d"),
                                    # 🆕 データ側で100倍したので、formatには % 記号だけを付ける
                                    "スマパチ比率": st.column_config.NumberColumn("P全体％", format="%.2f %%"),
                                    "通常P台": st.column_config.NumberColumn("通常P", format="%,d"),
                                    "通常P比": st.column_config.NumberColumn("通常P％", format="%.2f %%"),
                                    "低貸P台": st.column_config.NumberColumn("低貸P", format="%,d"),
                                    "低貸P比": st.column_config.NumberColumn("低貸P％", format="%.2f %%"),
                                    "スマスロ台数": st.column_config.NumberColumn("スマスロ", format="%,d"),
                                    "スマスロ比率": st.column_config.NumberColumn("S全体％", format="%.2f %%"),
                                    "通常S台": st.column_config.NumberColumn("通常S", format="%,d"),
                                    "通常S比": st.column_config.NumberColumn("通常S％", format="%.2f %%"),
                                    "低貸S台": st.column_config.NumberColumn("低貸S", format="%,d"),
                                    "低貸S比": st.column_config.NumberColumn("低貸S％", format="%.2f %%"),
                                }
                            )  
                        else:
                            st.info("3.0km圏内に競合店舗は見つかりませんでした。")
                            
                    except Exception as e:
                        st.error(f"スマート機データの取得に失敗しました: {e}")


                with tabs[8]: # 🏢 メーカー別
                    st.subheader("🏢 メーカー別設置シェア分析")
                    
                    # 1. 機種マスタのロード
                    df_m = load_machine_master()
                    
                    # 2. 設置詳細データの確保
                    p_dai_source = None
                    if 'P_dai_260408' in st.session_state:
                        p_dai_source = st.session_state['P_dai_260408']
                    elif 'P_dai_260408' in globals():
                        p_dai_source = globals()['P_dai_260408']
                    
                    if p_dai_source is None:
                        try:
                            p_dai_source = pd.read_csv("P_dai_260408.csv", encoding="cp932")
                            p_dai_source.columns = p_dai_source.columns.str.strip()
                        except:
                            pass

                    # 3. チェック
                    if (not df_m.empty) and (p_dai_source is not None):
                        try:
                            # --- 貸玉別の表示設定 ---
                            col_sel1, col_sel2 = st.columns([1, 2])
                            with col_sel1:
                                kashi_type = st.radio("表示対象を選択", ["すべて", "通常貸のみ", "低貸のみ"], horizontal=True)

                            # --- 3.0km圏内の店舗を特定 ---
                            df_calc = df.copy()
                            df_calc.iloc[:, 16] = pd.to_numeric(df_calc.iloc[:, 16], errors='coerce')
                            df_calc.iloc[:, 17] = pd.to_numeric(df_calc.iloc[:, 17], errors='coerce')
                            df_calc = df_calc.dropna(subset=[df_calc.columns[16], df_calc.columns[17]])
                            
                            base_lat = float(pd.to_numeric(store.iloc[16], errors='coerce'))
                            base_lon = float(pd.to_numeric(store.iloc[17], errors='coerce'))
                            df_calc['距離'] = df_calc.apply(lambda row: calculate_distance(base_lat, base_lon, row.iloc[16], row.iloc[17]), axis=1)
                            
                            area_stores = df_calc[df_calc['距離'] <= 3.0].sort_values('距離').copy()
                            area_store_ids = area_stores.iloc[:, 0].unique()

                            # --- 設置データとマスタの結合 ---
                            p_dai_calc = p_dai_source.copy()
                            p_dai_calc['pcode_str'] = p_dai_calc.iloc[:, 1].astype(str).str.strip()
                            
                            df_merged = pd.merge(
                                p_dai_calc,
                                df_m[['pcode', 'メーカー', '区分']],
                                left_on='pcode_str',
                                right_on='pcode',
                                how='left'
                            )
                            df_merged['台数'] = pd.to_numeric(df_merged.iloc[:, 3], errors='coerce').fillna(0)
                            df_merged = df_merged.rename(columns={df_merged.columns[0]: '店舗ID'})

                            # 貸区分判定ロジック（C列=index 2 の貸玉量を使用）
                            def judge_kashi_simple(row):
                                try:
                                    amt = float(row.iloc[2])
                                    ps_type = row['区分']
                                    if ps_type == 1: return "通常" if amt >= 4.0 else "低貸"
                                    if ps_type == 2: return "通常" if amt >= 10.0 else "低貸"
                                except: pass
                                return "不明"
                            
                            df_merged['貸区分'] = df_merged.apply(judge_kashi_simple, axis=1)

                            # フィルタリング
                            if kashi_type == "通常貸のみ":
                                df_merged = df_merged[df_merged['貸区分'] == "通常"]
                            elif kashi_type == "低貸のみ":
                                df_merged = df_merged[df_merged['貸区分'] == "低貸"]

                            # --- 1. 自店の分析 ---
                            my_id = store.iloc[0]
                            my_data = df_merged[df_merged['店舗ID'] == my_id]
                            
                            st.markdown(f"#### 🏠 自店のメーカー構成 ({kashi_type})")
                            m_cp, m_cs = st.columns(2)
                            
                            with m_cp:
                                p_my = my_data[my_data['区分'] == 1].groupby('メーカー')['台数'].sum().reset_index().sort_values('台数', ascending=False)
                                if not p_my.empty:
                                    st.write("📊 **パチンコ 自店構成**")
                                    st.plotly_chart(px.pie(p_my.head(15), values='台数', names='メーカー', hole=0.4), use_container_width=True)
                                else:
                                    st.caption("該当データなし")
                            
                            with m_cs:
                                s_my = my_data[my_data['区分'] == 2].groupby('メーカー')['台数'].sum().reset_index().sort_values('台数', ascending=False)
                                if not s_my.empty:
                                    st.write("🎰 **スロット 自店構成**")
                                    st.plotly_chart(px.pie(s_my.head(15), values='台数', names='メーカー', hole=0.4), use_container_width=True)
                                else:
                                    st.caption("該当データなし")

                            st.divider()

                            # --- 2. 3.0km圏内 競合比較一覧 ---
                            st.subheader(f"📍 3.0km圏内 店舗別メーカーシェア一覧 ({kashi_type})")
                            area_data = df_merged[df_merged['店舗ID'].isin(area_store_ids)]
                            
                            def get_full_maker_share(target_kbn):
                                sub_area = area_data[area_data['区分'] == target_kbn]
                                if sub_area.empty: return pd.DataFrame(), []
                                
                                pivot_df = sub_area.pivot_table(
                                    index='店舗ID', columns='メーカー', values='台数', aggfunc='sum'
                                ).fillna(0)
                                sorted_makers = pivot_df.sum().sort_values(ascending=False).index.tolist()
                                pivot_df = pivot_df[sorted_makers]
                                
                                info = area_stores.iloc[:, [0, 1, 4]].copy()
                                info.columns = ['店舗ID', '店舗名', 'グループ名']
                                info['距離(km)'] = area_stores['距離']
                                
                                res = pd.merge(info, pivot_df, on='店舗ID', how='inner').fillna(0)
                                row_total = res[sorted_makers].sum(axis=1)
                                for col in sorted_makers:
                                    res[col] = (res[col] / row_total * 100).fillna(0)
                                return res.sort_values('距離(km)'), sorted_makers

                            # パチンコ比較表
                            st.markdown("##### 【パチンコ】全メーカー設置比率")
                            p_compare, p_all_makers = get_full_maker_share(1)
                            if not p_compare.empty:
                                st.dataframe(p_compare.drop(columns=['店舗ID']), use_container_width=True, hide_index=True,
                                             column_config={"距離(km)": st.column_config.NumberColumn("距離", format="%.2f km"),
                                                            **{m: st.column_config.NumberColumn(m, format="%.1f %%") for m in p_all_makers}})
                            else:
                                st.caption("該当データなし")

                            st.write("")

                            # スロット比較表
                            st.markdown("##### 【スロット】全メーカー設置比率")
                            s_compare, s_all_makers = get_full_maker_share(2)
                            if not s_compare.empty:
                                st.dataframe(s_compare.drop(columns=['店舗ID']), use_container_width=True, hide_index=True,
                                             column_config={"距離(km)": st.column_config.NumberColumn("距離", format="%.2f km"),
                                                            **{m: st.column_config.NumberColumn(m, format="%.1f %%") for m in s_all_makers}})
                            else:
                                st.caption("該当データなし")

                        except Exception as e:
                            st.error(f"メーカー集計中にエラーが発生しました: {e}")
                            st.text(traceback.format_exc())
                    else:
                        st.warning("設置詳細データ（P_dai_260408）または機種マスタが読み込まれていません。")



            else:
                # --- 店舗検索画面 ---
                st.title("店舗別データ検索アプリ")
                st.sidebar.header("🔍 絞り込み")
                pref_master = df[['都道府県ID', '都道府県名']].drop_duplicates().sort_values('都道府県ID')
                prefs = ["すべて"] + pref_master["都道府県名"].tolist() 
                sel_pref = st.sidebar.selectbox("都道府県", prefs)
                s_group = st.sidebar.text_input("グループ名検索")
                s_name = st.sidebar.text_input("店舗名検索")
            
                st.sidebar.markdown("---")
                sort_opts = {"標準（設置台数順）": df.columns[20], "稼働率順": df.columns[58], "資産価値順": df.columns[79], "スマパチ比率": df.columns[87], "スマスロ比率": df.columns[89]}
                sel_sort = st.sidebar.selectbox("並び替え", list(sort_opts.keys()))
            
                f_df = df.copy()
                if sel_pref != "すべて": f_df = f_df[f_df["都道府県名"] == sel_pref]
                if s_group: f_df = f_df[f_df["グループ名"].str.contains(s_group, na=False)]
                if s_name: f_df = f_df[f_df["店舗名"].str.contains(s_name, na=False)]
            
                scol = sort_opts[sel_sort]
                f_df = f_df.sort_values(by=scol, ascending=False)
  
                btn_container = st.container()
                disp = f_df.copy()
                for c in [df.columns[58], df.columns[87], df.columns[89]]: disp[c] = disp[c].apply(format_pct)
                disp[df.columns[79]] = disp[df.columns[79]].apply(format_yen)
            
                d_cols = ["都道府県名", "店舗名", "グループ名", "総台数", "住所"]
                if sel_sort != "標準（設置台数順）": d_cols.append(scol)

                sel = st.dataframe(disp[d_cols], on_select="rerun", selection_mode="single-row", use_container_width=True)
              
                if len(sel.selection.rows) > 0:
                    idx = sel.selection.rows[0]
                    sid, sname = f_df.iloc[idx]["店舗ID"], f_df.iloc[idx]["店舗名"]
                    with btn_container:
                        st.success(f"✅ 選択中: {sname}")
                        st.link_button(f"👉 {sname} の詳細分析ページを開く", f"/?id={sid}", type="primary", use_container_width=True)
                else:
                    with btn_container: st.info("💡 分析したい店舗を一覧からクリックしてください。")

        # ---------------------------------------------------------
        # 【B】機種分析モード
        # ---------------------------------------------------------
        elif app_mode == "🎰 機種から探す":

            if "pcode" in qp:
                target_pcode = str(qp["pcode"])
                # 機種マスタに価格データを結合
                df_with_price = pd.merge(df_master, df_price, on='pcode', how='left')
                df_with_price['中古価格'] = df_with_price['中古価格'].fillna(0)
                
                target_rows = df_with_price[df_with_price['pcode'] == target_pcode]
                
                if not target_rows.empty:
                    target_m = target_rows.iloc[0]
                    st.header(f"🎰 {target_m['機種名']}")
                    st.link_button("🔙 機種検索に戻る", "/")
                    
                    m_tabs = st.tabs(["📝 基本情報", "📊 設置店舗ランキング", "💰 詳細スペック"])
                    
                    with m_tabs[0]:
                        st.subheader("📍 機種スペック概要")
                        c_info1, c_info2 = st.columns(2)
                        with c_info1:
                            st.write(f"**機種名**: {target_m['機種名']}")
                            st.write(f"**メーカー**: {target_m['メーカー']}")
                            st.write(f"**発売日**: {target_m['発売日']}")
                            m_type_label = "パチンコ" if target_m['区分'] == 1 else "スロット"
                            st.write(f"**区分**: {m_type_label}")

                        # --- 中古価格と順位の計算 ---
                        st.divider()
                        st.subheader("💰 市場価値")
                        
                        # 同じ区分のデータだけを抽出して順位を計算
                        same_type_df = df_with_price[df_with_price['区分'] == target_m['区分']]
                        price_rank = (same_type_df['中古価格'] > target_m['中古価格']).sum() + 1
                        total_count = len(same_type_df)

                        p_col1, p_col2 = st.columns(2)
                        with p_col1:
                            st.metric("現在の中古価格", format_yen(target_m['中古価格']))
                        with p_col2:
                            st.metric(f"{m_type_label}内 価格順位", f"{price_rank} 位", f"全 {total_count} 機種中")

                        # --- 既存の設置記録（最多設置台数など） ---
                        st.divider()
                        st.subheader("📈 設置記録（ピーク時）")
                        rec1, rec2 = st.columns(2)
                        with rec1:
                            st.metric("最多設置台数", f"{int(target_m['最多設置台数']):,} 台")
                            st.caption(f"記録日: {target_m['最多設置台数記録日']}")
                        with rec2:
                            st.metric("最多設置店舗数", f"{int(target_m['最多設置店舗数']):,} 店")
                            st.caption(f"記録日: {target_m['最多設置店舗記録日']}")


                    # --- タブ1: 設置店舗ランキング ---
                    with m_tabs[1]:
                        st.subheader("🏆 設置台数ランキング")
                        
                        try:
                            # 1. P_dai データの読み込み (A, B, D列のみ)
                            df_pdai = pd.read_csv("P_dai_260408.csv", encoding="cp932", usecols=[0, 1, 3], header=None, skiprows=1)
                            df_pdai.columns = ['店舗ID', 'pcode', '設置台数']
                            
                            # 2. 対象機種でフィルタリング
                            df_target_pdai = df_pdai[df_pdai['pcode'].astype(str) == target_pcode]
                            
                            if not df_target_pdai.empty:
                                # 3. 店舗ごとに台数を合計
                                ranking = df_target_pdai.groupby('店舗ID')['設置台数'].sum().reset_index()
                                
                                # 4. 店舗マスタ(df)から情報を取得
                                # I列(8番目)=都道府県ID, L列(11番目)=都道府県名 を確実に取得
                                # 列番号で指定して一時的なデータフレームを作成
                                store_info = df.iloc[:, [0, 1, 4, 8, 9, 14]].copy() 
                                store_info.columns = ['店舗ID', '店舗名', 'グループ名', '都道府県ID', '都道府県', '住所']
                                
                                # ランキングデータと店舗情報を結合
                                ranking_final = pd.merge(ranking, store_info, on='店舗ID', how='inner')
                                
                                # --- フィルターUI ---
                                st.write("---")
                                f_col1, f_col2 = st.columns(2)
                                
                                with f_col1:
                                    # 都道府県ID順に並べ替え
                                    pref_master = ranking_final[['都道府県ID', '都道府県']].drop_duplicates().sort_values('都道府県ID')
                                    pref_options = ["すべて"] + pref_master['都道府県'].tolist()
                                    sel_pref = st.selectbox("都道府県で絞り込み", pref_options)
                                
                                with f_col2:
                                    # グループ名は自由記入
                                    input_group = st.text_input("グループ名で検索", "")
                                
                                # --- フィルタリング実行 ---
                                if sel_pref != "すべて":
                                    ranking_final = ranking_final[ranking_final['都道府県'] == sel_pref]
                                
                                if input_group:
                                    ranking_final = ranking_final[ranking_final['グループ名'].str.contains(input_group, na=False)]
                                
                                # 5. 台数降順ソート
                                ranking_final = ranking_final.sort_values('設置台数', ascending=False).reset_index(drop=True)
                                ranking_final.index += 1
                                
                                # 表示
                                st.write(f"結果: {len(ranking_final)} 店舗")
                                st.dataframe(
                                    ranking_final[['設置台数', '店舗名', 'グループ名', '住所']], 
                                    use_container_width=True,
                                    column_config={"設置台数": st.column_config.NumberColumn("設置台数", format="%d 台")}
                                )
                            else:
                                st.warning("この機種を設置している店舗データがありません。")
                        except Exception as e:
                            st.error(f"ランキングの読み込みに失敗しました: {e}")


                    # --- タブ2: 詳細スペック (現在は空の状態) ---
                    with m_tabs[2]:
                        st.write("詳細スペック情報をここに表示予定です。") 
                else:
                    st.error("機種データが見つかりません。")
                    st.link_button("検索に戻る", "/")

            else:
                # --- 【機種一覧画面】 ---
                st.title("🎰 機種別マーケット検索")
                
                # --- サイドバーでの絞り込み ---
                st.sidebar.header("🔍 機種絞り込み")
                
                # 1. 基本検索
                m_type = st.sidebar.selectbox("区分", ["すべて", "パチンコ", "スロット"])
                m_maker = st.sidebar.text_input("メーカー名検索")
                m_name_search = st.sidebar.text_input("機種名検索")
                
 
                # 2. 販売日による期間絞り込み
                st.sidebar.subheader("📅 販売時期")
                
                # NaT（無効な日付）を除外して最小・最大を取得
                valid_dates = df_master['date_dt'].dropna()
                
                if not valid_dates.empty:
                    min_date = valid_dates.min().date()
                    max_date = valid_dates.max().date()
                    
                    # 期間選択（エラー回避のためPythonのdate型に変換して渡す）
                    date_range = st.sidebar.date_input(
                        "販売日の範囲",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                else:
                    st.sidebar.warning("有効な販売日データがありません。")
                    date_range = None
                
                # 3. スペック・特徴による絞り込み
                st.sidebar.subheader("✨ 特徴・スペック")
                f_smapachi = st.sidebar.checkbox("スマパチ (e機)")
                f_smaslo = st.sidebar.checkbox("スマスロ (L機)")
                f_dekaheso = st.sidebar.checkbox("デカへそ")
                
                # --- フィルタリング実行 ---
                f_m = df_master.copy()
                
                # 区分絞り込み
                if m_type == "パチンコ": f_m = f_m[f_m["区分"] == 1]
                elif m_type == "スロット": f_m = f_m[f_m["区分"] == 2]
                
                # 文字列検索
                if m_maker: f_m = f_m[f_m["メーカー"].str.contains(m_maker, na=False)]
                if m_name_search: f_m = f_m[f_m["機種名"].str.contains(m_name_search, na=False)]
                
                # 日付絞り込み（開始日と終了日が選択されている場合）
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                    f_m = f_m[(f_m['date_dt'].dt.date >= start_date) & (f_m['date_dt'].dt.date <= end_date)]
                

                # --- 特徴絞り込み（判定用列を使用） ---
                if f_smapachi:
                    f_m = f_m[f_m['スマパチ判定用'].str.contains("e機", na=False)]
                
                if f_smaslo:
                    # 確実に数値の3と比較
                    f_m = f_m[f_m['スマスロ判定用'] == 3]
                
                if f_dekaheso:
                    f_m = f_m[f_m['デカへそ判定用'] == "あり"]

                # ソート（新しい順）
                f_m = f_m.sort_values(by='date_dt', ascending=False, na_position='last')
                
                # 不要になった一時的な列を削除
                f_m = f_m.drop(columns=['date_dt'])

                # --- 表示処理 ---
                btn_space = st.empty()
                
                # 結果件数の表示
                st.caption(f"該当件数: {len(f_m)} 件")
                
                sel_m = st.dataframe(
                    f_m[['メーカー', '機種名', '発売日']], 
                    on_select="rerun", 
                    selection_mode="single-row", 
                    hide_index=True, 
                    use_container_width=True
                )

                if len(sel_m.selection.rows) > 0:
                    target_row = f_m.iloc[sel_m.selection.rows[0]]
                    btn_space.link_button(
                        f"👉 【詳細分析】{target_row['機種名']} を開く", 
                        f"/?pcode={target_row['pcode']}", 
                        type="primary", 
                        use_container_width=True
                    )
                else:
                    btn_space.info("💡 下の一覧から機種を選択してください。") 

except Exception as e:
    st.error(f"システムエラーが発生しました: {e}")
    import traceback
    st.code(traceback.format_exc())