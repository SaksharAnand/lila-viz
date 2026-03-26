import os
import json
import pandas as pd
import pyarrow.parquet as pq

MAP_CONFIG = {
    "AmbroseValley": {"scale": 900,  "origin_x": -370, "origin_z": -473},
    "GrandRift":     {"scale": 581,  "origin_x": -290, "origin_z": -290},
    "Lockdown":      {"scale": 1000, "origin_x": -500, "origin_z": -500},
}

def world_to_pixel(x, z, map_id):
    cfg = MAP_CONFIG.get(map_id)
    if not cfg:
        return None, None
    u = (x - cfg["origin_x"]) / cfg["scale"]
    v = (z - cfg["origin_z"]) / cfg["scale"]
    return round(u * 1024, 2), round((1 - v) * 1024, 2)

def is_bot(user_id):
    return str(user_id).isdigit()

def load_parquet(filepath):
    try:
        df = pq.read_table(filepath).to_pandas()
        df['event'] = df['event'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
        return df
    except Exception as e:
        print(f"  Skipping {filepath}: {e}")
        return None

def process_all_data(base_dir):
    day_folders = ["February_10","February_11","February_12","February_13","February_14"]
    all_frames = []
    for day in day_folders:
        folder = os.path.join(base_dir, day)
        if not os.path.exists(folder):
            continue
        files = os.listdir(folder)
        print(f"Loading {day}: {len(files)} files...")
        for filename in files:
            df = load_parquet(os.path.join(folder, filename))
            if df is not None and len(df) > 0:
                df['day'] = day
                all_frames.append(df)
    if not all_frames:
        print("ERROR: No files loaded.")
        return None
    combined = pd.concat(all_frames, ignore_index=True)
    print(f"Total rows: {len(combined):,}")
    return combined

def build_output(df):
    df['is_bot'] = df['user_id'].apply(is_bot)
    # datetime64[ms] -> int64 gives milliseconds directly
    df['ts_ms'] = df['ts'].astype('int64')
    match_start = df.groupby('match_id')['ts_ms'].min().rename('match_start_ms')
    df = df.join(match_start, on='match_id')
    df['ts_relative'] = df['ts_ms'] - df['match_start_ms']

    pixels = df.apply(lambda r: world_to_pixel(r['x'], r['z'], r['map_id']), axis=1)
    df['px'] = pixels.apply(lambda p: p[0])
    df['py'] = pixels.apply(lambda p: p[1])
    df = df.dropna(subset=['px','py'])

    print("Building match index...")
    matches = []
    for match_id, group in df.groupby('match_id'):
        duration = int(group['ts_relative'].max())
        clean = match_id.replace('.nakama-0','')
        matches.append({
            "match_id": clean, "raw_match_id": match_id,
            "map_id": group['map_id'].iloc[0], "day": group['day'].iloc[0],
            "humans": int(group[~group['is_bot']]['user_id'].nunique()),
            "bots":   int(group[ group['is_bot']]['user_id'].nunique()),
            "duration_ms": duration, "total_events": len(group)
        })
    matches.sort(key=lambda m:(m['day'],m['match_id']))
    print(f"Total matches: {len(matches)}")
    for m in [x for x in matches if x['duration_ms']>0][:3]:
        print(f"  Sample duration: {m['duration_ms']//60000}m {(m['duration_ms']%60000)//1000}s")

    print("Building per-match event data...")
    match_data = {}
    pos_evts = ['Position','BotPosition']
    act_evts  = ['Kill','Killed','BotKill','BotKilled','KilledByStorm','Loot']
    for match_id, group in df.groupby('match_id'):
        clean = match_id.replace('.nakama-0','')
        players = []
        for user_id, pg in group.groupby('user_id'):
            pg = pg.sort_values('ts_relative')
            path = [[r['px'],r['py'],int(r['ts_relative'])] for _,r in pg[pg['event'].isin(pos_evts)].iterrows()]
            evts = [{"type":r['event'],"px":r['px'],"py":r['py'],"ts":int(r['ts_relative'])} for _,r in pg[pg['event'].isin(act_evts)].iterrows()]
            players.append({"user_id":str(user_id),"is_bot":bool(pg['is_bot'].iloc[0]),"path":path,"events":evts})
        match_data[clean] = {"map_id":group['map_id'].iloc[0],"players":players}

    print("Building heatmap data...")
    heatmaps = {}
    for map_id in df['map_id'].unique():
        mdf = df[df['map_id']==map_id]
        heatmaps[map_id] = {
            'traffic': [[r['px'],r['py']] for _,r in mdf[mdf['event'].isin(pos_evts)].iterrows()],
            'kills':   [[r['px'],r['py']] for _,r in mdf[mdf['event'].isin(['Kill','BotKill'])].iterrows()],
            'deaths':  [[r['px'],r['py']] for _,r in mdf[mdf['event'].isin(['Killed','BotKilled','KilledByStorm'])].iterrows()],
            'loot':    [[r['px'],r['py']] for _,r in mdf[mdf['event']=='Loot'].iterrows()],
        }
    return matches, match_data, heatmaps

def save_output(matches, match_data, heatmaps, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,'matches.json'),'w') as f: json.dump(matches,f)
    mdir = os.path.join(out_dir,'matches')
    os.makedirs(mdir, exist_ok=True)
    for mid, data in match_data.items():
        with open(os.path.join(mdir,f"{mid}.json"),'w') as f: json.dump(data,f)
    with open(os.path.join(out_dir,'heatmaps.json'),'w') as f: json.dump(heatmaps,f)
    print(f"\n✅ All done! Saved {len(matches)} matches.")

if __name__ == "__main__":
    BASE_DIR = os.path.expanduser("~/lila-viz")
    df = process_all_data(BASE_DIR)
    if df is not None: 
        matches, match_data, heatmaps = build_output(df)
        save_output(matches, match_data, heatmaps, os.path.join(BASE_DIR,"data"))
