def get_station_names(df, filename='station_names.json'):
    import json
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    try:
        with open(filename) as f:
            station_names = json.load(f)
    except FileNotFoundError:
        station_names = dict()
        station_id_lt = pd.Series(np.concatenate([df['대여대여소ID'].unique(), df['반납대여소ID'].unique()], axis=0)).unique()
        for station_id in tqdm(station_id_lt):
            if station_id in df['대여대여소ID']:
                station_names[station_id] = df[df['대여대여소ID'] == station_id]['대여대여소명'].iloc[0]
            else:
                station_names[station_id] = df[df['반납대여소ID'] == station_id]['반납대여소명'].iloc[0]

        with open('station_names.json', 'w') as f:
            json.dump(station_names, f)
    
    return station_names

station_names = get_station_names()

def ranking(df, index, groupby, top=10):
    ranks = df[index].groupby(groupby, as_index=False).count()[[groupby, 'bicycle_id']].sort_values('bicycle_id', ascending=False).head(top)
    ranks = ranks.rename(columns={'bicycle_id': 'count'})
    ranks = ranks.set_index(groupby)[['station_name', 'count']]
    return ranks

def station_ranking(df, index, groupby, top):
    ranks = ranking(df, index, groupby, top)
    ranks['station_name'] = ranks[groupby].replace(station_names)
    return ranks