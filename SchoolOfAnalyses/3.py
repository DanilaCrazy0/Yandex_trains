import pandas as pd
import sqlite3
import numpy as np

conn = sqlite3.connect('moderation.db')
verdicts = pd.read_sql('SELECT * FROM verdicts;', conn)
conn.close()

# print(verdicts)


def get_sample(df):
    user_ids = df['uid'].values
    verdicts = df['verdict'].values
    unique_users, user_indices = np.unique(user_ids, return_inverse=True)

    is_yes = (verdicts == 'Yes')
    accepted_counts = np.bincount(user_indices, weights=is_yes)
    total_counts = np.bincount(user_indices)
    acceptance_rates = accepted_counts / total_counts

    user_to_rate = dict(zip(unique_users, acceptance_rates))
    rejected = df[df['verdict'] == 'No'].copy()

    rejected['acceptance_rate'] = rejected['uid'].map(user_to_rate)
    top_banners = rejected.sort_values('acceptance_rate', ascending=False).head(100)

    return top_banners['banner_id']


print(get_sample(verdicts))
