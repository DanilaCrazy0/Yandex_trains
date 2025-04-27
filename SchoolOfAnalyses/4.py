import pandas as pd
from datetime import datetime
pd.set_option('display.max_columns', None)

N = int(input())
first_data = []
for i in range(N):
    st = input().split()
    record = {
        'number_1': st[0],
        'arrive': datetime.strptime(st[2], "%H:%M:%S"),
        'cost_1': int(st[3])
    }
    first_data.append(record)

M = int(input())
second_data = []
for i in range(M):
    st = input().split()
    record = {
        'number_2': st[0],
        'departure': datetime.strptime(st[1], "%H:%M:%S"),
        'cost_2': int(st[3])
    }
    second_data.append(record)

df1 = pd.DataFrame(first_data)
df2 = pd.DataFrame(second_data)

big_df = df1.merge(df2, how='cross')
big_df['timedelta'] = (big_df['departure'] - big_df['arrive']).dt.total_seconds() / 60
big_df = big_df[big_df['timedelta'] >= 15]
big_df['total_cost'] = big_df['cost_1'] + big_df['cost_2']
big_df = big_df.sort_values(by=['total_cost', 'departure'])
print(big_df['number_1'].iloc[0])
print(big_df['number_2'].iloc[0])

