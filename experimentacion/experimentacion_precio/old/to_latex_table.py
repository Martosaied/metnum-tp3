import json
import pandas as pd

# parse x:
with open('results_todos_ordenados.json') as f:
  data = json.load(f)

columns=['RMSE', 'RMSLE', 'R2', 'MAE']
for col in columns:
    df = pd.DataFrame.from_dict(data[col], orient='index', columns=columns)
    with open(f'{col}.txt', 'w') as f:
        with pd.option_context("max_colwidth", 1000):
            f.write(df.to_latex(column_format='p{6cm}|r|r|r|r'))