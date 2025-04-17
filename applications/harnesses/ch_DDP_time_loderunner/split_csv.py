import pandas as pd

csv_file = "ddp_study_time.csv"
df = pd.read_csv(csv_file, comment='#')

# Number of splits = number of teammates
n_splits = 5
chunk_size = len(df) // n_splits + (len(df) % n_splits > 0)

usernames = ['wish', 'hickmank', 'dschodt', 'spandit', 'galgal'] 

for i, name in enumerate(usernames):
    chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
    output_file = f"hyperparameters.{name}.csv"
    chunk.to_csv(output_file, index=False)
    print(f"Wrote {len(chunk)} rows to {output_file}")
