from pathlib import Path
import pandas as pd

url_df = 'https://fbref.com/en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures'
df = pd.read_html(url_df)

df = pd.read_html(url_df)[0]
df.head()
df.columns = [' '.join(col).strip() for col in df.columns]

df = df.reset_index(drop=True)
df.head()

df.columns = [col.replace(" ", "") for col in df.columns]
df = df.dropna(how='all')
df = df.drop('MatchReport', axis=1)
df = df.drop('Notes', axis=1)

df['Split Score'] = df['Score'].str.split('–')
df['Home Score'] = df['Split Score'].str[0]
df['Away Score'] = df['Split Score'].str[1]
df = df.drop('Split Score', axis=1)

def get_winner(row):
    # Replace any non-standard dash with a standard hyphen and split scores
    home_score = row['Home Score']
    away_score = row['Away Score']    
    
    # Determine the winner
    if home_score > away_score:
        return row["Home"]
    elif home_score < away_score:
        return row["Away"]
    else:
        return "Draw"
    
# Apply the function to each row and create a new column 'Winner'
df["Winner"] = df.apply(get_winner, axis=1)