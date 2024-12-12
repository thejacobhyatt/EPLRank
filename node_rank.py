import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Initialize graph and load data
G = nx.DiGraph()
df = pd.read_csv('final_2024.csv')
bet_df = pd.read_csv('odds_2024.csv')
cumulative_results = pd.DataFrame()


# Project the winner based on the rankings
def project_winner(row, rankings):
    home_rank = rankings.get(row["Home"], float('inf'))
    away_rank = rankings.get(row["Away"], float('inf'))
    
    # Project a tie if ranks are within 3 spots
    if abs(home_rank - away_rank) <= 0:
        return "Draw"
    return "Home" if home_rank > away_rank else "Away"

def get_personalization(previous_weeks):
    personalization = {}

# Iterate over each row in the DataFrame
    for index, row in previous_weeks.iterrows():
        home_team = row['Home']
        away_team = row['Away']
        
        home_personal = row['HCPA_norm'] + row['HPrgP_norm'] + row['HPrgR_norm'] + row['HSCA_norm']
        away_personal = row['ACPA_norm'] + row['APrgP_norm'] + row['APrgR_norm'] + row['ASCA_norm']


        home_personal = home_personal
        away_personal = away_personal
        # Add to the existing value if the team is already in the dictionary
        personalization[home_team] = personalization.get(home_team, 0) + home_personal
        personalization[away_team] = personalization.get(away_team, 0) + away_personal 

    return personalization

def page_rank(G, personalization_vec):
    pagerank = nx.pagerank(G, alpha=0.95, personalization=personalization_vec, max_iter=100000, tol=1e-06, nstart=None, weight='weight', dangling=None)
    sorted_by_value = dict(sorted(pagerank.items(), key=lambda item: item[1], reverse=True))
    rankings = {}
    # Print in sorted order by value
    for i, (key, value) in enumerate(sorted_by_value.items(), start=1):
        rankings[key] = i
    
    return rankings

def backtest_dnb(df, bet_amount=10):
    """
    Backtest model performance using winner, projected_winner, and odds with Draw No Bet (DNB).

    Args:
    - df (DataFrame): DataFrame with columns 'Winner', 'Projected Winner', and odds columns.
    - bet_amount (float): Fixed bet amount for each prediction.

    Returns:
    - result (dict): Metrics including total bets, total returns, net profit, ROI, and accuracy.
    - updated_df (DataFrame): Updated DataFrame with success and profit columns.
    """
    # Add a success column (1 if prediction is correct, 0 otherwise)
    df['success'] = df['Projected Winner'] == df['Winner']
    
    # Add a draw column (1 if match ends in a draw, 0 otherwise)
    df['is_draw'] = df['Winner'] == 'Draw'
    
    # Calculate profit/loss for each row
    def calculate_profit(row):
        if row['is_draw']:
            return 0  # Bet is voided, no profit or loss
        elif row['success']:
            return (row['successful_bet_odds'] - 1) * bet_amount  # Profit for correct prediction
        else:
            return -bet_amount  # Loss for incorrect prediction

    df['profit'] = df.apply(calculate_profit, axis=1)
    
    # Calculate metrics
    total_bets = len(df) * bet_amount
    total_returns = df['profit'].sum() + total_bets
    net_profit = df['profit'].sum()
    roi = (net_profit / total_bets) * 100
    accuracy = df['success'].mean() * 100

    result = {
        'total_bets': total_bets,
        'total_returns': total_returns,
        'net_profit': net_profit,
        'roi': roi,
        'accuracy': accuracy
    }

    return result, df

accuracy_list = []
teams = df['Home'].unique()


    # Normalize home and away stats
home_columns = ['HAtt 3rd', 'HAtt Pen', 'HSucc', 'HSucc%', 'HPrgC', 'HCPA', 'HLive', 'HPrgR', 'Home Score', 'HxG', 'HSCA', 'HPrgP']
away_columns = ['AAtt 3rd', 'AAtt Pen', 'ASucc', 'ASucc%', 'APrgC', 'ACPA', 'ALive', 'APrgR', 'Away Score', 'AxG', 'ASCA', 'APrgP' ]

for col in home_columns + away_columns:
    df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())


# Iterate over weeks 4 to 19
for week_num in range(4, 39):
    

    df['cost'] = df['HxG'] - .9 * df['AxG'] # + df['HAtt 3rd'] - df['AAtt 3rd']

    # Get the last 3 weeks of data
    last_three_weeks = [week_num - 3, week_num - 2, week_num - 1]
    previous_weeks = df[df['Wk'].isin(last_three_weeks)]
    
    # Create edgelist
    edgelist = list(zip(previous_weeks['Home'], previous_weeks['Away'], previous_weeks['cost']))

    # Modify edgelist to handle weight changes
    mod_edgelist = [
        (t[0], t[1], round(abs(t[2]), 2)) if t[2] < 0 else (t[1], t[0], round(t[2], 2)) 
        for t in edgelist
    ]

    # Add nodes and weighted edges to the graph
    for team1, team2, weight in mod_edgelist:
        G.add_edge(team1, team2, weight=weight)
    
    p = get_personalization(previous_weeks)    

    pagerank = page_rank(G, p)

    # Rank teams based on PageRank scores
    rankings = {team: rank for rank, (team, score) in enumerate(sorted(pagerank.items(), key=lambda item: item[1], reverse=True), start=1)}


    # Test week predictions and accuracy calculation
    test_week = df[df['Wk'] == week_num].copy()
    test_week["Projected Winner"] = test_week.apply(project_winner, axis=1, rankings=rankings)
    cumulative_results = pd.concat([cumulative_results, test_week], ignore_index=True)

    correct_predictions = (test_week["Projected Winner"] == test_week["Winner"]).sum()
    total_predictions = len(test_week)
    accuracy = (correct_predictions / total_predictions) * 100
    accuracy_list.append(accuracy)

# Print the average accuracy across weeks
print(f"Average accuracy: {np.mean(accuracy_list):.2f}%")


from pathlib import Path

filepath = Path('cummulative.csv')  
cumulative_results.to_csv(filepath, index=False)


def plot():
    labels = [f'Bar {i+1}' for i in range(len(accuracy_list))]
    plt.bar(labels, accuracy_list)
    average_value = np.mean(accuracy_list)
    plt.axhline(average_value, color='red', linestyle='--', label=f'Average: {average_value:.2f}')
    plt.axhline(50, color='black', linestyle='--', label=f'Base: {50:.2f}')
    plt.legend()
    plt.show()