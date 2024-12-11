import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('2024.csv')


def project_winner(row, rankings):
    home_rank = rankings.get(row["Home"], float('inf'))
    away_rank = rankings.get(row["Away"], float('inf'))
    
    # Project a tie if the ranks are within 3 spots
    if abs(home_rank - away_rank) <= 3:
        return "Draw"
    elif home_rank < away_rank:
        return row["Home"]
    else:
        return row["Away"]
    
def create_G(week_num):
    G = nx.DiGraph()
    df['cost'] = (df["xG"] - df["xG.1"])# + (df['Home Score'] - df['Away Score'])
    
    last_three = [week_num-3,week_num-2,week_num-1]

    week_1_df = df[df['Wk'].isin(last_three)]
    edgelist = list(zip(week_1_df['Home'], week_1_df['Away'], week_1_df['cost']))

    mod_edgelist = []
    for t in edgelist:
        if t[2] < 0:
            modified_tuple = (t[0], t[1], round(abs(t[2]), 2))
        else:
            modified_tuple = (t[1], t[0], round(t[2], 2))

        mod_edgelist.append(modified_tuple)


    nodelist = pd.unique(df['Home'])
    # Assuming week_1_df is a pandas DataFrame with columns 'Home', 'Away', and 'cost'
    weightlist = dict(zip(zip(mod_edgelist[0], mod_edgelist[1]), mod_edgelist[2]))


    for team1, team2, weight in mod_edgelist:
        G.add_edge(team2, team1, weight=weight)


    edge_labels = nx.get_edge_attributes(G, 'weight')
    return G


def page_rank(G):
    pagerank = nx.pagerank(G, alpha=0.95, personalization=None, max_iter=100000, tol=1e-06, nstart=None, weight='weight', dangling=None)

    sorted_by_value = dict(sorted(pagerank.items(), key=lambda item: item[1], reverse=True))
    rankings = {}
    # Print in sorted order by value
    for i, (key, value) in enumerate(sorted_by_value.items(), start=1):
        rankings[key] = i
    
    return rankings

def node_rank(G):
    rankings = {}

    incoming_weights = {
        node: sum(data['weight'] for _, _, data in G.in_edges(node, data=True))
        for node in G.nodes
    }

    sorted_by_value = dict(sorted(incoming_weights.items(), key=lambda item: item[1], reverse=True))


    for i, (key, value) in enumerate(incoming_weights.items(), start=1):
        rankings[key] = i

    return rankings

def validate_week(week_num, rank_dic):
    test_week_df = df[df['Wk'] == week_num].copy()  # Make a copy to avoid SettingWithCopyWarning
    test_week_df.loc[:, "Projected Winner"] = test_week_df.apply(project_winner, axis=1, args=(rank_dic,))

    # Raw Accuracy
    # correct_predictions = (test_week_df["Projected Winner"] == test_week_df["Winner"]).sum()
    # Draw No Bet
    correct_predictions = ((test_week_df["Projected Winner"] == test_week_df["Winner"]) | (test_week_df["Winner"] == "Draw")).sum()
    total_predictions = len(test_week_df)
    accuracy = (correct_predictions / total_predictions) * 100

    return int(accuracy)

if __name__ == "__main__":
    for week_num in range(4,38):
        accuracy_list = []
        G = create_G(week_num)

        rankings = node_rank(G)

        accuracy_list.append(validate_week(week_num, rankings))

    
    print(np.mean(accuracy_list))
