import networkx as nx
import pandas as pd
import ast
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
df = pd.read_json("merged_videoId_vs_Alt_id.txt", orient="index")

df = df[df.altmetric_ids != ""]

B = nx.Graph()

for index, row in df.iterrows():
    for alt_id in ast.literal_eval(row['altmetric_ids']):
        B.add_edge(row['link'], alt_id)
        
c = bipartite.clustering(B,mode='min')
print(B.edges(data=True))

# Separate by group
l, r = bipartite.sets(B)
pos = {}
    
# Update position for node from each group
pos.update((node, (1, index)) for index, node in enumerate(l))
pos.update((node, (2, index)) for index, node in enumerate(r))

nx.draw(B, pos=pos)
plt.show()