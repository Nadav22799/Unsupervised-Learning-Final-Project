import networkx as nx
import json

from tqdm import tqdm

def read_json_file(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return js_graph

file = 'deezer_edges.json'

graph_dics = read_json_file(file)


# average_node_connectivity
with open("periphery_size.txt", "tw") as fout:
    for i in tqdm(range(0,len(graph_dics))):
        index = str(i)
        G = nx.Graph()
        G.add_edges_from(graph_dics[index])
        
        c = c = nx.algorithms.barycenter(G)
        value = len(nx.algorithms.periphery(G))

        fout.write(f"{value}\n")