# Graph Theory and Network Analysis

## Introduction

Graph theory is a field of mathematics and computer science that deals with the study of graphs, which are mathematical structures used to model pairwise relations between objects. In this context, networks are considered as graphs and are pivotal in analyzing and interpreting real-world complex systems.

Network analysis, as an application of graph theory, is indispensable in understanding the structure and dynamics of social networks, transportation systems, biological networks, and in many other disciplines. The framework of graph theory equips researchers with tools and techniques to study connectivity, community structure, and the importance of individual elements within a network.

In this notebook, we delve into the essential concepts of graph theory and network analysis, focusing on graph representations, algorithms, centrality measures, community detection, and applications in network science. 

## Graph Representations

### Theory

Graph representations are crucial for the effective implementation and analysis of graph-based problems. There are various ways to represent a graph:

1. **Adjacency Matrix:** A 2D array, `A`, where the cell `A[i][j]` is `1` if there is an edge from vertex `i` to vertex `j`; otherwise, it is `0`. This representation is space-inefficient for sparse graphs.
   
2. **Adjacency List:** Each vertex maintains a list of its adjacent vertices. This is efficient in terms of space for sparse graphs and makes iterating over edges easy.

3. **Edge List:** A list of all edges represented as pairs of vertices. This representation is compact and useful when we need to work directly with edges.

4. **Incidence Matrix:** A matrix with rows representing vertices and columns representing edges. The cell value `1` if the vertex is at one end of the edge, useful in theoretical analysis.

### Examples

**Adjacency Matrix Example:**
```python
import numpy as np

# Example graph with 3 vertices and edges between (0,1), (1,2), and (2,0)
adjacency_matrix = np.array([[0, 1, 0],
                             [0, 0, 1],
                             [1, 0, 0]])
```

**Adjacency List Example:**
```python
# Adjacency list representation of the same graph
adjacency_list = {
    0: [1],
    1: [2],
    2: [0]
}
```

Applications:

- **Social Networks:** Adjacency matrices can be useful for fast lookup operations while an adjacency list can be more efficient for traversing a social network graph.
  
- **Flight Routes:** Airlines use adjacency lists to efficiently navigate and manage flight paths between airports.

## Graph Algorithms

### Theory

Graph algorithms are fundamental in extracting meaningful insights from graphs and include:

1. **Depth-First Search (DFS):** Explores as far as possible along each branch before backtracking.
   
2. **Breadth-First Search (BFS):** Examines neighbors level by level, useful for shortest path in unweighted graphs.

3. **Dijkstra's Algorithm:** Finds the shortest path from a source to other vertices in a weighted graph.

4. **Kruskal’s and Prim’s Algorithms:** Used for finding the minimum spanning tree in a weighted graph.

### Examples

**Breadth-First Search Example:**
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(set(graph[vertex]) - visited)
            print(vertex)

bfs(adjacency_list, 0)
```

**Dijkstra's Algorithm Example:**
```python
import heapq

def dijkstra(graph, start):
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_vertex]:
            continue
        
        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

graph_with_weights = {
    0: [(1, 1), (2, 4)],
    1: [(2, 2)],
    2: [(0, 4)]
}
dijkstra(graph_with_weights, 0)
```

Applications:

- **Shortest Path in Cities:** BFS can perform well for routing in cities with unweighted graphs for equal-length blocks.

- **Network Routing Protocols:** Dijkstra’s algorithm is extensively used in routing information protocol (RIP) network routers to determine the shortest path.

## Centrality Measures

### Theory

Centrality measures are metrics that identify the most important vertices within a graph. Key centrality measures include:

1. **Degree Centrality:** Number of edges connected to a vertex. High degree centrality indicates an influential node.
   
2. **Betweenness Centrality:** Number of shortest paths passing through a vertex, identifying bridges within networks.

3. **Closeness Centrality:** Average length of the shortest path from a vertex to all other vertices, indicating how long it will take to spread information from a vertex.

4. **Eigenvector Centrality:** Measures a node's influence based on its connections. High eigenvector centrality indicates connections to important nodes.

### Examples

**Degree Centrality Example:**
```python
import networkx as nx

G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3)])
degree_centrality = nx.degree_centrality(G)
print(degree_centrality)
```

**Betweenness Centrality Example:**
```python
betweenness_centrality = nx.betweenness_centrality(G)
print(betweenness_centrality)
```

Applications:

- **Influential Persons in Social Networks:** Degree centrality is a simple yet effective measure to identify influencers in social platforms.

- **Vulnerability in Infrastructure Networks:** Betweenness centrality can identify critical nodes in a transportation system or communication network.

## Community Detection

### Theory

Community detection in graphs involves the grouping of vertices into clusters such that there are denser connections within clusters than between them. Major methods include:

1. **Modularity Maximization:** Measures the density of links inside communities compared to links between communities. Algorithms like the Girvan-Newman and Louvain method use this principle.

2. **Spectral Clustering:** Uses the eigenvalues of graph Laplacian matrix for clustering nodes.

3. **Label Propagation:** Nodes adopt the majority label among their neighbors, iteratively detecting communities.

### Examples

**Louvain Method for Community Detection:**
```python
import community as community_louvain

partition = community_louvain.best_partition(G)
print(partition)
```

Applications:

- **Social Media Analysis:** Detecting groups of users with common interests or interactions.

- **Biological Networks:** Identifying functional modules or communities of proteins forming complexes.

## Applications in Network Science

### Theory

Network science explores real-world complex systems modeled as graphs to understand their behavior, functionality, and evolution. Networks studied include:

- **Social Networks:** Analyze interactions, influence, and community structures among people.
  
- **Biological Networks:** Understand metabolic pathways or neural connections.

- **Information Networks:** Structure of the World Wide Web and content delivery networks.

### Examples

**Social Network Analysis with NetworkX:**
```python
# Analyzing a simple social network
social_network = nx.Graph()
social_network.add_edges_from([("Alice", "Bob"), ("Bob", "Cathy"), ("Cathy", "Alice"), ("Cathy", "Dan")])

centrality = nx.degree_centrality(social_network)
print("Degree Centrality:", centrality)
```

Applications:

- **Epidemiology:** Studying the spread of diseases in contact networks allows for intervention strategies.
  
- **Internet Architecture:** Traffic modeling and optimization in network infrastructures.

In summary, graph theory and network analysis provide an extensive toolkit for understanding, modeling, and analyzing complex systems encountered in numerous fields from sociology to computer science and beyond. By understanding graph representations, algorithms, and measures, we can gain insights into the intricate dynamics of networks.