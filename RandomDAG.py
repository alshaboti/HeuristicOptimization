import random 
import networkx as nx
randint = random.randint
class RandomDAG:
    def __init__(self, nodes, n_edges):

        self.n_nodes = len(nodes)
        self.nodes = nodes
        self.n_edges = n_edges

        if n_edges > self.n_nodes * (self.n_nodes - 1):
            self.n_edges = self.n_nodes * (self.n_nodes - 1)

        self.randDAG = nx.DiGraph()

    # connected graph req (n-1) edges at least
    # DAG can't be more than n(n-1) edges
    # https://ipython.org/ipython-doc/3/parallel/dag_dependencies.html

    def random_dag(self):
         
        """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
        # add nodes, labeled 0...nodes:
        
        for i in range(self.n_nodes):
            self.randDAG.add_node(self.nodes[i])

        child_parent = {}
        shuffled_nodes  = self.nodes[:]
        random.shuffle(shuffled_nodes)
        
        # to avoid infinit loop, need to have better solution
        round = 1000
        while self.n_edges > 0 and round > 0:
            round -= 1

            a = random.choice(self.nodes)            
            b = random.choice(self.nodes)
            while a == b or self.randDAG.has_edge(a, b):
                b = random.choice(self.nodes)
                
            self.randDAG.add_edge(a, b)
            if nx.is_directed_acyclic_graph(self.randDAG):
                self.n_edges -= 1
                parent = child_parent.get(b)
                if parent is None:
                    parent = [a]
                else:
                    parent.append(a)
                child_parent[b] = parent
                # print(a,"-> ", b)
            else:
                # we closed a loop!
                self.randDAG.remove_edge(a, b)
                
        return self.randDAG, child_parent

    def get_custom_DAG(self, longest_path_len):         
        """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
        # add nodes, labeled 0...nodes:
        for i in range(self.n_nodes):
            self.randDAG.add_node(self.nodes[i])

        child_parent = {}
        
        task_funcs = []
        shuffled_nodes  = self.nodes[:]
        random.shuffle(shuffled_nodes)        
        n_task_edges = longest_path_len-1

        for _ in range(n_task_edges):
            a = shuffled_nodes[0]
            task_funcs.append(a)
            b = shuffled_nodes[1]
            self.randDAG.add_edge(a, b)
            # print(a, ">", b)
            parent = child_parent.get(b)
            if parent is None:
                parent = [a]
            else:
                parent.append(a)
            child_parent[b] = parent            
            shuffled_nodes = shuffled_nodes[1:]
        rem_edges = self.n_edges - n_task_edges
        
        # to avoid infinit loop, need to have better solution
        round = 0
        while rem_edges > 0 and round < 1000:
            round += 1

            a = random.choice(self.nodes)            
            b = random.choice(self.nodes)
            while a == b or self.randDAG.has_edge(a, b):
                b = random.choice(self.nodes)
                
            self.randDAG.add_edge(a, b)
            if nx.is_directed_acyclic_graph(self.randDAG) and \
                len(self.dag_longest_path(self.randDAG)) == longest_path_len:
                rem_edges -= 1
                # print(a,">", b)
                parent = child_parent.get(b)
                if parent is None:
                    parent = [a]
                else:
                    parent.append(a)
                child_parent[b] = parent
            else:
                # we closed a loop!
                self.randDAG.remove_edge(a, b)
                
        if round > 998:        
            print("Can't construct a task after ", round, " trials")
        
        return self.randDAG, child_parent
        
    def dag_longest_path(self, DAG):
        return nx.dag_longest_path(DAG)