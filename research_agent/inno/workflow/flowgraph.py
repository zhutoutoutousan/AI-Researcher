from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import json
from typing import Dict
from copy import deepcopy
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)  # use node_id as key
        self.node_id_counter = 0  # for unique node_id
        self.node_name_to_id = {}  # node_name -> node_id
        self.nodes = {}  # node_id -> {'node_name': str, 'node_id': int, ...}
        self.edge_attributes = {}  # (u_id, v_id) -> attributes

    def add_node(self, node_name, **attributes):
        """
        Add a node, return its node_id if it exists, otherwise create a new node.

        Parameters:
            node_name (str): The node name.
            **attributes: Other optional node attributes.
        """
        if node_name not in self.node_name_to_id:
            node_id = self.node_id_counter
            self.node_name_to_id[node_name] = node_id
            # initialize node attributes dictionary
            node_attrs = {'node_name': node_name, 'node_id': node_id}
            node_attrs.update(attributes)  # add other attributes
            self.nodes[node_id] = node_attrs
            self.node_id_counter += 1
        else:
            # if the node exists, update its attributes
            node_id = self.node_name_to_id[node_name]
            self.nodes[node_id].update(attributes)
        return self.node_name_to_id[node_name]
    def update_nodes(self, nodes):
        for node in nodes:
            self.add_node(node['node_name'], **node.get('node_attrs', {}))
    def update_node(self, node_name, **attributes):
        assert node_name in self.node_name_to_id, f"Node {node_name} does not exist"
        node_id = self.node_name_to_id[node_name]
        self.nodes[node_id].update(attributes)

    def add_edge(self, u, v, **node_attributes):
        """
        Add an edge from node u to node v, and add or update node attributes at the same time.

        Parameters:
            u (str): The start node name.
            v (str): The target node name.
            **node_attributes: Optional node attributes, which will be applied to nodes u and v.
        """
        # Add or update node u's attributes
        u_id = self.add_node(u, **node_attributes.get('u_attrs', {}))
        # Add or update node v's attributes
        v_id = self.add_node(v, **node_attributes.get('v_attrs', {}))
        self.graph[u_id].append(v_id)
        # Store edge attributes
        self.edge_attributes[(u_id, v_id)] = node_attributes.get('edge_attrs', {})

    def add_edges(self, edges):
        for edge in edges:
            if len(edge) == 3:
                self.add_edge(edge[0], edge[1], **edge[2])
            else:
                self.add_edge(edge[0], edge[1])
    def detect_cycle_util(self, v, visited, rec_stack):
        visited[v] = True
        rec_stack[v] = True

        for neighbor in self.graph[v]:
            if not visited[neighbor]:
                if self.detect_cycle_util(neighbor, visited, rec_stack):
                    return True
            elif rec_stack[neighbor]:
                return True

        rec_stack[v] = False
        return False

    def has_cycle(self):
        # collect all nodes (including nodes without outgoing edges)
        all_nodes = set(self.graph.keys())
        for neighbors in self.graph.values():
            all_nodes.update(neighbors)

        visited = {node: False for node in all_nodes}
        rec_stack = {node: False for node in all_nodes}

        for node in all_nodes:
            if not visited[node]:
                if self.detect_cycle_util(node, visited, rec_stack):
                    return True
        return False

    def find_cycles(self):
        cycles = []

        # collect all nodes
        all_nodes = set(self.graph.keys())
        for neighbors in self.graph.values():
            all_nodes.update(neighbors)

        def dfs_cycle(node, visited, rec_stack, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    dfs_cycle(neighbor, visited, rec_stack, path)
                elif neighbor in rec_stack:
                    # find cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:].copy())

            path.pop()
            rec_stack.remove(node)

        visited = set()
        for node in all_nodes:
            if node not in visited:
                dfs_cycle(node, visited, set(), [])

        return cycles

    def find_all_paths(self, start, end, max_cycle_repeat=2):
        start_id = self.add_node(start)
        end_id = self.add_node(end)

        def is_cycle_complete(path, node):
            # check if the cycle is complete
            if node not in path:
                return True
            last_idx = len(path) - 1
            while last_idx >= 0 and path[last_idx] != node:
                last_idx -= 1
            # from the last occurrence to the end must form a complete cycle
            cycle_nodes = set(path[last_idx:])
            for i in range(last_idx, len(path) - 1):
                if path[i + 1] not in self.graph[path[i]]:
                    return False
            return True

        def is_valid_path(path):
            # check if there is a longest repeated substring in the path
            n = len(path)
            for length in range(2, n // 2 + 1):
                for i in range(n - 2 * length + 1):
                    if path[i:i + length] == path[i + length:i + 2 * length]:
                        return False
            return True

        def dfs(current, end, path, paths):
            path.append(current)

            if current == end:
                paths.append(path.copy())
            else:
                for neighbor in self.graph[current]:
                    count = path.count(neighbor)
                    if count < max_cycle_repeat and is_cycle_complete(path, neighbor):
                        dfs(neighbor, end, path, paths)

            path.pop()

        # collect all paths
        all_paths = []
        dfs(start_id, end_id, [], all_paths)

        # convert node_id to node_name
        all_paths_named = []
        for path in all_paths:
            named_path = [self.nodes[node]['node_name'] for node in path]
            all_paths_named.append(named_path)

        # filter paths
        filtered_paths = self.filter_paths(all_paths_named)

        return filtered_paths
    
    def set_start(self, start):
        self.start = start
        self.nodes[self.add_node(start)]['color'] = 'red'
        self.nodes[self.add_node(start)]['shape'] = 's'
    def set_end(self, end):
        self.end = end
        self.nodes[self.add_node(end)]['color'] = 'green'
        self.nodes[self.add_node(end)]['shape'] = '^'


    def filter_paths(self, all_paths):
        """
        Filter out shorter paths that can be generated by repeating substrings in longer paths.
        For example, if there is a path S -> A -> B -> C -> D -> C -> D -> F -> Z,
        then remove the path S -> A -> B -> C -> D -> F -> Z
        """
        def is_subpath(short, long):
            # 检查 short 是否是 long 的子序列
            it = iter(long)
            return all(node in it for node in short)

        # 按路径长度降序排序
        sorted_paths = sorted(all_paths, key=lambda x: len(x), reverse=True)
        filtered = []

        for path in sorted_paths:
            if not any(is_subpath(path, existing) for existing in filtered):
                filtered.append(path)

        return filtered

    def visualize(self):
        G = nx.DiGraph()

        # add nodes and collect colors and shapes
        node_colors = []
        node_shapes = defaultdict(list)
        for node_id, attrs in self.nodes.items():
            label = attrs.get('node_name', f"Node{node_id}")
            shape = attrs.get('shape', 'o')  # 默认形状为圆形
            node_shapes[shape].append(node_id)
            color = attrs.get('color', 'lightblue')
            node_colors.append(color)
            G.add_node(node_id, label=label)

        # add edges
        for u, neighbors in self.graph.items():
            for v in neighbors:
                edge_attr = self.edge_attributes.get((u, v), {})
                G.add_edge(u, v, **edge_attr)

        # get node labels
        labels = {node: attrs['node_name'] for node, attrs in self.nodes.items()}

        # get node colors
        node_color_map = [attrs.get('color', 'lightblue') for node, attrs in self.nodes.items()]

        # get node shapes
        shapes = set(attrs.get('shape', 'o') for attrs in self.nodes.values())

        # set plot layout
        pos = nx.spring_layout(G, seed=42)  # fixed layout for reproducibility

        plt.figure(figsize=(12, 8))

        # draw nodes with different shapes
        for shape in shapes:
            shaped_nodes = [node for node in self.nodes if self.nodes[node].get('shape', 'o') == shape]
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=shaped_nodes,
                node_shape=shape,
                node_color=[self.nodes[node].get('color', 'lightblue') for node in shaped_nodes],
                node_size=1500,
                alpha=0.9
            )

        # draw node labels
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color='black')

        # adjust edge colors or styles based on edge attributes
        edge_colors = []
        for u, v in G.edges():
            edge_attr = self.edge_attributes.get((u, v), {})
            edge_colors.append(edge_attr.get('color', 'black'))

        # draw edges, ensure arrows are displayed
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_colors,
            arrows=True,
            arrowstyle='->',
            arrowsize=20,
            connectionstyle='arc3,rad=0.1',  # increase arc3 rad to avoid arrow overlap
            width=2
        )

        plt.title("Visualization", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def merge_paths(self, paths):
        """
        Merge shorter paths into longer paths while keeping the longer paths unchanged.
        Remove the longest common substring at the beginning and end, then insert the middle part.
        
        Parameters:
            paths (list of list): Multiple paths, each path is a list of node names.
        
        Returns:
            list: The merged paths.
        """
        if not paths:
            return []

        # sort paths by length in descending order
        paths_sorted = sorted(paths, key=lambda p: len(p), reverse=True)
        merged_path = paths_sorted[0].copy()

        # build a directed graph for topological sorting
        G = nx.DiGraph()
        for path in paths:
            for i in range(len(path) - 1):
                G.add_edge(path[i], path[i + 1])

        # create node order mapping (using the longest path as the baseline order)
        node_order = {node: idx for idx, node in enumerate(merged_path)}
        
        for short_path in paths_sorted[1:]:
            # find the longest common substring at the beginning
            start_idx = 0
            while start_idx < len(short_path):
                if short_path[start_idx] in merged_path:
                    # check if it is the start of the common sequence
                    merged_idx = merged_path.index(short_path[start_idx])
                    j = 1
                    while (start_idx + j < len(short_path) and 
                        merged_idx + j < len(merged_path) and 
                        short_path[start_idx + j] == merged_path[merged_idx + j]):
                        j += 1
                    start_idx = start_idx + j - 1
                    break
                start_idx += 1

            # find the longest common substring at the end
            end_idx = len(short_path) - 1
            while end_idx > start_idx:
                if short_path[end_idx] in merged_path:
                    # check if it is the end of the common sequence
                    merged_idx = merged_path.index(short_path[end_idx])
                    j = 1
                    while (end_idx - j > start_idx and 
                        merged_idx - j >= 0 and 
                        short_path[end_idx - j] == merged_path[merged_idx - j]):
                        j += 1
                    end_idx = end_idx - j + 1
                    break
                end_idx -= 1

            # get the sequence to be inserted (after removing the overlapping sequence at the beginning and end)
            insert_sequence = short_path[start_idx+1:end_idx]

            if insert_sequence:
                # find a suitable insertion position in merged_path
                insert_pos = merged_path.index(short_path[start_idx]) + 1
                
                # assign a base order to the entire sequence
                sequence_base_order = -1
                for node in insert_sequence:
                    if node in node_order:
                        sequence_base_order = max(sequence_base_order, node_order[node])
                
                if sequence_base_order == -1:
                    # If the sequence does not contain known nodes, determine the position based on the topological order
                    pred_pos = node_order[short_path[start_idx]]
                    succ_pos = node_order[merged_path[insert_pos]]
                    sequence_base_order = (pred_pos + succ_pos) / 2

                # Check if the sequence already exists
                sequence_exists = False
                for i in range(len(merged_path) - len(insert_sequence) + 1):
                    if merged_path[i:i+len(insert_sequence)] == insert_sequence:
                        sequence_exists = True
                        break

                # Only insert if the sequence does not exist
                if not sequence_exists:
                    for node in insert_sequence:
                        merged_path.insert(insert_pos, node)
                        node_order[node] = sequence_base_order
                        insert_pos += 1

        return merged_path

    def get_node_predecessors_successors(self):
        """
        Get the direct predecessors and successors of each node.
        
        Returns:
            dict: {node_name: {'predecessors': set(), 'successors': set()}}
        """
        result = {}
        
        # initialize the result dictionary
        for node_id in self.nodes:
            node_name = self.nodes[node_id]['node_name']
            result[node_name] = {
                'predecessors': set(),
                'successors': set()
            }
        
        # build direct predecessor and successor relationships
        for u_id, neighbors in self.graph.items():
            u_name = self.nodes[u_id]['node_name']
            for v_id in neighbors:
                v_name = self.nodes[v_id]['node_name']
                # add direct relationships
                result[v_name]['predecessors'].add(u_name)
                result[u_name]['successors'].add(v_name)
        self.node_predecessors_successors = result
        
        return result
    
    def path2workflow(self, path):
        workflow_steps = []
        if hasattr(self, 'node_predecessors_successors') is False:
            self.get_node_predecessors_successors()
        
        for node in path:
            if node == self.end or node == self.start:
                continue
            output_flag = False
            n_predecessors = self.node_predecessors_successors[node]['predecessors']
            n_successors = self.node_predecessors_successors[node]['successors']
            node_id = self.node_name_to_id[node]
            agent_tools = deepcopy(self.nodes[node_id].get('agent_tools', []))
            ops_agent_tools = deepcopy(self.nodes[node_id].get('ops_agent_tools', []))
            for successor in n_successors:
                if successor == self.end:
                    output_flag = True
                    continue
                # agent_tools.append(f"transfer_to_{successor}")
            input_text = []
            for predecessor in n_predecessors:
                if predecessor == self.start:
                    input_text.append(f'Input {predecessor}')
                else: 
                    input_text.append(f'The output of {predecessor} agent')
            input_text = ','.join(input_text)

            output_text = self.nodes[node_id].get('output', '')


            if output_flag:
                workflow_steps.append({"agent_name": node, "agent_tools": agent_tools, "input": input_text, "output": f"Output {self.end}", "ops_agent_tools": ops_agent_tools})
            else:
                workflow_steps.append({"agent_name": node, "agent_tools": agent_tools, "input": input_text, "output": output_text, "ops_agent_tools": ops_agent_tools})
        
        return workflow_steps
    def get_workflow_steps(self):
        paths = self.find_all_paths(self.start, self.end, max_cycle_repeat=3)
        # merge paths
        merged_path = self.merge_paths(paths)
        workflow = self.path2workflow(merged_path)
        workflow = self.refine_workflow(workflow)
        return workflow
    def refine_workflow(self, workflow): 
        agent_dict = {}
        work_lens = len(workflow)
        for step in workflow: 
            agent_dict[step['agent_name']] = set()
        for i in range(work_lens - 1): 
            step_front = workflow[i]
            step_back = workflow[i + 1]
            for tool in step_front['agent_tools']: 
                agent_dict[step_front['agent_name']].add(tool)
            for tool in step_back['agent_tools']: 
                agent_dict[step_back['agent_name']].add(tool)
            agent_dict[step_front['agent_name']].add('transfer_to_' + '_'.join(step_back['agent_name'].lower().split(' ')))
        for i in range(work_lens): 
            agent_name = workflow[i]['agent_name']
            workflow[i]['agent_tools'] = list(agent_dict[agent_name])
            self.nodes[self.node_name_to_id[agent_name]]['agent_tools'] = list(agent_dict[agent_name])
        return workflow
    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        graph = cls()
        edges = [(edge['start'], edge['end']) for edge in data['edges']]
        for node in data['nodes']:
            if node['is_start']:
                start = node['agent_name']
                # data['nodes'].remove(node)
            if node['is_end']:
                end = node['agent_name']
                # data['nodes'].remove(node)
        graph.set_start(start)
        graph.set_end(end)
        graph.add_edges(edges)
        node_attrs = [{'node_name': node['agent_name'], 'node_attrs': {'agent_tools': node['agent_tools'], 'output': node['output']}} for node in data['nodes']]
        graph.update_nodes(node_attrs)
        return graph
    @classmethod
    def from_dict(cls, data: Dict):
        graph = cls()
        edges = [(edge['start'], edge['end']) for edge in data['edges']]
        for node in data['nodes']:
            if node['is_start']:
                start = node['agent_name']
                # data['nodes'].remove(node)
            if node['is_end']:
                end = node['agent_name']
                # data['nodes'].remove(node)
        graph.set_start(start)
        graph.set_end(end)
        graph.add_edges(edges)
        node_attrs = [{'node_name': node['agent_name'], 'node_attrs': {'agent_tools': node['agent_tools'], 'output': node['output']}} for node in data['nodes']]
        graph.update_nodes(node_attrs)
        return graph
    def to_dict(self):
        graph_dict = {}
        graph_dict['nodes'] = []
        graph_dict['edges'] = []
        for node_id, node in self.nodes.items():
            graph_dict['nodes'].append({'agent_name': node['node_name'], 'agent_tools': node['agent_tools'], 'output': node['output'], 'is_start': self.start == node['node_name'], 'is_end': self.end == node['node_name']})
        
        for u, v in self.graph.items():
            for v_id in v:
                graph_dict['edges'].append({'start': self.nodes[u]['node_name'], 'end': self.nodes[v_id]['node_name']})
        return graph_dict
