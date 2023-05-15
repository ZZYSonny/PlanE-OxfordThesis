from utils.common_imports import *
import hashlib
import pickle

SPQR_NODE = typing.Tuple[str, sageall.Graph]
SPQR_EDGE = typing.Tuple[int, int, typing.Union[None, str]]

CUT_PAIR = typing.Tuple[int, int, str]
NODE_MAP = typing.Dict[int, int]
EDGE_MAP = typing.Dict[SPQR_EDGE, typing.Union[None, int, 'ENCODING']]

PLANAR_EMBEDDING = typing.Dict[int, list[int]]

P_BEGIN = -6
P_END = -5
S_BEGIN = -4
S_END = -3
R_BEGIN = -2
R_END = -1
C_BEGIN = -8
C_END = -7

BRACKET_BEGIN = {
    'S': S_BEGIN,
    'P': P_BEGIN,
    'Q': S_BEGIN,
    'R': R_BEGIN,
    'C': C_BEGIN,
}

BRACKET_END = {
    'S': S_END,
    'P': P_END,
    'Q': S_END,
    'R': R_END,
    'C': C_END,
}

EDGE_OFFSET = 1000
NODE_OFFSET = 2000
BC_NODE_OFFSET = 3000

class ENCODING:
    code: list[int]
    spqr_node: SPQR_NODE
    spqr_cycle: list[SPQR_EDGE]
    spqr_code: list[int]
    spqr_adj_code: list['ENCODING']

    def __init__(self):
        self.code = []
        self.spqr_node = None
        self.spqr_cycle = []
        self.spqr_code = []
        self.spqr_adj_code = []

    def __lt__(self, other: typing.Union[int, 'ENCODING']):
        if isinstance(other, int):
            return False
        else:
            return self.code < other.code
    
    def __eq__(self, other: 'ENCODING'):
        if isinstance(other, int):
            return False
        else:
            return self.code == other.code

    def append_bracket(self, bracket: int):
        assert(bracket < 0)
        self.code.append(bracket)

    def append_char(self, char: int):
        assert(char >= 0)
        self.code.append(char)

    def append_node(self, code: typing.Union[int, 'ENCODING']):
        if isinstance(code, int):
            self.code.append(code)
        else:
            self.code.extend(code.code)

    def append_edge(self, edge_code: typing.Union[int, 'ENCODING']):
        if isinstance(edge_code, int):
            self.code.append(edge_code)
        else:
            self.code.extend(edge_code.code)
            self.spqr_adj_code.append(edge_code)
    
    def get_cycles(self):
        ans = {}
        if self.spqr_node is not None:
            ans[self.spqr_node] = (
                self.spqr_cycle,
                self.spqr_code
            )
        for adj_code in self.spqr_adj_code:
            ans.update(adj_code.get_cycles())
        return ans

EMPTY_ENCODING = ENCODING()

def construct_spqr_code(spqr_node, cycle, nid, node_code, edge_code):
    code = ENCODING()
    code.spqr_node = spqr_node
    code.append_bracket(BRACKET_BEGIN[spqr_node[0]])
    # Generating the encoding for isomorphism testing
    # Add nid to the encoding
    for i,e in enumerate(cycle):
        (u,v,label) = e
        code.append_char(nid[u])
    # Add node encoding
    for u,i in nid.items():
        code.append_char(NODE_OFFSET+i)
        code.append_node(node_code[u])
    # Add edge encoding
    for i,e in enumerate(cycle):
        if e in edge_code:
            code.append_char(EDGE_OFFSET+i)
            code.append_edge(edge_code[e])
    code.append_bracket(BRACKET_END[spqr_node[0]])

    # Generating the information for data preprocessing
    code.spqr_cycle = cycle
    code.spqr_code = [
        nid[u]
        for u,v,label in cycle
    ]
    return code


def encode_tcc_with_edge_with_direction(spqr_node: SPQR_NODE, e0: CUT_PAIR,
                                        node_code: NODE_MAP, edge_code: EDGE_MAP, 
                                        d: int):
    (spqr_type, g) = spqr_node
    # Get planar embedding
    if not hasattr(g, '_embedding'):
        is_planar = g.is_planar(set_embedding=True)
        assert(is_planar)

    adj: PLANAR_EMBEDDING = getattr(g, "_embedding")

    # Next edge
    def next_eid(u:int, now:int):
        l = len(adj[u])
        return (now+d+l)%l
    
    # node in g -> id in the cycle
    nid = {
        e0[0]: 0,
        e0[1]: 1
    }
    # Visited nodes
    vis = set([e0])
    # Used edges
    used_edges:set[tuple[int,int]] = set()
    # List of edges in the cycle
    cycle = [e0]

    last_node, now_node, _ = e0
    for _ in range(1,2*g.num_edges()):
        used_edges.add((last_node,now_node))

        if now_node not in vis:
            # New vertex
            # Use the next edge
            eid = next_eid(now_node, adj[now_node].index(last_node))
            next_node = adj[now_node][eid]
            vis.add(now_node)
        elif (now_node, last_node) not in used_edges:
            # Old vertex, reversed edge unused
            # Use the reversed edge
            next_node = last_node
        else:
            # Old vertex, reversed edge used
            # Find the next edge
            eid = next_eid(now_node, adj[now_node].index(last_node))
            next_node = adj[now_node][eid]
            while next_node is None or (now_node, next_node) in used_edges:
                eid = next_eid(now_node, eid)
                next_node = adj[now_node][eid]

        used_edges.add((now_node,next_node))
        last_node, now_node = now_node, next_node
        nid.setdefault(next_node, len(nid))

        # Add edge to cycle
        [edge_name] = g.edge_label(last_node, now_node)
        cycle.append((last_node, now_node, edge_name))
    
    code = construct_spqr_code(spqr_node, cycle, nid, node_code, edge_code)
    return code

def encode_r_node(spqr_node: SPQR_NODE, e0: CUT_PAIR,
                  node_code: NODE_MAP, edge_code: EDGE_MAP):
    return min(
        encode_tcc_with_edge_with_direction(spqr_node, e0, node_code, edge_code, d)
        for d in [-1, 1]
    )

def encode_s_node(spqr_node: SPQR_NODE, e0: CUT_PAIR,
                  node_code: NODE_MAP, edge_code: EDGE_MAP):
    (spqr_type, g) = spqr_node
    (u,v,label) = e0
    nodes = list(g.depth_first_search(u))
    if nodes[1]!=v:
        nodes = [u] + list(reversed(nodes[1:]))
    cycle = [
        (x,y,label)
        for x,y in zip(nodes, nodes[1:]+[u])
        for label in g.edge_label(x,y)
    ]
    nid = {
        x:i
        for i,x in enumerate(nodes)
    }

    code = construct_spqr_code(spqr_node, cycle, nid, node_code, edge_code)

    return code


def encode_p_node(spqr_node: SPQR_NODE, e0: CUT_PAIR,
                  node_code: NODE_MAP, edge_code: EDGE_MAP):
    (spqr_type, g) = spqr_node
    cycle = sorted(list(
        g.edge_iterator(e0[0])), 
        key=lambda e: edge_code.get(e, EMPTY_ENCODING)
    )
    cycle = [e0] + [e for e in cycle if e!=e0]

    nid = {
        e0[0]: 0,
        e0[1]: 1
    }
    code = construct_spqr_code(spqr_node, cycle, nid, node_code, edge_code)
    return code


def encode_r_root(spqr_node: SPQR_NODE, node_code: NODE_MAP, edge_code: EDGE_MAP):
    return min(
        encode_r_node(spqr_node, pair, node_code, edge_code)
        for u,v,label in spqr_node[1].edge_iterator()
        for pair in [(u,v,label), (v,u,label)]
    )


def encode_s_root(spqr_node: SPQR_NODE, node_code: NODE_MAP, edge_code: EDGE_MAP):
    return min(
        encode_s_node(spqr_node, pair, node_code, edge_code)
        for u,v,label in spqr_node[1].edge_iterator()
        for pair in [(u,v,label), (v,u,label)]
    )

def encode_p_root(spqr_node: SPQR_NODE, node_code: NODE_MAP, edge_code: EDGE_MAP):
    return min(
        encode_p_node(spqr_node, pair, node_code, edge_code)
        for u,v,label in spqr_node[1].edge_iterator()
        for pair in [(u,v,label), (v,u,label)]
    )

def encode_bcc(bcc_sage: sageall.Graph, node_code0: NODE_MAP, edge_code0: EDGE_MAP):
    tree = connectivity.TriconnectivitySPQR(bcc_sage).get_spqr_tree()

    def encode_from_center(center):
        node_code = node_code0.copy()
        edge_code = edge_code0.copy()
        # Encode non root nodes
        for fa,cur in reversed(list(tree.breadth_first_search(center, edges=True))):
            (type_fa, g_fa) = fa
            (type_cur, g_cur) = cur
            [cut_pair] = list(
                set(g_cur.edge_iterator()) & set(g_fa.edge_iterator())
            )

            rev_pair = (cut_pair[1], cut_pair[0], cut_pair[2])
            match type_cur:
                case 'S'|'Q':
                    code1 = encode_s_node(cur, cut_pair, node_code, edge_code)
                    code2 = encode_s_node(cur, rev_pair, node_code, edge_code)
                case 'P':
                    code1 = encode_p_node(cur, cut_pair, node_code, edge_code)
                    code2 = encode_p_node(cur, rev_pair, node_code, edge_code)
                case 'R':
                    code1 = encode_r_node(cur, cut_pair, node_code, edge_code)
                    code2 = encode_r_node(cur, rev_pair, node_code, edge_code)
                case _:
                    raise Exception("Unknown SPQR type")
            edge_code[cut_pair] = code1
            edge_code[rev_pair] = code2
    
        # Encode root node
        match center[0]:
            case 'S'|'Q':
                code = encode_s_root(center, node_code, edge_code)
            case 'P':
                code = encode_p_root(center, node_code, edge_code)
            case 'R':
                code = encode_r_root(center, node_code, edge_code)
            case _:
                raise Exception("Unknown SPQR type")
        
        return code
    
    min_code, min_center = min((
        (encode_from_center(center), center)
        for center in tree.center()
    ), key=lambda x: x[0])
    
    return tree, min_center, min_code

def encode_cc(cc_sage: sageall.Graph, node_code: NODE_MAP, edge_code: EDGE_MAP, pwl_iter:int=3):
    tree: sageall.Graph = connectivity.blocks_and_cuts_tree(cc_sage)

    for cur_iter in range(pwl_iter):
        bcc_codes = {
            node_cur: encode_bcc(cc_sage.subgraph(node_cur), node_code, edge_code)
            for (type_cur, node_cur) in tree.vertex_iterator()
            if type_cur=='B'
        }

        for (type_cur, node_cur) in tree.vertex_iterator():
            if type_cur=='C':
                code = ENCODING()
                code.append_bracket(C_BEGIN)

                code.append_node(node_code[node_cur])

                child_code = sorted(
                    bcc_codes[neigh[1]][2]
                    for neigh in tree.neighbors((type_cur, node_cur))
                )
                for i,c in enumerate(child_code):
                    code.append_char(BC_NODE_OFFSET+i)
                    code.append_node(c)

                code.append_bracket(C_END)
                node_code[node_cur] = int(hashlib.sha256(pickle.dumps(code)).hexdigest()[:8], 16)
    
    return tree, bcc_codes