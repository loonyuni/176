class SuffixTree(object):
    
    class Node(object):
        """ Suffix tree node.  Contains startset, length of substring from T
            labeling the incoming edge.  Also contains outgoing edges, and some
            (hopefully O(1) space) extra annotations. """
        
        def __init__(self, start, length, id=None):
            self.start = start    # startset into T of substring on edge leading into this node
            self.length = length      # length of substring on edge leading into this node
            self.suffixLink = None # suffix link from this node
            self.id = id      # id; for leaf nodes, this is the suffix
            self.children = {}     # childrengoing edges; characters x nodes
            self.parent = None   # extra per-node info
            self.key = ''
            self.depth = 0

    def addKey(self): # dfs to add self.key
		stack = []
		start = self.root
		stack.append(start)

		while stack:
			curr = stack.pop()
			for childKey in curr.children:
				child = curr.children[childKey]
				child.key = childKey
				print childKey
				stack.append(child)


    def splitEdge(self, node, childC, length):
        """ Create a new node in the middle of the edge given by node, childC """
        assert length > 0
        assert childC in node.children
        child = node.children[childC]
        assert length < child.length
        mid = self.Node(child.start, length)
        self.nodes.append(mid)
        mid.parent = node
        child.parent = mid
        node.children[childC] = mid
        child.start += length
        child.length -= length
        mid.children[self.s[child.start]] = child
        return mid
    
    def path(self, p):
        i = 0
        curr, s = self.root, self.s
        while i < len(p):
            childC = p[i]
            if childC not in curr.children:
                return None, None, None
            child = curr.children[childC]
            assert childC == s[child.start]
            i += 1
            j = 1
            while j < child.length and i < len(p):
                if p[i] != s[child.start + j]:
                    return None, None, None
                j += 1 
                i += 1 
            if j == child.length:
                curr = child 
            else:
                assert i == len(p)
                return (curr, childC, j) 
        return (curr, None, 0)
    
    def skipCount(self, v, length, start):
        if v.length != 0 and v.suffixLink is None:
            start = v.start
            length += v.length
            v = v.parent
            assert v.length == 0 or v.suffixLink is not None
        if v.length == 0:
            assert length > 0
            length -= 1
            start += 1
            if length > 0:
                node, childC, depth = self.path(self.s[start:start+length])
                assert node is None or depth == 0 or depth < node.children[childC].length
                return node, childC, depth
            else:
                return v, None, 0
        else:
            assert v.suffixLink is not None
            sv = curr = v.suffixLink
            assert len(sv.children) > 0
            if length == 0:
                return sv, None, 0
            assert start is not None
            i = start
            while length > 0:
                childC = self.s[i]
                next = curr.children[childC]
                if next.length < length:
                    length -= next.length
                    i += next.length
                    curr = next
                elif next.length == length:
                    return next, None, 0
                else:
                    assert length < curr.children[childC].length
                    return curr, childC, length
            assert False
    
    def ukkonen(self, root, end):
        """ Perform all phases and extensions of Ukkonen's algorithm """
        s = self.s
        skipExt = 1
        prevLeaf = end
        for i in xrange(0, len(s)-1): # phasing
            prevInternal = None
            c = s[i+1] 
            prevNode = prevLeaf.parent
            prevDepth = i + 1 - prevLeaf.start
            prevStart = prevLeaf.start
            for j in xrange(skipExt, i+2): 
                node, childC, depth = self.skipCount(prevNode, prevDepth, prevStart)
                assert node is not None
                if depth == 0: # ended
                    if prevInternal is not None:
                        assert prevInternal.suffixLink is None
                        prevInternal.suffixLink = node
                        prevInternal = None
                    if c not in node.children: # new leaf
                        prevNode = node
                        prevDepth = 0
                        prevStart = i+1
                        node.children[c] = prevLeaf = self.Node(i+1, len(s)-i-1, id=j)
                        self.nodes.append(prevLeaf)
                        node.children[c].parent = node
                    else: 
                        break 
                else: # in between nodes
                    assert childC is not None
                    child = node.children[childC]
                    if s[child.start + depth] != c:
                        mid = prevNode = self.splitEdge(node, childC, depth)
                        prevDepth, prevStart = 0, i+1
                        mid.children[c] = prevLeaf = self.Node(i+1, len(s)-i-1, id=j)
                        self.nodes.append(prevLeaf)
                        mid.children[c].parent = mid
                        if prevInternal is not None:
                            assert prevInternal.suffixLink is None
                            prevInternal.suffixLink = mid
                        prevInternal = mid
                    else:
                        assert prevInternal is None
                        break 
                skipExt = max(skipExt, j+1)
            assert prevInternal is None

    def __init__(self, s, toDot=None, sanity=False):
        """ Ukkonen's algorithm to build suffix tree """
        s += '$'
        self.s = s
        self.root = root = self.Node(0, 0)
        root.parent = None
        root.children[s[0]] = end = self.Node(0, len(s), id=0)
        end.parent = root
        assert len(end.children) == 0
        self.nodes = [root, end]
        self.ukkonen(root, end)
        self.addKey()


s = 'BANANA'
st = SuffixTree(s)
depths = []
leafs = []
stack = []
start = st.root
start.depth = 0
stack.append(start)

while stack:
    curr = stack.pop()
    if len(curr.children) == 0: #leaf node
        leafs.append(curr)
        depths.append(curr.depth)
    for childKey in curr.children:
        child = curr.children[childKey]
        child.depth = child.length
        child.depth += curr.length 
        stack.append(child)

compare = sorted(zip(leafs, depths), key = lambda x: x[1])[::-1]

second = compare[2]
child = second[0]
path = [child]
while child.parent:
	path.insert(0, child.parent)
	child = child.parent
print path

label = []
for node in path:
	print '--------'
	print node.key
	print node.start
	print node.length
	label.append(s[node.start:node.start+node.length])

print ''.join(label)

