class SuffixTree(object):
    
    class Node(object):
        def __init__(self, lab):
            self.lab = lab # label on path leading to this node
            self.out = {}  # outgoing edges; maps characters to nodes
    
    def __init__(self, s):
        """ Make suffix tree, without suffix links, from s in quadratic time
            and linear space """
        s += '$'
        self.root = self.Node(None)
        self.root.out[s[0]] = self.Node(s) # trie for just longest suf
        for i in xrange(1, len(s)):
            cur = self.root
            j = i
            while j < len(s):
                print s[j]
                if s[j] in cur.out:
                    child = cur.out[s[j]]
                    lab = child.lab
                    k = j+1 
                    print '-----'
                    print k-j
                    print lab
                    while k-j < len(lab) and s[k] == lab[k-j]:
                        k += 1
                    if k-j == len(lab):
                        cur = child # we exhausted the edge
                        j = k
                    else:
                        cExist, cNew = lab[k-j], s[k]
                        mid = self.Node(lab[:k-j])
                        mid.out[cNew] = self.Node(s[k:])
                        mid.out[cExist] = child
                        child.lab = lab[k-j:]
                        # mid becomes new child of original parent
                        cur.out[s[j]] = mid
                else:
                    cur.out[s[j]] = self.Node(s[j:])
    

st = SuffixTree('abcab')

curr = st.root
print curr.lab
a = curr.out['a']
print a.lab
print a.out

n = a.out['c']
print n.lab

