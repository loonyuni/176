""" String Matching Problem Set
    Implement each of the functions bellow using the algorithms covered in class.
    You can construct additional functions and data structures but you should not
    change the functions APIs.
"""
################################################################################################

''' Helper Data Structures and methods '''


def zAlg(s):
    Z = [len(s)] + [0] * len(s)
    
    for i in xrange(1, len(s)):
        if s[i] == s[i-1]:
            Z[1] += 1
        else:
            break

    right, left = 0, 0
    if Z[1] > 0:
        right, left = Z[1], 1

    for j in xrange(2, len(s)):
        assert Z[j] == 0

        if j > right: #1
            for i in xrange(j, len(s)):
                if s[i] == s[i-j]:
                    Z[j] += 1
                else:
                    break
            right, left = j + Z[j] - 1, j
        else:#2
            beta = right-j+1
            Zj = Z[j-left]
            if beta > Zj:#2a
                Z[j] = Zj
            else:#2b
                match = 0
                for i in xrange(right+1, len(s)):
                    if s[i] == s[i-j]:
                        match += 1
                    else:
                        break
                left, right = j, right + match
                Z[j] = right - j + 1
    return Z


class SuffixTree(object):
    """ 
    using Ukkonen's algorithm 
    References: 
    [1] http://web.stanford.edu/~mjkay/gusfield.pdf 
    [2] http://stackoverflow.com/questions/9452701/ukkonens-suffix-tree-algorithm-in-plain-english
    [3] http://www.cs.jhu.edu/~langmea/resources/lecture_notes/suffix_trees.pdf
    """
    def __init__(self, s):
        s += '$'
        self.s = s
        self.root = root = self.Node(0, 0)
        root.parent = None
        root.children[s[0]] = end = self.Node(0, len(s), label=0)
        end.parent = root
        self.end = end
        assert len(end.children) == 0
        self.nodes = [root, end]
        self.ukkonen(root, end)

        self.skipExt = 0

    class Node(object):
        def __init__(self, start, length, label=None):
            self.children = {}
            self.parent = {}
            self.suffixLink = None
            self.length = length # length of substring on edge to node 
            self.start = start # index of where substring starts in string
            self.depth = None
            self.label = label

    def getPath(self, node, end = None):
        '''
        returns path from node to root
        if end, then returns path from node to end node
        '''
        s = self.s
        path = [node]
        curr = node
        while curr.parent:
            path.insert(0, curr.parent)
            if end:
                if curr.parent == end:
                    return path
            curr = curr.parent
        return path

    def getLabel(self, node, end = None):
        s = self.s
        path = self.getPath(node, end)
        label = []
        for p in path:
            label.append(s[p.start:p.start+p.length])
        return ''.join(label)

    def splitEdge(self, node, childC, length):
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
            if length > 0: # occurs in middle
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
        s = self.s
        skipExt = 1
        prevLeaf = end
        self.prevLeaf = end
        for i in xrange(0, len(s)-1): # phasing
            prevInternal = None
            c = s[i+1] 
            prevNode = prevLeaf.parent
            prevDepth = i + 1 - prevLeaf.start
            prevStart = prevLeaf.start
            for j in xrange(skipExt, i+2): 
                node, childrenc, depth = self.skipCount(prevNode, prevDepth, prevStart)
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
                        node.children[c] = prevLeaf = self.prevLeaf = self.Node(i+1, len(s)-i-1, label=j)
                        self.nodes.append(prevLeaf)
                        node.children[c].parent = node
                    else: 
                        break 
                else: # in between nodes
                    assert childrenc is not None
                    child = node.children[childrenc]
                    if s[child.start + depth] != c:
                        mid = prevNode = self.splitEdge(node, childrenc, depth)
                        prevDepth, prevStart = 0, i+1
                        mid.children[c] = prevLeaf = self.prevLeaf= self.Node(i+1, len(s)-i-1, label=j)
                        self.nodes.append(prevLeaf)
                        mid.children[c].parent = mid
                        if prevInternal is not None:
                            assert prevInternal.suffixLink is None
                            prevInternal.suffixLink = mid
                        prevInternal = mid
                    else:
                        assert prevInternal is None
                        break 
                skipExt = self.skipExt = max(skipExt, j+1)
            assert prevInternal is None

    # def suffixArray(self):
    #     sa = []
    #     def visit(node):
    #         if len(node.children) == 0:
    #             sa.append(node.length)
    #         for c, child in sorted(n.children.iteritems()):
    #             print child.children
    #             visit(child)
    #     return sa

    # def addWord(self, word, end):
    #     s = word + '#'
    #     skipExt = self.skipExt
    #     prevLeaf = end
    #     for i in xrange(0, len(s)-1): # phasing
    #         prevInternal = None
    #         c = s[i+1] 
    #         prevNode = prevLeaf.parent
    #         prevDepth = i + 1 - prevLeaf.start
    #         prevStart = prevLeaf.start
    #         for j in xrange(skipExt, i+2): 
    #             print prevDepth
    #             node, childrenc, depth = self.skipCount(prevNode, prevDepth, prevStart) # aciveNode, activeEdge, activeLength
    #             assert node is not None
    #             if depth == 0: # ended
    #                 if prevInternal is not None:
    #                     assert prevInternal.suffixLink is None
    #                     prevInternal.suffixLink = node
    #                     prevInternal = None
    #                 if c not in node.children: # new leaf
    #                     prevNode = node
    #                     prevDepth = 0
    #                     prevStart = i+1
    #                     node.children[c] = prevLeaf = self.Node(i+1, len(s)-i-1, label=j)
    #                     self.nodes.append(prevLeaf)
    #                     node.children[c].parent = node
    #                 else: 
    #                     break 
    #             else: # in between nodes
    #                 assert childrenc is not None
    #                 child = node.children[childrenc]
    #                 if s[child.start + depth] != c:
    #                     mid = prevNode = self.splitEdge(node, childrenc, depth)
    #                     prevDepth, prevStart = 0, i+1
    #                     mid.children[c] = prevLeaf = self.Node(i+1, len(s)-i-1, label=j)
    #                     self.nodes.append(prevLeaf)
    #                     mid.children[c].parent = mid
    #                     if prevInternal is not None:
    #                         assert prevInternal.suffixLink is None
    #                         prevInternal.suffixLink = mid
    #                     prevInternal = mid
    #                 else:
    #                     assert prevInternal is None
    #                     break 
    #             skipExt = max(skipExt, j+1)
    #         assert prevInternal is None



class GeneralizedST(object):
    '''
    Creates a generalized suffix tree in O(n^2) times
    Refernces:
    [1] http://www.cs.jhu.edu/~langmea/resources/lecture_notes/suffix_trees.pdf
    '''
    class Node(object):
        def __init__(self, label):
            self.label = label
            self.children = {}
            self.color = ''

    def __init__(self, s1, s2):
        s1 += '#'
        s2 += '$'
        self.root = self.Node('')
        self.root.children[s1[0]] = self.Node(s1)
        for i in xrange(1, len(s1)):
            curr = self.root
            j = i
            while j < len(s1):
                if s1[j] in curr.children:
                    child = curr.children[s1[j]]
                    label = child.label
                    k = j+1
                    while k-j < len(label) and s1[k] == label[k-j]:
                        k += 1
                    if k-j == len(label):
                        curr = child
                        j = k
                    else:
                        old = label[k-j]

                        new = s1[k]   
                        split = self.Node(label[:k-j])
                        split.children[new] = self.Node(s1[k:])
                        child.label = label[k-j:]
                        curr.children[s1[j]] = split
                else:
                    curr.children[s1[j]] = self.Node(s1[j:])
        # start from root and add s2            
        for i in xrange(0, len(s2)):
            curr = self.root
            j = i
            while j < len(s2):
                if s2[j] in curr.children:
                    child = curr.children[s2[j]]
                    label = child.label
                    k = j+1
                    while k-j < len(label) and s2[k] == label[k-j]:
                        k += 1
                    if k-j == len(label):
                        curr = child
                        j = k
                    else:
                        old = label[k-j]
                        new = s2[k]   
                        split = self.Node(label[:k-j])
                        split.children[new] = self.Node(s2[k:])
                        child.label = label[k-j:]
                        curr.children[s2[j]] = split
                else:
                    curr.children[s2[j]] = self.Node(s2[j:])

def naiveSuffixArray(s):
    # too lazy to radix sort
    sa = sorted([(s[i:], i) for i in xrange(0, len(s))])
    return map(lambda x: x[1], sa)

def lcp(i, sa, s):
    assert i  >= 1
    assert i < len(sa)
    suff1 = s[sa[i]:]
    suff2 = s[sa[i+1]:]
    return lcp_helper(suff1, suff2)

def lcp_helper(suff1, suff2):
    n = min(len(suff1), len(suff2))
    for i in xrange(0, n):
        if suff1[i] != suff2[i]:
            return i
    return n


################################################################################################

def multiple_pattern_match(patterns_list):
    """
    Given: A string Text and a collection of strings Patterns.

    Return: All starting positions in Text where a string from Patterns appears as a substring.

    Use one or more of the algorithms you learned about in class.
    Do NOT use python regex utilities

    Input:
        AATCGGGTTCAATCGGGGT
        ATCG
        GGGT

    Output:
        1 4 11 15

    >>> multiple_pattern_match(["AATCGGGTTCAATCGGGGT", "ATCG", "GGGT"])
    [1, 4, 11, 15]

    """
    T = patterns_list.pop(0)
    results = []
    matches = []
    for pattern in patterns_list:
        s = pattern + '$' + T
        Z = zAlg(s)
        for i in xrange(len(pattern)+1, len(s)):
            if Z[i] >= len(pattern):
                matches.append(i-(len(pattern)+1))          
    return sorted(matches)



def longest_repeat(s):
    """
    Find the longest repeat in a string.

    Given: A string Text.

    Return: A longest repeat in Text, i.e., a longest substring of Text that appears in Text more than once.

    Input:
        ATATCGTTTTATCGTT

    Output:
        TATCGTT

    >>> longest_repeat("ATATCGTTTTATCGTT")
    'TATCGTT'
    """
    #find deepest internal node in suffix tree
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
    result = []
    for leaf in compare:
        curr = leaf[0]
        if curr.parent: 
            result.append(st.getLabel(curr.parent))

    return max(result, key = len)

    # sa = naiveSuffixArray(s)
    # result = ''
    # for i in xrange(1, len(sa)-1):
    #     lcpLen = lcp(i, sa, s)
    #     if lcpLen > len(result):
    #         result = s[sa[i]:sa[i]+lcpLen]

    # return result


def longest_common_substring(s1, s2):
    """
    Find the longest substring shared by two strings.

    Given: Strings Text1 and Text2.

    Return: The longest substring that occurs in both Text1 and Text2.

    Input:
        TCGGTAGATTGCGCCCACTC
        AGGGGCTCGCAGTGTAAGAA

    Output:
        AGA

    >>> longest_common_substring("TCGGTAGATTGCGCCCACTC", "AGGGGCTCGCAGTGTAAGAA")
    'AGA'
    """
    # if tree, make generalized ST and find deepest internal node that has both 
    # terminator strings in it's offspring

    s = s1 + '#' + s2
    sa = naiveSuffixArray(s)


    lcs = ''

    for i in xrange(1, len(s)-1):
        # suffixes both from s1
        if sa[i] < len(s1) and sa[i+1] < len(s1):
            continue
        # suffixes both from s2
        if sa[i] > len(s1) and sa[i+1] > len(s1):
            continue
        lcpLen = lcp(i, sa, s)
        if lcpLen > len(lcs):
            lcs = s[sa[i]:sa[i]+lcpLen]
    return lcs

def longest_palindromic_substring(s):
    """
    Find the longest palindromic substring.

    Given: Strings Text

    Return: The longest substring of the Text that is also a palindrome

    Input:
        GCGTTCAACTCGG

    Output:
        TCAACT

    >>> longest_palindromic_substring("GCGTTCAACTCGG")
    'TCAACT'
    """
    # if my suffix tree wasn't a pain in the butt, you would create a generalized
    # suffix tree with s, and reversed s --> s#s[::-1]$ 
    return longest_common_substring(s, s[::-1])


def color_dfs (root, st): 
    if len(root.children) == 0:
        root.color = root.label[-1]
        return root.color
    else:
        root.color = ''.join([color_dfs(child, st) for child in root.children.values()])
        root.color = ''.join(set(root.color))
        return root.color

def shortest_non_shared_substring(s1, s2):
    """
    Find the shortest substring of one string that does not appear in another string.
    If there are multiple such strings, output the one lexicographically smallest.

    Given: Strings Text1 and Text2.

    Return: The shortest substring of Text1 that does not appear in Text2.

    Input:
        CCAAGCTGCTAGAGG
        CATGCTGGGCTGGCT

    Output:
        AA

    >>> shortest_non_shared_substring("CCAAGCTGCTAGAGG", "CATGCTGGGCTGGCT")
    'AA'
    """
    # kind of like problem 2 of written hw
    # make generalized suffix tree
    # take path from root to first prefixes of last edges that only has either #, or $ in children
    # then return the shortest of these 
    st = GeneralizedST(s1, s2)
    color_dfs(st.root, st)

    depths = []
    leafs = []
    stack = []
    start = st.root
    start.depth = 0
    stack.append(start)
    min_root = [None, float("inf")]

    while stack:
        curr = stack.pop()
        if (not min_root[0] and len(curr.label) > 0) or (len(curr.color) == 1 and len(curr.label) < min_root[1]):
            min_root[0] = curr
            min_root[1] = len(curr.label)
        for childKey in curr.children:
            child = curr.children[childKey]            
            stack.append(child)
    return min_root[0].label


def longest_k_repeat_substring(k, s):
    """
    Find the longest exactly k-times repeated substring

    Given: Strings Text

    Return: The longest substring of the Text that is repeated exactly k times, break ties lexicographically

    Input:
        3
        AAACCACACACAAA

    Output:
        CACA

    >>> longest_k_repeat_substring(3, "AAACCACACACAAA")
    'CACA'
    """
    # want to find the deepest internal node k -1 total off spring

    if k == 2:
        return longest_repeat(s)

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
    result = []
    for leaf in compare:
        curr = leaf[0]
        if curr.parent:
            currParent = curr.parent
            currParentOffSpring = len(currParent.children)
            while currParentOffSpring < k:
                currParentOffSpring += 1
                if currParentOffSpring == k:
                    if curr.parent.parent == None:
                        break
                    result.append(st.getLabel(curr.parent.parent))
                curr = currParent.parent
        else: 
            break
    return max(result, key=len)


def shortest_non_substring(s):
    """
    Find the shortest string that is not a substring
    If there are multiple such strings, output the one that is lexicographically smallest

    Given: Strings Text

    Return: The shortest string that is not a substring of the text

    Input:
        GCGTTCAACTCGG

    Output:
        AG

    >>> shortest_non_substring("GCGTTCAACTCGG")
    'AG'
    """
    # want to do a mutual recursion 
    # go down tree and look for missing alphabet outgoing node 
    # and add string so far on that node to the missing node

    st = SuffixTree(s)
    stack = [st.root]
    s += '$'
    alphabet = set(s)
    results = []
    while stack:
        curr = stack.pop()
        addLetter = alphabet-set(curr.children.keys())
        if len(addLetter) > 0:
            results.append(list(addLetter)[0]+st.getLabel(curr))
        for childC in curr.children:
            child = curr.children[childC]
            stack.append(child)
    results = sorted([valid for valid in results if '$' not in valid], key = lambda item: (len(item), item ))
    if results:
        return results[0]
    else:
        return None



