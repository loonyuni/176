""" BWT and FM index Problem Set
    Implement each of the functions bellow using the algorithms covered in class.
    You can construct additional functions and data structures but you should not
    change the functions APIs.
"""

def bwt(s):
    """
    (10 points)

    Given:
        s = a string text

    Return: the string BWT transform of s

    Input:
        banana$
    Output:
        annb$aa

    > bwt("banana$")
    'annb$aa'

    """
    
    rotations = sorted([(s*2)[i:i+len(s)] for i in range(0, len(s))])
    return ''.join([array[-1] for array in rotations])


def ibwt(s):
    """
    (10 points)

    Given:
        s = a string text

    Return: The inverse BWT transform of s

    Input:
        annb$aa

    Output:
        banana$

    > ibwt("annb$aa")
    'banana$'

    """

    total = [''] * len(s)
    for i in range(len(s)):
        
        total = sorted([s[i] + total[i] for i in range(len(s))])
    result = [row for row in total if row[-1] == '$']
    return result[0]

def exact_match(s, p):
    """
    (10 points)

    Given:
        s = a string text
        p = a string pattern

    Return: The positions of pattern p in s,
			and the sequence of sp,ep pairs
			resulting from the FM-index
			calculation

    Input:
        banana
        ana

    Output:
        ([2, 4], [(1, 3), (5, 6), (2, 3)])

    > exact_match("banana", "ana")
    ([2, 4], [(1, 3), (5, 6), (2, 3)])

    """

    


def bowtie(s, p, q, n, k):
    """
    (20 points)

    Given:
        s = a string text
        p = a string pattern
        q = a quality score array for p
        n = maximum number of mismatches
        k = maximum number of backtracks

    Return: The aligned starting position of s in p

    Input:
        GATTACA
        AGA
        [40, 15, 35]
        2
        2

    Output:
        5

    Note: Only allow A<->T and G<->C mismatches

    > bowtie('GATTACA', 'AGA', [40, 15, 35], 2, 2)
    5

    """
    raise NotImplementedError()

class FM(object):
    def __init__(self, string):
        self.bwt = bwt(string)
        self.first, self.last = self.getFL()

    def getFL():
        bwt = self.bwt
        ranks = []
        lCount = {}
        for c in bwt:
            if c not in lCount:
                lCount[c] = 0
            ranks.append(lCount[c])
            lCount[c] += 1

        occ = {}


