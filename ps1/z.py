def z(s):
    ''' Use Z-algorithm to preprocess given string.  See
        Gusfield for complete description of algorithm. '''
    
    Z = [len(s)] + [0] * len(s)
    assert len(s) > 1
    
    # Initial comparison of s[1:] with prefix
    for i in xrange(1, len(s)):
        if s[i] == s[i-1]:
            Z[1] += 1
        else:
            break
    
    r, l = 0, 0
    if Z[1] > 0:
        r, l = Z[1], 1
    
    for k in xrange(2, len(s)):
        assert Z[k] == 0
        if k > r:
            # Case 1
            for i in xrange(k, len(s)):
                if s[i] == s[i-k]:
                    Z[k] += 1
                else:
                    break
            r, l = k + Z[k] - 1, k
        else:
            # Case 2
            # Calculate length of beta
            nbeta = r - k + 1
            Zkp = Z[k - l]
            if nbeta > Zkp:
                # Case 2a: Zkp wins
                Z[k] = Zkp
            else:
                # Case 2b: Compare characters just past r
                nmatch = 0
                for i in xrange(r+1, len(s)):
                    if s[i] == s[i - k]:
                        nmatch += 1
                    else:
                        break
                l, r = k, r + nmatch
                Z[k] = r - k + 1
    return Z

def zMatch(p, t):
    s = p + "$" + t

    Z = z(s)
    occurrences = []
    for i in xrange(len(p) + 1, len(s)):
        if Z[i] >= len(p):
            occurrences.append(i - (len(p) + 1))
    return occurrences

print zMatch("ATCG", "AATCGGGTTCAATCGGGGT")