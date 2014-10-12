# Author: Rohin Shah

def direct_comparison(string, i, j):
    """
    Compare characters starting at position i to those starting at
    position j and return the number of matching characters.

    >>> direct_comparison(' axyaxz$xaxyaxyaxz', 1, 4)
    2
    >>> direct_comparison(' axyaxz$xaxyaxyaxz', 1, 3)
    0
    >>> direct_comparison(' axyaxz$xaxyaxyaxz', 1, 9)
    5
    >>> direct_comparison(' axyaxz$xaxyaxyaxz', 1, 12)
    6
    >>> direct_comparison(' axyaxz$xaxyaxyaxz', 3, 11)
    3
    """
    assert i < j
    count = 0
    while j < len(string) and string[i] == string[j]:
        count += 1
        i += 1
        j += 1
    return count

def z_algorithm(string, debug=True, latex=False):
    """
    Computes the Z-scores for the string using the Z-algorithm.

    >>> z_algorithm('aaab')
    Direct comparison gives Z_2 = 2 with a Z-box of aa.
    <BLANKLINE>
    k = 3:  We have l = 2, r = 3
    Case 2:  k <= r, j = k - l + 1 = 2, beta = a, Z_j = 2
    Case 2b: Z_j > |beta|, so Z_k = |beta| = 1
    |---A---|           
    |   |---A---|       
    |   |-B-|   |       
    |   |   |-B-|       
    |-B-|   |   |       
      a   a   a   b
      1   2   3   4   
         j,l k,r        
    <BLANKLINE>
    k = 4:  We have l = 2, r = 3
    Case 1:  k > r. Direct comparison gives Z_k = 0.
    [None, None, 2, 1, 0]

    >>> z_algorithm('abaaba', False)
    [None, None, 0, 1, 3, 0, 1]

    >>> z_algorithm('axyaxz$xaxyaxyaxz', False)
    [None, None, 0, 0, 2, 0, 0, 0, 0, 5, 0, 0, 6, 0, 0, 2, 0, 0]

    >>> z_algorithm('aaaaa', False)
    [None, None, 4, 3, 2, 1]
    """
    size = len(string)

    # Since we 1-index strings, add a dummy character (a space) to the
    # beginning, which will be ignored since it is at index 0
    string = ' ' + string

    # State for Z algorithm: l, r, Z-scores
    l, r = 0, 0
    zscores = [ None ] * (size + 1)

    # Leave zscores[0] and zscores[1] as None (since Z_0 and Z_1 are
    # not defined).

    # Compute zscores[2] by direct character comparison
    zscores[2] = direct_comparison(string, 1, 2)
    if zscores[2] != 0:
        l, r = 2, 1 + zscores[2]
    debugger(debug, latex, 'base case', zscores[2], string)

    debugger(debug, latex, 'loop beginning')
    for k in range(3, size + 1):

        if k > r:
            zscores[k] = direct_comparison(string, 1, k)
            debugger(debug, latex, '1', k, l, r, zscores[k], string)
            if zscores[k] != 0:
                l, r = k, k + zscores[k] - 1

        else:
            j = k - l + 1
            beta_size = r - k + 1
            zj = zscores[j]

            if zj < beta_size:
                zscores[k] = zj
                debugger(debug, latex, '2a', k, l, r, zscores[k], string, j, zj)
            elif zj > beta_size:
                zscores[k] = beta_size
                debugger(debug, latex, '2b', k, l, r, zscores[k], string, j, zj)
            else:
                extra = direct_comparison(string, beta_size + 1, r + 1)
                zscores[k] = beta_size + extra
                debugger(debug, latex, '2c', k, l, r, zscores[k], string, j, zj)
                l, r = k, k + zscores[k] - 1

    debugger(debug, latex, 'loop end')

    return zscores

def exact_match(pattern, text, debug=True, latex=False):
    """
    Returns indices of all occurrences of pattern in text.

    >>> exact_match('axyaxz', 'xaxyaxyaxz', False)
    [5]
    >>> exact_match('axy', 'xaxyaxyaxz', False)
    [2, 5]
    >>> exact_match('axyz', 'xaxyaxyaxz', False)
    []
    """
    if '$' in pattern or '$' in text:
        raise ValueError('Cannot have $ in the pattern or text')

    # Run Z-algorithm on P$T.
    zscores = z_algorithm(pattern + '$' + text, debug, latex)

    # Return the indices which have a Z-score equal to |pattern|
    psize = len(pattern)
    return [i - psize - 1 for i in range(len(zscores)) if zscores[i] == psize]


# Code to print out debug information
def debugger(debug, latex, case, *args):
    if not debug:
        return

    if case == 'loop beginning':
        if latex:
            print '\\begin{enumerate}'
    elif case == 'loop end':
        if latex:
            print '\\end{enumerate}'
    elif case == 'base case':
        print_base_case(latex, *args)
    else:
        # One of the loop iterations
        if latex:
            print '\\begin{minipage}{\\linewidth}'

        print_iteration(latex, *args[:5])

        if case == '1':
            print_case1(latex, *args)
        else:
            print_case2(latex, *args)

            if case == '2a':
                print_case2a(latex, *args)
            elif case == '2b':
                print_case2b(latex, *args)
            else:
                print_case2c(latex, *args)

            print_ascii_box(latex, case, *args)

        if latex:
            print '\\end{minipage}'


def zbox_str(latex, k, zk, string):
    if latex and zk != 0:
        return ' with a $Z$-box of \\texttt{{{0}}}'.format(string[k:k+zk])
    elif zk != 0:
        return ' with a Z-box of {0}'.format(string[k:k+zk])
    else:
        return ''

def print_base_case(latex, z2, string):
    zbox = zbox_str(latex, 2, z2, string)
    if latex:
        print 'Direct comparison gives $Z_2 = {0}${1}.'.format(z2, zbox)
    else:
        print 'Direct comparison gives Z_2 = {0}{1}.'.format(z2, zbox)


def print_iteration(latex, k, l, r, zk, string):
    if latex:
        print '\n\\item $k = {0}$:  We have $l = {1}$, $r = {2}$\\\\'.format(k, l, r)
    else:
        print '\nk = {0}:  We have l = {1}, r = {2}'.format(k, l, r)


def print_case1(latex, k, l, r, zk, string):
    zbox = zbox_str(latex, k, zk, string)
    if latex:
        print 'Case 1:  $k > r$. Direct comparison gives $Z_k = {0}${1}.\\\\'.format(zk, zbox)
    else:
        print 'Case 1:  k > r. Direct comparison gives Z_k = {0}{1}.'.format(zk, zbox)


def print_case2(latex, k, l, r, zk, string, j, zj):
    beta = string[k:r+1]
    if latex:
        print 'Case 2:  $k \\le r$, $j = k - l + 1 = {0}$, $\\beta = \\texttt{{{1}}}$, $Z_j = {2}$\\\\'.format(j, beta, zj)
    else:
        print 'Case 2:  k <= r, j = k - l + 1 = {0}, beta = {1}, Z_j = {2}'.format(j, beta, zj)


def print_case2a(latex, k, l, r, zk, string, j, zj):
    if latex:
        print 'Case 2a: $Z_j < |\\beta|$, so $Z_k = Z_j = {0}$'.format(zj)
    else:
        print 'Case 2a: Z_j < |beta|, so Z_k = Z_j = {0}'.format(zj)


def print_case2b(latex, k, l, r, zk, string, j, zj):
    beta = string[k:r+1]
    if latex:
        print 'Case 2b: $Z_j > |\\beta|$, so $Z_k = |\\beta| = {0}$'.format(len(beta))
    else:
        print 'Case 2b: Z_j > |beta|, so Z_k = |beta| = {0}'.format(len(beta))


def print_case2c(latex, k, l, r, zk, string, j, zj):
    if latex:
        print 'Case 2c: $Z_j = |\\beta|$, so $Z_k$ is at least $|\\beta|$.  Using direct comparison, $Z_k = {0}$.'.format(zk)
    else:
        print 'Case 2c: Z_j = |beta|, so Z_k is at least |beta|.  Using direct comparison, Z_k = {0}.'.format(zk)


def print_ascii_box(latex, case, k, l, r, zk, string, j, zj):
    if latex:
        print '\\begin{verbatim}'

    factor = 4
    line_size = factor * len(string)

    vertical_line_indices = []
    def make_ascii_box_line(start_base_index, end_base_index, middle_char):
        start = (start_base_index - 1) * factor
        end = end_base_index * factor
        vertical_line_indices.extend([start, end])

        line = [' ' for i in range(line_size)]
        line[start+1:end] = ['-'] * (end - start - 1)
        line[(start + end) // 2] = middle_char
        for index in vertical_line_indices:
            if line[index] == ' ' or line[index] == '-':
                line[index] = '|'
        return ''.join(line)

    print make_ascii_box_line(1, r - l + 1, 'A')
    print make_ascii_box_line(l, r, 'A')
    print make_ascii_box_line(j, r - l + 1, 'B')
    print make_ascii_box_line(k, r, 'B')
    if case == '2a' and zj != 0:
        print make_ascii_box_line(j, j + zj - 1, 'G')
        print make_ascii_box_line(k, k + zj - 1, 'G')
        print make_ascii_box_line(1, zj, 'G')
        print make_ascii_box_line(l, l + zj - 1, 'G')
    elif case != '2a':
        print make_ascii_box_line(1, r - k + 1, 'B')
    spacing = ' ' * (factor - 1)
    print ' ' * (factor // 2) + spacing.join([x for x in string[1:]])

    nums = [str(x) for x in range(1, len(string))]
    padded_nums = [x + (' ' * max(0, factor - len(x))) for x in nums]
    print ' ' * (factor // 2) + ''.join(padded_nums)

    # In case 2, we know l < k <= r
    # Since j = k - l + 1 and l >= 2, we have j < k.
    # So, j could be equal to l, and k could be equal to r.
    # No other pair of indices could be equal.

    last_line = [' '] * line_size
    jpos, lpos, kpos, rpos = [factor * x - (factor // 2) for x in [j, l, k, r]]
    if jpos == lpos:
        last_line[jpos-1:jpos+2] = 'j,l'
    else:
        last_line[jpos] = 'j'
        last_line[lpos] = 'l'
    if kpos == rpos:
        last_line[kpos-1:kpos+2] = 'k,r'
    else:
        last_line[kpos] = 'k'
        last_line[rpos] = 'r'

    print ''.join(last_line)

    if latex:
        print '\n\\end{verbatim}'
