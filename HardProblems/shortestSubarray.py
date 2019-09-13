# ~ https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/
# ~ 862. Shortest Subarray with Sum at Least K
# ~ Hard

# ~ 512

# ~ 13

# ~ Favorite

# ~ Share
# ~ Return the length of the shortest, non-empty, contiguous subarray of A with sum at least K.

# ~ If there is no non-empty subarray with sum at least K, return -1.

 

# ~ Example 1:

# ~ Input: A = [1], K = 1
# ~ Output: 1
# ~ Example 2:

# ~ Input: A = [1,2], K = 4
# ~ Output: -1
# ~ Example 3:

# ~ Input: A = [2,-1,2], K = 3
# ~ Output: 3
import collections
def shortestSubarray(A, K):
        N = len(A)
        B = [0] * (N + 1)
        for i in range(N): B[i + 1] = B[i] + A[i]
        d = collections.deque()
        res = N + 1
        for i in xrange(N + 1):
            while d and B[i] - B[d[0]] >= K: res = min(res, i - d.popleft())
            while d and B[i] <= B[d[-1]]: d.pop()
            d.append(i)
        return res if res <= N else -1

A = [1,2,3]
K = 4

print(shortestSubarray(A, K))

