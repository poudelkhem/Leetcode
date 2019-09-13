329. Longest Increasing Path in a Matrix
Hard

1149

21

Favorite

Share
Given an integer matrix, find the length of the longest increasing path.

From each cell, you can either move to four directions: left, right, up or down. You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).

Example 1:

Input: nums = 
[
  [9,9,4],
  [6,6,8],
  [2,1,1]
] 
Output: 4 
Explanation: The longest increasing path is [1, 2, 6, 9].

The idea is very simple and it's my first sharing. For each node, we record the maximum benefits of robbing it or not separately, [not rob, rob]. Consider the current node, if you decide to rob it, you cannot rob its direct children, thus, you can obtain root.val+left[not rob]+right[not rob]; while if you decide not to rob it, it doesn't matter whether you rob or not rob its children, so just find the maximum values from the children, thus you obtain max(left[rob], left[not rob])+max(right[rob], right[not rob]). Finally, return the maximum value. Hope this helps.


class Solution:
    def longestIncreasingPath(self, matrix):
        
        if not matrix:
            return 0
        
        h, w = len(matrix), len(matrix[0])
        
        dp = [[0] * w for _ in range(h)]
        
        def dfs(i, j):
            if not dp[i][j]:
                adj = [dfs(i+di, j+dj) for di, dj in zip([-1,1,0,0], [0,0,-1,1]) if 0<=i+di<h and 0<=j+dj<w and matrix[i+di][j+dj] < matrix[i][j]]
                dp[i][j] = 1 + (max(adj) if adj and adj[0] else 0)
            return dp[i][j]
                    
        return max(dfs(i, j) for i in range(h) for j in range(w))
    
Next challenges:
Reconstruct Itinerary
Lonely Pixel II
Network Delay Time




65. Valid Number
Hard

479

3503

Favorite

Share
Validate if a given string can be interpreted as a decimal number.

Some examples:
"0" => true
" 0.1 " => true
"abc" => false
"1 a" => false
"2e10" => true
" -90e3   " => true
" 1e" => false
"e3" => false
" 6e-1" => true
" 99e2.5 " => false
"53.5e93" => true
" --6 " => false
"-+3" => false
"95a54e53" => false


337. House Robber III
Medium

1701

35

Favorite

Share
The thief has found himself a new place for his thievery again. There is only one entrance to this area, called the "root." Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that "all houses in this place forms a binary tree". It will automatically contact the police if two directly-linked houses were broken into on the same night.

Determine the maximum amount of money the thief can rob tonight without alerting the police.

Example 1:

Input: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1



def rob(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    return max(self.helper(root))
    def helper(self, root):
        if root is None: 
            return (0,0)
        lc=self.helper(root.left)
        rc=self.helper(root.right)
        return (max(lc)+max(rc), root.val+lc[1]+rc[1])
        
 44. Wildcard Matching
Hard

1218

78

Favorite

Share
Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*'.

'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).
The matching should cover the entire input string (not partial).

Note:

s could be empty and contains only lowercase letters a-z.
p could be empty and contains only lowercase letters a-z, and characters like ? or *.
Example 1:

Input:
s = "aa"
p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".

class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        if not p:
            return not s
        
        level = {0}
        for i, c in enumerate(p):
            
            if not level:
                return False
            
            if c == "*":
                level = set(range(min(level), len(s)+1))
                #print(level)
            else:
                level = {j+1 for j in level if j < len(s) and (s[j] == c or c == "?")}
        
        return len(s) in level
        
        
        Next challenges:
Regular Expression Matching

10. Regular Expression Matching
Hard

2922

553

Favorite

Share
Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.

'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).

Note:

s could be empty and contains only lowercase letters a-z.
p could be empty and contains only lowercase letters a-z, and characters like . or *.
Example 1:

Input:
s = "aa"
p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".

Next challenges:
K Inverse Pairs Array
Next Closest Time
Remove Vowels from a String

class Solution(object):
    def isMatch(self, s, p):
        lenS, lenP = len(s), len(p)
        dp = [[False] * (lenP + 1) for i in range(lenS + 1)]

        # initialization, when p is empty, always Flase, when s is empty:
        dp[0][0] = True
        for j in range(2, lenP + 1): dp[0][j] = dp[0][j - 2] and p[j - 1] == '*'

        # dp
        for i in range(1, lenS + 1):
            for j in range(1, lenP + 1):
                dp[i][j] = dp[i][j - 2] or (p[j - 2] in (s[i - 1], '.') and dp[i - 1][j]) if p[j - 1] == '*' \
                    else dp[i - 1][j - 1] and p[j - 1] in ('.', s[i - 1])
        return dp[lenS][lenP]
        
        
 140. Word Break II
Hard

1111

261

Favorite

Share
Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences.

Note:

The same word in the dictionary may be reused multiple times in the segmentation.
You may assume the dictionary does not contain duplicate words.
Example 1:

Input:
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]
Output:
[
  "cats and dog",
  "cat sand dog"
]
class Solution(object):
    def wordBreak(self, s, wordDict):
        res = []
        self.dfs(s, wordDict, '', res)
        return res

    def dfs(self, s, dic, path, res):
    # Before we do dfs, we check whether the remaining string 
    # can be splitted by using the dictionary,
    # in this way we can decrease unnecessary computation greatly.
        if self.check(s, dic): # prunning
            if not s:
                res.append(path[:-1])
                return # backtracking
            for i in xrange(1, len(s)+1):
                if s[:i] in dic:
                    # dic.remove(s[:i])
                    self.dfs(s[i:], dic, path+s[:i]+" ", res)

    # DP code to check whether a string can be splitted by using the 
    # dic, this is the same as word break I.                
    def check(self, s, dic):
        dp = [False for i in xrange(len(s)+1)]
        dp[0] = True
        for i in xrange(1, len(s)+1):
            for j in xrange(i):
                if dp[j] and s[j:i] in dic:
                    dp[i] = True
        return dp[-1]
Next challenges:
Concatenated Words

124. Binary Tree Maximum Path Sum
Hard

1876

140

Favorite

Share
Given a non-empty binary tree, find the maximum path sum.

For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.

Example 1:

Input: [1,2,3]

       1
      / \
     2   3

Output: 6
Example 2:

Input: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

Output: 42


Next challenges:
Path Sum
Sum Root to Leaf Numbers
Path Sum IV
Longest Univalue Path

class Solution:
    def maxPathSum(self, root):
        
        def helper(root):
            if root:
                l = max(helper(root.left), 0)
                r = max(helper(root.right), 0)
                max1=root.val + l + r
                res[0] = max(res[0], max1)
                max2= max(l, r)
                return root.val +max2
            else:
                return 0
        
        res = [float('-inf')]
        helper(root)
        return res[0]
        
  https://briangordon.github.io/2014/08/the-skyline-problem.html
  218. The Skyline Problem
Hard

1340

67

Favorite

Share
A city's skyline is the outer contour of the silhouette formed by all the buildings in that city when viewed from a distance. Now suppose you are given the locations and height of all the buildings as shown on a cityscape photo (Figure A), write a program to output the skyline formed by these buildings collectively (Figure B).

Buildings  Skyline Contour
The geometric information of each building is represented by a triplet of integers [Li, Ri, Hi], where Li and Ri are the x coordinates of the left and right edge of the ith building, respectively, and Hi is its height. It is guaranteed that 0 ≤ Li, Ri ≤ INT_MAX, 0 < Hi ≤ INT_MAX, and Ri - Li > 0. You may assume all buildings are perfect rectangles grounded on an absolutely flat surface at height 0.

For instance, the dimensions of all buildings in Figure A are recorded as: [ [2 9 10], [3 7 15], [5 12 12], [15 20 10], [19 24 8] ] .

The output is a list of "key points" (red dots in Figure B) in the format of [ [x1,y1], [x2, y2], [x3, y3], ... ] that uniquely defines a skyline. A key point is the left endpoint of a horizontal line segment. Note that the last key point, where the rightmost building ends, is merely used to mark the termination of the skyline, and always has zero height. Also, the ground in between any two adjacent buildings should be considered part of the skyline contour.

For instance, the skyline in Figure B should be represented as:[ [2 10], [3 15], [7 12], [12 0], [15 10], [20 8], [24, 0] ].

Notes:

The number of buildings in any input list is guaranteed to be in the range [0, 10000].
The input list is already sorted in ascending order by the left x position Li.
The output list must be sorted by the x position.
There must be no consecutive horizontal lines of equal height in the output skyline. For instance, [...[2 3], [4 5], [7 5], [11 5], [12 7]...] is not acceptable; the three lines of height 5 should be merged into one in the final output as such: [...[2 3], [4 5], [12 7], ...]

  from heapq import heappush, heappop
class Solution(object):
    def getSkyline(self, buildings):
        # add start-building events
        # also add end-building events(acts as buildings with 0 height)
        # and sort the events in left -> right order
        events = [(L, -H, R) for L, R, H in buildings]
        events += list({(R, 0, 0) for _, R, _ in buildings})
        events.sort()

        # res: result, [x, height]
        # live: heap, [-height, ending position]
        res = [[0, 0]]
        live = [(0, float("inf"))]
        for pos, negH, R in events:
            # 1, pop buildings that are already ended
            # 2, if it's the start-building event, make the building alive
            # 3, if previous keypoint height != current highest height, edit the result
            while live[0][1] <= pos: heappop(live)
            if negH: heappush(live, (negH, R))
            if res[-1][1] != -live[0][0]:
                res += [ [pos, -live[0][0]] ]
        return res[1:]
        
 1.First, sort the critical points by its left endpoints. We treat R in (R,0,None) as left endpoint. Then scan across the critical points from left to right.

2.We only push right end points onto the heap. Think of it as a proxy for the entire rectangle. The key is its negative height because heapq implements min-heap. The heap keeps track of the current max height.

3.In the for-loop, when we encounter a left end point that is larger than maxheight (hp[0][0]), we pop hp until all right endpoints smaller than the current left end point are gone. Interestingly, we don't traverse through the heap and remove a rectangle every time an incoming left endpoint comes along. Because we only care about the max height, aka, heap[0][0].

3.Finally, after updating the heap, we check whether the current max height (hp[0[0]) differs from the last max height (res[-1][1] ), if so, we append the hp[0][0] as the height .
In short,
a. if the height at current left point is the first in the heap (after we just updated it),then negH == -hp[0][0].
b. if the height at current left point is not the first in the heap ,that means it is either completely overshadowed by the taller buildings or it will be used when the taller building is popped from the heap. In the second case, don't forget that our lower building's right endpoint is still in the heap, when taller building is popped from the heap, and the lower building's height becomes the max height.

O(nlogn) time. O(n) space
  Next challenges:
Falling Squares
https://leetcode.com/problems/the-skyline-problem/discuss/61261/10-line-Python-solution-104-ms


239. Sliding Window Maximum
Hard

1972

117

Favorite

Share
Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.

Example:

Input: nums = [1,3,-1,-3,5,3,6,7], and k = 3
Output: [3,3,5,5,6,7] 
Explanation: 

Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
 
 The key is to just keep the que sorted. You want to really have it argsorted so you can remove the indexes that will no longer be in the window. 
 Last trick is to realize that when you are adding new item to the que, you want to slide it into place while removing elements less than it. 
 So basically now you get O(n) instead of O(k * n) because if you removed z out of m elements on the previouse run,
  you will then remove no more that m - z the next run hence not looping through k every window and 
  if you didn't remove anything then you didn't loop at all through the que--needless to say but also not looping through k elements in that case.



class Solution(object):
    def maxSlidingWindow(self, nums, k):
        d = collections.deque()
        #print(d)
        out = []
        for i, n in enumerate(nums):
            #print(i,n)
            while d and nums[d[-1]] < n:
                d.pop()
            d += i,
            #print(d[0])
            if d[0] == i - k:
                d.popleft()
            if i >= k - 1:
                out += nums[d[0]],
        return out
        
Next challenges:
Minimum Window Substring
Longest Substring with At Most Two Distinct Characters
Paint House II


130. Surrounded Regions
Medium

881

460

Favorite

Share
Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region.

Example:

X X X X
X O O X
X X O X
X O X X
After running your function, the board should be:

X X X X
X X X X
X X X X
X O X X
Explanation:

Surrounded regions shouldn’t be on the border, which means that any 'O' on the border of the board are not flipped to 'X'. 
Any 'O' that is not on the border and it is not connected to an 'O' on the border will be flipped to 'X'. 
Two cells are connected
 if they are adjacent cells connected horizontally or vertically.
 
 class Solution(object):
    def solve(self,board):
        queue = []
        for r in range(len(board)):
            for c in range(len(board[0])):
                if (r in [0, len(board)-1] or c in [0, len(board[0])-1]) and board[r][c] == 'O':
                    queue.append((r,c))
        while queue:
            r,c = queue.pop(0)
            if 0<=r<len(board) and 0<=c<len(board[0]) and board[r][c] == 'O':
                board[r][c] = 'V'
                queue.append((r-1,c))
                queue.append((r+1,c))
                queue.append((r,c-1))
                queue.append((r,c+1))

        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] == 'V':
                    board[r][c] = 'O'
                elif board[r][c] == 'O':
                    board[r][c] = 'X'
Next challenges:
Walls and Gates


295. Find Median from Data Stream
Hard

1355

27

Favorite

Share
Median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. So the median is the mean of the two middle value.

For example,
[2,3,4], the median is 3

[2,3], the median is (2 + 3) / 2 = 2.5

Design a data structure that supports the following two operations:

void addNum(int num) - Add a integer number from the data stream to the data structure.
double findMedian() - Return the median of all elements so far.
 
 
 1. If all integer numbers from the stream are between 0 and 100, how would you optimize it?

We can maintain an integer array of length 100 to store the count of each number along with a total count. Then, we can iterate over the array to find the middle value to get our median.

Time and space complexity would be O(100) = O(1).

2. If 99% of all integer numbers from the stream are between 0 and 100, how would you optimize it?

In this case, we need an integer array of length 100 and a hashmap for these numbers that are not in [0,100].

from heapq import heappush, heappop
class MedianFinder(object):

    

    def __init__(self):
        self.max_heap, self.min_heap = [], [] # lower nums, higher nums

    def addNum(self, num):
        heappush(self.min_heap, num)
        heappush(self.max_heap, -heappop(self.min_heap))

        m, n = len(self.max_heap), len(self.min_heap), 
        if m > n: # min heap can only be larger 1, so if m is greater we're breaking this property
            heappush(self.min_heap, -heappop(self.max_heap))

    def findMedian(self):
        m, n = len(self.min_heap), len(self.max_heap)
        if m == n: 
            valr=float(self.min_heap[0] - self.max_heap[0]) / 2
            return valr
        return self.min_heap[0]



# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
Next challenges:
Sliding Window Median


315. Count of Smaller Numbers After Self
Hard

1293

59

Favorite

Share
You are given an integer array nums and you have to return a new counts array. The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i].

Example:

Input: [5,2,6,1]
Output: [2,1,1,0] 
Explanation:
To the right of 5 there are 2 smaller elements (2 and 1).
To the right of 2 there is only 1 smaller element (1).
To the right of 6 there is 1 smaller element (1).
To the right of 1 there is 0 smaller element.

https://pymotw.com/2/bisect/

import bisect
class Solution(object):
    def countSmaller(self, nums):
        sorted_arr = []
        rst = []
        for num in nums[::-1]:
            #print(nums[::-1])
            idx = bisect.bisect_left(sorted_arr, num)
            #print(idx)
            rst.append(idx)
            sorted_arr.insert(idx, num)
            #print(idx,sorted_arr)
            
        return rst[::-1]
        
 Next challenges:
Count of Range Sum
Queue Reconstruction by Height
Reverse Pairs

84. Largest Rectangle in Histogram
Hard

2197

59

Favorite

Share
Given n non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram.

 To get the bigest rectangle area, we should check every rectangle with lowest point at i=1...n.

Define Si := the square with lowest point at i.
To calculate faster this Si , we have to use a stack stk which stores some indices.

The elements in stk satisfy these properties:

the indices as well as the corresponding heights are in ascending order
for any adjecent indices i and j (eg. s=[...,i,j,...]), any index k between i and j are of height higher than j:
height[k]>height[j]
We loop through all indices, when we meet an index k with height lower than elements in stk (let's say, lower than index i in stk), we know that the right end of square Si is just k-1. And what is the left end of this square? Well it is just the index to the left of i in stk !

Another important thing is that we should append a 0 to the end of height, so that all indices in stk will be checked this way.
 class Solution(object):
    def largestRectangleArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        mx = 0
        stk = []
        height.append(0) # very important!!
        for k in xrange(len(height)): 
            while(stk and height[k]<height[stk[-1]]):
                rect = height[stk.pop()] * (k if not stk else k-stk[-1]-1)
                mx = max(mx, rect)
            stk.append(k)
        return mx
      Next challenges:
Maximal Rectangle



76. Minimum Window Substring
Hard

2655

177

Favorite

Share
Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

Example:

Input: S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"
Note:

If there is no such window in S that covers all characters in T, return the empty string "".
If there is such window, you are guaranteed that there will always be only one unique minimum window in S.

Next challenges:
Substring with Concatenation of All Words
Minimum Size Subarray Sum
Permutation in String
Smallest Range Covering Elements from K Lists
Minimum Window Subsequence

import sys
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        required, min_len = [0] * 128, sys.maxint
        # record numbers of each character in t
        for ch in t:
            required[ord(ch)] += 1
        left = right = min_start = 0
        # count: number of characters that are still required
        count = len(required) - required.count(0)
        while right < len(s):
            while count > 0 and right < len(s):
                # move right till s[left : right] has all characters in t
                required[ord(s[right])] -= 1
                if required[ord(s[right])] == 0:
                    count -= 1
                right += 1
            # s[left : right] has all characters in t, move left pointer
            while count == 0:
                required[ord(s[left])] += 1
                if required[ord(s[left])] == 1:
                    # now s[left : right] misses one character, update min_len if necessary
                    count = 1
                    if right - left < min_len:
                        min_len, min_start = right - left, left
                left += 1

        return s[min_start : min_start+min_len] if min_len != sys.maxint else ''
        
        from collections import Counter

class Solution(object):
    def minWindow(self,S, T):
        """
        Minimum Window Substring

        :param str S:
        :param str T:
        :return str:
        """
        Tc = Counter(T)
        print(Tc)
        Sc = Counter()

        best_i = -sys.maxsize
        best_j = sys.maxsize

        i = 0

        for j, char in enumerate(S):
            Sc[char] += 1
            print(Sc[char],Tc)

            while Sc & Tc == Tc:
                if j - i < best_j - best_i:
                    best_i, best_j = i, j

                Sc[S[i]] -= 1
                i += 1

        return S[best_i : best_j + 1] if best_j - best_i < len(S) else ""



Runtime: 72 ms, faster than 90.37% of Python online submissions for Minimum Window Substring.
Memory Usage: 12.4 MB, less than 100.00% of Python online submissions for Minimum Window Substring.
Next challenges:
Substring with Concatenation of All Words
Minimum Size Subarray Sum
Permutation in String
Smallest Range Covering Elements from K Lists
Minimum Window Subsequence


212. Word Search II
Hard

1311

75

Favorite

Share
Given a 2D board and a list of words from the dictionary, find all words in the board.

Each word must be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

 

Example:

Input: 
board = [
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]
words = ["oath","pea","eat","rain"]

Output: ["eat","oath"]
 

Note:

All inputs are consist of lowercase letters a-z.
The values of words are distinct.
class Trie:
    def __init__(self):
        self.letter2Node = {}
        self.isWord = False
        self.word = None
        
class Dictionary:
    def __init__( self):
        self.root = Trie()
        
    def add(self, word):
        curNode = self.root
        for l in word:
            if l not in curNode.letter2Node:
                curNode.letter2Node[ l ] = Trie()
            curNode = curNode.letter2Node[ l ]
        curNode.isWord = True
        curNode.word = word

class Solution(object):
    def findWords(self, board, words):
        # create my dictionary
        myDic = Dictionary()
        # add the words to the dictionary 
        for word in words:
            myDic.add(word)
        finalWords = {}
        
        
        visited = [[False for letter in row] for row in board]
        # explore every location as "pivots"
        for i in range( len(board) ):
            for j in range( len(board[0]) ):
                # location, board, visited, trieNode, dictionary
                self.explore( i, j, board, visited, myDic.root, finalWords )
        
        return list(finalWords.keys())
    
    def explore( self, i, j, board, visited, trieNode, finalWords ):
        # base condition
        if visited[i][j]:
            return
    
        letter = board[i][j]
        if letter not in trieNode.letter2Node:
            return
        
        visited[i][j] = True
        trieNode = trieNode.letter2Node[letter]
        # check if we have reached the end
        if trieNode.isWord:
            finalWords[ trieNode.word ] = True
        
        # get all the neighbor letters #(i,j) position
        neighborsL = self.getNeighbors(i, j, board, visited)
        for neighbor in neighborsL:
            # explote to see if any word will match
            self.explore(neighbor[0], neighbor[1], board, visited, trieNode, finalWords)
            
        # backtracking
        visited[i][j] = False
    
    def getNeighbors(self, i, j , board, visited):
        neighbors = []
        lenRow = len( board )
        lenCol = len( board[0] )
        for iN, jN in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rN = i+iN
                cN = j+jN
                if 0 <= rN < lenRow and 0 <= cN < lenCol and not visited[rN][cN]:
                    neighbors.append( [rN, cN ] )
        
        return neighbors
  Next challenges:
Unique Paths III




