301. Remove Invalid Parentheses
Hard

1633

72

Favorite

Share
Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.

Note: The input string may contain letters other than the parentheses ( and ).

Example 1:

Input: "()())()"
Output: ["()()()", "(())()"]
Example 2:

Input: "(a)())()"
Output: ["(a)()()", "(a())()"]
Example 3:

Input: ")("
Output: [""]

class Solution(object):
    def removeInvalidParentheses(self, s):
        res = []
        self.visited = set([s])
        self.dfs(s, self.invalid(s), res)
        return res
    
    def dfs(self, s, n, res):
        if n == 0:
            res.append(s)
            return
        for i in range(len(s)):
            if s[i] in ('(',')'):
                new_s = s[:i]+s[i+1:]
                if new_s not in self.visited and self.invalid(new_s) < n:
                    self.visited.add(new_s)
                    self.dfs(new_s, self.invalid(new_s), res)
        
    def invalid(self, s):
        plus = minus = 0
        memo = {"(":1, ")":-1}
        for c in s:
            plus += memo.get(c,0)
            minus += 1 if plus < 0 else 0
            plus = max(0, plus)
        return plus + minus
        
        
Next challenges:
Minimum Height Trees
Cousins in Binary Tree
Lowest Common Ancestor of Deepest Leaves
        
        
        312. Burst Balloons
Hard

1512

47

Favorite

Share
Given n balloons, indexed from 0 to n-1. Each balloon is painted with a number on it represented by array nums. You are asked to burst all the balloons. If the you burst balloon i you will get nums[left] * nums[i] * nums[right] coins. Here left and right are adjacent indices of i. After the burst, the left and right then becomes adjacent.

Find the maximum coins you can collect by bursting the balloons wisely.

Note:

You may imagine nums[-1] = nums[n] = 1. They are not real therefore you can not burst them.
0 ≤ n ≤ 500, 0 ≤ nums[i] ≤ 100
Example:

Input: [3,1,5,8]
Output: 167 
Explanation: nums = [3,1,5,8] --> [3,5,8] -->   [3,8]   -->  [8]  --> []
             coins =  3*1*5      +  3*5*8    +  1*3*8      + 1*8*1   = 167
             
             
             Before started, I removed all balloons with number 0, and put an additional "1" at the beginning and end, for that based on the definition of this problem, one can imagine there're implicitly two "1" balloons at the beginning and end but would never be burst. The balloon array now becomes: [1,...,x,x,x,x,x,x,...,1], where the x's are original nonzero balloons.

I feel the trickiest part is to sort out what you really need to calculate in each DP sub-problem. In each sub-problem I have 3 pointers "l", "m" and "r" located as below:

       l   m     r
  [...,x,x,x,x,x,x,...]
I focus on the region (l,r), and assign m as the last balloon to be burst in this region. I need to calculate:

max coins after the balloons in region (l,m) are burst

max coins after the balloons in region (m,r) are burst

nums[l]*nums[m]*nums[r]

Note I'm using exclusive region notation, which means the lth and rth balloons are not burst in this sub-problem.

With each iteration I gradually increase the interval between balloons l and r. Such process is equivalent to beginning from the 1st burst balloon.
 As the interval to be considered increases, all the possible combination of sub-intervals within current interval would have been calculated in previous iterations.

In the end I just return the regional max coins excluding the first and the last balloons, which are the 2 extra balloons I appended before started (now you can see why they're needed).
             def maxCoins(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums = [1]+[n for n in nums if n!=0]+[1]
    regional_max_coins = [[0 for i in xrange(len(nums))] for j in xrange(len(nums))]
    for balloons_to_burst in xrange(1, len(nums)-1): # number of balloons in (l,r) region
        for l in xrange(0, len(nums)-balloons_to_burst-1): # for m and r to be assigned legally
            r = l+balloons_to_burst+1
            for m in xrange(l+1,r):
                regional_max_coins[l][r] = max(regional_max_coins[l][r], regional_max_coins[l][m]+regional_max_coins[m][r]+nums[l]*nums[m]*nums[r])
    return regional_max_coins[0][-1]
    
    Next challenges:
Minimum Cost to Merge Stones



72. Edit Distance
Hard

2403

35

Favorite

Share
Given two words word1 and word2, find the minimum number of operations required to convert word1 to word2.

You have the following 3 operations permitted on a word:

Insert a character
Delete a character
Replace a character
Example 1:

Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
Example 2:

Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')

Next challenges:
One Edit Distance
Delete Operation for Two Strings
Minimum ASCII Delete Sum for Two Strings
Uncrossed Lines


85. Maximal Rectangle
Hard

1663

54

Favorite

Share
Given a 2D binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

Example:

Input:
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
Output: 6
based on largest rectangle in histogram

def maximalRectangle(self, matrix):
    if not matrix or not matrix[0]:
        return 0
    n = len(matrix[0])
    height = [0] * (n + 1)
    ans = 0
    for row in matrix:
        for i in xrange(n):
            height[i] = height[i] + 1 if row[i] == '1' else 0
        stack = [-1]
        for i in xrange(n + 1):
            while height[i] < height[stack[-1]]:
                h = height[stack.pop()]
                w = i - 1 - stack[-1]
                ans = max(ans, h * w)
            stack.append(i)
    return ans
    
    similar to 
    https://leetcode.com/problems/largest-rectangle-in-histogram/
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
                #print("inside while",k,height[stk[-1]])
                rect = height[stk.pop()] * (k if not stk else k-stk[-1]-1)
                mx = max(mx, rect)
            stk.append(k)
            #print(k,stk)
        return mx  
    
    Next challenges:
Array Nesting
Non-decreasing Array
New 21 Game

32. Longest Valid Parentheses
Hard

2183

100

Favorite

Share
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

Example 1:

Input: "(()"
Output: 2
Explanation: The longest valid parentheses substring is "()"
Example 2:

Input: ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()"

class Solution(object):
    def longestValidParentheses(self, s):
        stack, result = [(-1, ')')], 0
        for i, paren in enumerate(s):
            if paren == ')' and stack[-1][1] == '(':
                stack.pop()
                result = max(result, i - stack[-1][0])
            else:
                stack += (i, paren),
        return result
           
Next challenges:
Split Array Largest Sum
Find Duplicate File in System
Camelcase Matching
