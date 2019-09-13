338. Counting Bits
Medium

1575

116

Favorite

Share
Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array.

Example 1:

Input: 2
Output: [0,1,1]
Example 2:

Input: 5
Output: [0,1,1,2,1,2]
Follow up:

It is very easy to come up with a solution with run time O(n*sizeof(integer)). But can you do it in linear time O(n) /possibly in a single pass?
Space complexity should be O(n).
Can you do it like a boss? Do it without using any builtin function like __builtin_popcount in c++ or in any other language.


class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        ret = [0]
        while num > len(ret) - 1:
            #print(ret)
            ret += [a + 1 for a in ret]
        return ret[0 : num + 1]
        
 Next challenges:
Maximal Square
Burst Balloons
Guess Number Higher or Lower II





406. Queue Reconstruction by Height
Medium

1749

196

Favorite

Share
Suppose you have a random list of people standing in a queue. Each person is described by a pair of integers (h, k), where h is the height of the person and k is the number of people in front of this person who have a height greater than or equal to h. Write an algorithm to reconstruct the queue.

Note:
The number of people is less than 1,100.

 
Example

Input:
[[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]

Output:
[[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]


Pick out tallest group of people and sort them in a subarray (S). 
Since there's no other groups of people taller than them, therefore each guy's index will be just as same as his k value.
For 2nd tallest group (and the rest), insert each one of them into (S) by k value. So on and so forth.
E.g.
input: [[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]
subarray after step 1: [[7,0], [7,1]]
subarray after step 2: [[7,0], [6,1], [7,1]]

step 1: sort
[7,0], [7,1], [6,1], [5,0], [5,2], [4,4]

step 2: insert by height
[7,0]
[7,0], [7,1]
[7,0], [6,1], [7,1]
[5,0], [7,0], [6,1], [7,1]
[5,0], [7,0], [5,2], [6,1], [7,1]
[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]

class Solution(object):
    def reconstructQueue(self, people):
        people.sort(key=lambda p: (p[0], -p[1]), reverse=True)
        res = []
        [res.insert(p[1], p) for p in people]
        return res
        
       Next challenges:
Meeting Rooms II
Remove Duplicate Letters
Course Schedule III


739. Daily Temperatures
Medium

1522

43

279. Perfect Squares
Medium

1658

137

Favorite

Share
Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.

Example 1:

Input: n = 12
Output: 3 
Explanation: 12 = 4 + 4 + 4.
Example 2:

Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.

Next challenges:
Ugly Number II
class Solution(object):
    def numSquares(self,n):
        dp = [0] + [float('inf')]*n
        #print(dp)
        for i in range(1, n+1):
            dp[i] = min(dp[i-j*j] for j in range(1, int(i**0.5)+1)) + 1
        return dp[n]

Favorite

Share
Given a list of daily temperatures T, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead.

For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].

Note: The length of temperatures will be in the range [1, 30000]. Each temperature will be an integer in the range [30, 100].

if len(T)==0: return []
	stack = []
	ans = [0]*len(T)
	pos = 0
	for i in range(len(T)):
		while stack and stack[-1][1] < T[i]:
			idx, _ = stack.pop()
			ans[idx] = i - idx
		stack.append((pos,T[i]))
		pos += 1
	return ans
	
	
	Next challenges:
Next Greater Element I
...




647. Palindromic Substrings
Medium

1571

81

Favorite

Share
Given a string, your task is to count how many palindromic substrings in this string.

The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.

Example 1:

Input: "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".
 

Example 2:

Input: "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".

We perform a "center expansion" among all possible centers of the palindrome.

Let N = len(S). There are 2N-1 possible centers for the palindrome: we could have a center at S[0], between S[0] and S[1], at S[1], between S[1] and S[2], at S[2], etc.

To iterate over each of the 2N-1 centers, we will move the left pointer every 2 times, and the right pointer every 2 times starting with the second (index 1). Hence, left = center / 2, right = center / 2 + center % 2.

From here, finding every palindrome starting with that center is straightforward: while the ends are valid and have equal characters, record the answer and expand.

def countSubstrings(self, S):
    N = len(S)
    ans = 0
    for center in xrange(2*N - 1):
        left = center / 2
        right = left + center % 2
        while left >= 0 and right < N and S[left] == S[right]:
            ans += 1
            left -= 1
            right += 1
    return ans
    
    
    Next challenges:
Encode and Decode Strings
Stone Game
Number of Submatrices That Sum to Target


399. Evaluate Division
Medium

1480

116

Favorite

Share
Equations are given in the format A / B = k, where A and B are variables represented as strings, and k is a real number (floating point number). Given some queries, return the answers. If the answer does not exist, return -1.0.

Example:
Given a / b = 2.0, b / c = 3.0.
queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? .
return [6.0, 0.5, -1.0, 1.0, -1.0 ].

The input is: vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> queries , where equations.size() == values.size(), and the values are positive. This represents the equations. Return vector<double>.

According to the example above:

equations = [ ["a", "b"], ["b", "c"] ],
values = [2.0, 3.0],
queries = [ ["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"] ]. 


64. Minimum Path Sum
Medium

1573

42

Favorite

Share
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

class Solution(object):
    
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid or not grid[0]: return 0
        m=len(grid)
        n=len(grid[0])
        helper=[float("inf")]*(n+1)
        helper[1]=0
        for i in range(m):
            for j in range(1,n+1):
                helper[j]=min(helper[j-1],helper[j])+grid[i][j-1]
                print(helper[j])
        return helper[-1]
        
        
        Next challenges:
Dungeon Game
Cherry Pickup


Given n, how many structurally unique BST's (binary search trees) that store values 1 ... n?

Example:

Input: 3
Output: 5
Explanation:
Given n = 3, there are a total of 5 unique BST's:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
   
   
   Catalan Formula = 2n!/(n+1!*n!)

class Solution(object):
    def numTrees(self, n):
        facn=1
        for i in range(1,n+1):
            facn *= i
        fac2n=facn
        facnp=facn*(n+1)
        for i in range(n+1,2*n+1):
            fac2n *= i
        ans = fac2n/(facn*facnp)
        return ans
        
   Next challenges:
Unique Binary Search Trees II



621. Task Scheduler
Medium

1873

325

Favorite

Share
Given a char array representing tasks CPU need to do. It contains capital letters A to Z where different letters represent different tasks. Tasks could be done without original order. Each task could be done in one interval. For each interval, CPU could finish one task or just be idle.

However, there is a non-negative cooling interval n that means between two same tasks, there must be at least n intervals that CPU are doing different tasks or just be idle.

You need to return the least number of intervals the CPU will take to finish all the given tasks.

 

Example:

Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B.

Next challenges:
Rearrange String k Distance Apart
Reorganize String

394. Decode String
Medium

1769

96

Favorite

Share
Given an encoded string, return its decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there won't be input like 3a or 2[4].

Examples:

s = "3[a]2[bc]", return "aaabcbc".
s = "3[a2[c]]", return "accaccacc".
s = "2[abc]3[cd]ef", return "abcabccdcdcdef"

class Solution:
    def decodeString(self, s):
        stack = [['', 1, '']]
        a = n = ''
        for c in s:
            if c.isalpha():
                a += c
            elif c.isdigit():
                n += c
            elif c == '[':
                stack.append([a, int(n), ''])
                a = n = ''
            else:
                p, t, b = stack.pop()
                #print("p",p,"t",t,"b",b,"a",a)
                stack[-1][-1] += p + t * (b + a)
                a = ''
        return stack.pop()[-1] + a
        
 Next challenges:
Encode String with Shortest Length
Number of Atoms
Brace Expansion


494. Target Sum
Medium

1565

70

Favorite

Share
You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols + and -. For each integer, you should choose one from + and - as its new symbol.

Find out how many ways to assign symbols to make sum of integers equal to target S.

Example 1:
Input: nums is [1, 1, 1, 1, 1], S is 3. 
Output: 5
Explanation: 

-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3

There are 5 ways to assign symbols to make the sum of nums be target 3.
Note:
The length of the given array is positive and will not exceed 20.
The sum of elements in the given array will not exceed 1000.
Your output answer is guaranteed to be fitted in a 32-bit integer.



309. Best Time to Buy and Sell Stock with Cooldown
Medium

1530

59

Favorite

Share
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times) with the following restrictions:

You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)
Example:

Input: [1,2,3,0,2]
Output: 3 
Explanation: transactions = [buy, sell, cooldown, buy, sell]

class Solution(object):
    def maxProfit(self, prices):
        have, cool, free = float('-inf'), float('-inf'), 0
        for p in prices:
            free, have, cool = max(free, cool), max(have, free - p), have + p
        return max(free, cool)
        
 Next challenges:
Ugly Number II
Paint Fence
Sentence Screen Fitting


114. Flatten Binary Tree to Linked List
Medium

1707

224

Favorite

Share
Given a binary tree, flatten it to a linked list in-place.

For example, given the following tree:

    1
   / \
  2   5
 / \   \
3   4   6
The flattened tree should look like:

1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6


class Solution(object):
    def flatten(self, root):
        
	# recursive solution
        if not root:
            return 
        
        l, r = root.left, root.right
        root.left, root.right = None, l
        
        self.flatten(l)
        
        while root.right:
            root = root.right
        root.right = r
        self.flatten(r)
        
        Next challenges:
Flatten a Multilevel Doubly Linked List


def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        
        
        q = [root]
        par = None
        while q:
            node = q.pop()
            if node:
                q += [node.right, node.left]
                if par:
                    par.left = None
                    par.right = node
                par = node
                
 560. Subarray Sum Equals K
Medium

2347

68

Favorite

Share
Given an array of integers and an integer k, you need to find the total number of continuous subarrays whose sum equals to k.

Example 1:
Input:nums = [1,1,1], k = 2
Output: 2
Note:
The length of the array is in range [1, 20,000].
The range of numbers in the array is [-1000, 1000] and the range of the integer k is [-1e7, 1e7].

560. Subarray Sum Equals K
Medium

2347

68

Favorite

Share
Given an array of integers and an integer k, you need to find the total number of continuous subarrays whose sum equals to k.

Example 1:
Input:nums = [1,1,1], k = 2
Output: 2
Note:
The length of the array is in range [1, 20,000].
The range of numbers in the array is [-1000, 1000] and the range of the integer k is [-1e7, 1e7].

Next challenges:
Continuous Subarray Sum
Subarray Product Less Than K
Find Pivot Index
Subarray Sums Divisible by K


Just wanted to share a clear explanation that helped me.

First of all, the basic idea behind this code is that, whenever the sums has increased by a value of k, we've found a subarray of sums=k.
I'll also explain why we need to initialise a 0 in the hashmap.
Example: Let's say our elements are [1,2,1,3] and k = 3.
and our corresponding running sums = [1,3,4,7]
Now, if you notice the running sums array, from 1->4, there is increase of k and from 4->7, there is an increase of k. So, we've found 2 subarrays of sums=k.

But, if you look at the original array, there are 3 subarrays of sums==k. Now, you'll understand why 0 comes in the picture.

In the above example, 4-1=k and 7-4=k. Hence, we concluded that there are 2 subarrays.
However, if sums==k, it should've been 3-0=k. But 0 is not present in the array. To account for this case, we include the 0.
Now the modified sums array will look like [0,1,3,4,7]. Now, try to see for the increase of k.

0->3
1->4
4->7
Hence, 3 sub arrays of sums=k
This clarified some confusions I had while doing this problem.
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        count = 0
        sums = 0
        d = dict()
        d[0] = 1
        
        for i in range(len(nums)):
            sums += nums[i]
            count += d.get(sums-k,0)
            d[sums] = d.get(sums,0) + 1
        
        return(count)
        
        
 416. Partition Equal Subset Sum
Medium

1422

40

Favorite

Share
Given a non-empty array containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

Note:

Each of the array element will not exceed 100.
The array size will not exceed 200.
 

Example 1:

Input: [1, 5, 11, 5]

Output: true

Explanation: The array can be partitioned as [1, 5, 5] and [11].
 class Solution(object):
    def canPartition(self, nums):
        target = sum(nums)
        if target % 2:
            return False
        target //= 2
        dp = [True] + [False]*target
        for n in nums:
            for s in range(target, n-1, -1):
                dp[s] = dp[s] or dp[s-n]
        return dp[-1]
 Next challenges:
Partition to K Equal Sum Subsets


221. Maximal Square
Medium

1515

35

Favorite

Share
Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

Example:

Input: 

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
https://leetcode.com/problems/maximal-square/discuss/61935/6-lines-Visual-Explanation-O(mn)

# O(m*n) space, one pass  
def maximalSquare2(self, matrix):
    if not matrix:
        return 0
    r, c = len(matrix), len(matrix[0])
    dp = [[int(matrix[i][j]) for j in xrange(c)] for i in xrange(r)]
    res = max(max(dp))
    for i in xrange(1, r):
        for j in xrange(1, c):
            dp[i][j] = (min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j])+1)*int(matrix[i][j])
            res = max(res, dp[i][j]**2)
    return res
    
    Next challenges:
Maximal Rectangle
Largest Plus Sign


142. Linked List Cycle II
Medium

1648

129

Favorite

Share
Given a linked list, return the node where the cycle begins. If there is no cycle, return null.

To represent a cycle in the given linked list, we use an integer pos which represents the position (0-indexed) in the linked list where tail connects to. If pos is -1, then there is no cycle in the linked list.

Note: Do not modify the linked list.

 

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: tail connects to node index 1
Explanation: There is a cycle in the linked list, where tail connects to the second node

https://leetcode.com/problems/linked-list-cycle-ii/discuss/44859/Python-O(N)-NO-extra-space-with-MATHEMATICAL-explanation GOOD eXPLANATION

Next challenges:
Reorder List
Linked List Components
Squares of a Sorted Array

31. Next Permutation
Medium

2107

666

Favorite

Share
Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).

The replacement must be in-place and use only constant extra memory.

Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.

1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1

Next challenges:
Permutations II
Permutation Sequence
Palindrome Permutation II


399. Evaluate Division
Medium

1485

116

Favorite

Share
Equations are given in the format A / B = k, where A and B are variables represented as strings, and k is a real number (floating point number). Given some queries, return the answers. If the answer does not exist, return -1.0.

Example:
Given a / b = 2.0, b / c = 3.0.
queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? .
return [6.0, 0.5, -1.0, 1.0, -1.0 ].

The input is: vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> queries , where equations.size() == values.size(), and the values are positive. This represents the equations. Return vector<double>.

According to the example above:

equations = [ ["a", "b"], ["b", "c"] ],
values = [2.0, 3.0],
queries = [ ["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"] ]. 

"""Current permutation should share longest prefix with next permutation and it's very similar to a carry in addition. Three steps: 1) from right to left find the longest increasing sequence, 2) carry to the previous number, 3) reverse the increasing sequence."""

class Solution(object):
    def nextPermutation(self, nums):
        
        # from right to left find longest increasing sequence
        i = len(nums) - 1
        while i > 0 and nums[i - 1] >= nums[i]: i -= 1
        # swap nums[i - 1] and min(number in nums[i...end] which is larger than nums[i - 1])
        # consider it as a carry to nums[i - 1]
        if i > 0:
            j, pre = len(nums) - 1, nums[i - 1]
            while j >= i and nums[j] <= pre: j -= 1
            nums[i - 1], nums[j] = nums[j], nums[i - 1]
        # reverse nums[i...end]
        k = len(nums) - 1
        while i < k:
            nums[i], nums[k] = nums[k], nums[i]
            i, k = i + 1, k - 1




