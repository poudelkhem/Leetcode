19. Remove Nth Node From End of List
Medium

2041

154

Favorite

Share
Given a linked list, remove the n-th node from the end of list and return its head.

Example:

Given linked list: 1->2->3->4->5, and n = 2.

After removing the second node from the end, the linked list becomes 1->2->3->5.
Note:

Given n will always be valid.

Follow up:

Could you do this in one pass?

class Solution:
    # one pass solution with O(n) time, O(1) space
    def removeNthFromEnd(self, head, n):
        dummy = ListNode(None) # dummy matters
        dummy.next = head
        ptr1 = ptr2 = dummy
        while n:
            ptr2 = ptr2.next
            n -= 1
        while ptr2.next:
            ptr1 = ptr1.next
            ptr2 = ptr2.next
        ptr1.next = ptr1.next.next
        return dummy.next
 
 
 22. Generate Parentheses
Medium

3100

192

Favorite

Share
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given n = 3, a solution set is:

[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]


29. Divide Two Integers
Medium

737

3547

Favorite

Share
Given two integers dividend and divisor, divide two integers without using multiplication, division and mod operator.

Return the quotient after dividing dividend by divisor.

The integer division should truncate toward zero.

Example 1:

Input: dividend = 10, divisor = 3
Output: 3
Example 2:

Input: dividend = 7, divisor = -3
Output: -2
Note:

Both dividend and divisor will be 32-bit signed integers.
The divisor will never be 0.
Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 231 − 1 when        
        
  class Solution:
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        res = []
        def dfs (l, r, path, res):
            if r < l or l == -1 or r == -1:
                return
            if l == 0 and r == 0:
                res.append(path[:])
            else:
                dfs(l-1, r, path + "(", res)
                dfs(l, r-1, path + ")", res)
        dfs(n, n, "", res)
        return (res)
        
38. Count and Say
Easy

857

6653

Favorite

Share
The count-and-say sequence is the sequence of integers with the first five terms as following:

1.     1
2.     11
3.     21
4.     1211
5.     111221
1 is read off as "one 1" or 11.
11 is read off as "two 1s" or 21.
21 is read off as "one 2, then one 1" or 1211.

Given an integer n where 1 ≤ n ≤ 30, generate the nth term of the count-and-say sequence.

Note: Each term of the sequence of integers will be represented as a string.

class Solution:
    def countAndSay(self, n):
    	if n == 1:
    		return '1'
    	s = '11'
    	for i in range(n-2):
    		c, t = 1, ''
    		for j in range(len(s)-1):
    			if s[j+1] == s[j]:
    				c +=1
    			else:
                    
    				t += str(c)+s[j]
               
    				c = 1
    		if s[-2] == s[-1]:
    			t += str(c)+s[-1]
    		else:
    			t += '1'+s[-1]
    		s = t
    	return(s)


29. Divide Two Integers
Medium

737

3549

Favorite

Share
Given two integers dividend and divisor, divide two integers without using multiplication, division and mod operator.

Return the quotient after dividing dividend by divisor.

The integer division should truncate toward zero.

Example 1:

Input: dividend = 10, divisor = 3
Output: 3
Example 2:

Input: dividend = 7, divisor = -3
Output: -2

class Solution:
    def divide(self, dividend, divisor):
        a, b, r, t = abs(dividend), abs(divisor), 0, 1
        while a >= b or t > 1:
            if a >= b: r += t; a -= b; t += t; b += b
            else: t >>= 1; b >>= 1
        return min((-r, r)[dividend ^ divisor >= 0], (1 << 31) - 1)
        
53. Maximum Subarray
Easy

4815

177



33. Search in Rotated Sorted Array
Medium

2765

342

Favorite

Share
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of O(log n).

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

class Solution:
# @param {integer[]} nums
# @param {integer} target
# @return {integer}
def search(self, nums, target):
    if not nums:
        return -1
    return self.binarySearch(nums, target, 0, len(nums)-1)
    
def binarySearch(self, nums, target, start, end):
    if end < start:
        return -1
    mid = (start+end)/2
    if nums[mid] == target:
        return mid
    if nums[start] <= target < nums[mid]: # left side is sorted and has target
        return self.binarySearch(nums, target, start, mid-1)
    elif nums[mid] < target <= nums[end]: # right side is sorted and has target
        return self.binarySearch(nums, target, mid+1, end)
    elif nums[mid] > nums[end]: # right side is pivoted
        return self.binarySearch(nums, target, mid+1, end)
    else: # left side is pivoted
        return self.binarySearch(nums, target, start, mid-1)


Next challenges:
Search in Rotated Sorted Array II
Find Minimum in Rotated Sorted Array

Favorite

Share
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Example:

Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
Follow up:

If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.

class Solution:
    def maxSubArray(self, nums):
        for i in range(1, len(nums)):
            nums[i] = max(nums[i], nums[i] + nums[i-1])
        return max(nums)
        
66. Plus One
Easy

950

1687

Favorite

Share
Given a non-empty array of digits representing a non-negative integer, plus one to the integer.

The digits are stored such that the most significant digit is at the head of the list, and each element in the array contain a single digit.

You may assume the integer does not contain any leading zero, except the number 0 itself.

Example 1:

Input: [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
Example 2:

Input: [4,3,2,1]
Output: [4,3,2,2]
Explanation: The array represents the integer 4321.

class Solution:
    def plusOne(self, digits):
        result = []
        str_digits = ''.join(str(i) for i in digits)
        int_digits = int(str_digits) + 1
        str_digits = str(int_digits)
        for i in str_digits:
            result.append(int(i))
        return result

4. Median of Two Sorted Arrays
Hard

4793

684

Favorite

Share
There are two sorted arrays nums1 and nums2 of size m and n respectively.

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

You may assume nums1 and nums2 cannot be both empty.

Example 1:

nums1 = [1, 3]
nums2 = [2]

The median is 2.0

class Solution(object):
    
    def findMedianSortedArrays(self, nums1, nums2):
        added = nums1 + nums2
        added = sorted(added)
        if (len(added) % 2 == 1):
            medi = added[len(added) // 2]
        else:
            a = added[len(added) // 2]
            b = added[len(added) // 2 - 1]
            medi = float(a + b) / 2
        return medi
        
  10. Regular Expression Matching
Hard

2900

552

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
Example 2:

Input:
s = "aa"
p = "a*"
Output: true
Explanation: '*' means zero or more of the preceding element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".
Example 3:

Input:
s = "ab"
p = ".*"
Output: true
Explanation: ".*" means "zero or more (*) of any character (.)".
Example 4:

Input:
s = "aab"
p = "c*a*b"
Output: true
Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore, it matches "aab".
Example 5:

Input:
s = "mississippi"
p = "mis*is*p*."
Output: false

725. Split Linked List in Parts
Medium

397

83

Favorite

Share
Given a (singly) linked list with head node root, write a function to split the linked list into k consecutive linked list "parts".

The length of each part should be as equal as possible: no two parts should have a size differing by more than 1. This may lead to some parts being null.

The parts should be in order of occurrence in the input list, and parts occurring earlier should always have a size greater than or equal parts occurring later.

Return a List of ListNode's representing the linked list parts that are formed.

Examples 1->2->3->4, k = 5 // 5 equal parts [ [1], [2], [3], [4], null ]
Example 1:
Input: 
root = [1, 2, 3], k = 5
Output: [[1],[2],[3],[],[]]

class Solution(object):
    def splitListToParts(self, root, k):
    
        if not root: return [None]*k

        head = root
        length, i = [0]*k, 0

        while root:
            length[i] += 1
            root = root.next
            i = (i + 1) % k

        res = []
        for i in range(len(length)):
            if length[i] == 0:
                res.append(None)
            else:
                res.append(head)
                for _ in range(length[i]-1):
                    head = head.next
                head.next, head = None, head.next       
        return res
        
167. Two Sum II - Input array is sorted
Easy

1016

422

Favorite

Share
Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2.

Note:

Your returned answers (both index1 and index2) are not zero-based.
You may assume that each input would have exactly one solution and you may not use the same element twice.
Example:

Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore index1 = 1, index2 = 2.

23. Merge k Sorted Lists
Hard

2777

181

Favorite

Share
Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

Example:

Input:
[
  1->4->5,
  1->3->4,
  2->6
]
Output: 1->1->2->3->4->4->5->6

from operator import attrgetter

class Solution:
    # @param a list of ListNode
    # @return a ListNode
    def mergeKLists(self, lists):
        sorted_list = []
        for head in lists:
            curr = head
            while curr is not None:
                sorted_list.append(curr)
                curr = curr.next

        sorted_list = sorted(sorted_list, key=attrgetter('val'))
        #print(sorted_list,attrgetter('val'))
        for i, node in enumerate(sorted_list):
            try:
                node.next = sorted_list[i + 1]
            except:
                node.next = None

        if sorted_list:
            return sorted_list[0]
        else:
            return None
            
            
            
            33. Search in Rotated Sorted Array
Medium

2765

342

Favorite

Share
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of O(log n).

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
            
            class Solution:
    # @param {integer[]} nums
    # @param {integer} target
    # @return {integer}
    def search(self, nums, target):
        if not nums:
            return -1
        return self.binarySearch(nums, target, 0, len(nums)-1)

    def binarySearch(self, nums, target, start, end):
        if end < start:
            return -1
        mid = (start+end)/2
        if nums[mid] == target:
            return mid
        if nums[start] <= target < nums[mid]: # left side is sorted and has target
            return self.binarySearch(nums, target, start, mid-1)
        elif nums[mid] < target <= nums[end]: # right side is sorted and has target
            return self.binarySearch(nums, target, mid+1, end)
        elif nums[mid] > nums[end]: # right side is pivoted
            return self.binarySearch(nums, target, mid+1, end)
        else: # left side is pivoted
            return self.binarySearch(nums, target, start, mid-1)

            
34. Find First and Last Position of Element in Sorted Array
Medium

1843

95

Favorite

Share
Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

Your algorithm's runtime complexity must be in the order of O(log n).

If the target is not found in the array, return [-1, -1].

Example 1:

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
Example 2:

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]

class Solution(object):
    
    def bsearch(self, nums, target):
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) / 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left
    
    
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # Idea1: two binary search to find bounds.
        left = self.bsearch(nums, target-0.5)
        right = self.bsearch(nums, target+0.5) - 1
        if right < left:
            return [-1, -1]
        return [left, right]
        
        
 36. Valid Sudoku
Medium

957

347

Favorite

Share
Determine if a 9x9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

Each row must contain the digits 1-9 without repetition.
Each column must contain the digits 1-9 without repetition.
Each of the 9 3x3 sub-boxes of the grid must contain the digits 1-9 without repetition.
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """

        map_row = [{} for _ in xrange(9)]
        map_col = [{} for _ in xrange(9)]
        map_cell = [[{} for _ in xrange(3)] for __ in xrange(3)]
        for i in xrange(9):
            for j in xrange(9):
                char = board[i][j]
                if char == '.': continue
                if char in map_row[i]: return False
                else: map_row[i][char] = [i,j]
                if char in map_col[j]: return False
                else: map_col[j][char] = [i,j]
                if char in map_cell[i/3][j/3]: return False
                else: map_cell[i/3][j/3][char] = [i,j]
        return True      
 Next challenges:
Sudoku Solver



39. Combination Sum
Medium

2275

70

Favorite

Share
Given a set of candidate numbers (candidates) (without duplicates) and a target number (target), find all unique combinations in candidates where the candidate numbers sums to target.

The same repeated number may be chosen from candidates unlimited number of times.

Note:

All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
Example 1:

Input: candidates = [2,3,6,7], target = 7,
A solution set is:
[
  [7],
  [2,2,3]
]
class Solution:
    def combinationSum(self, candidates, target):
        candidates.sort()
        ans = []
        def helper(sums,path):
            for n in candidates:
                if not path or n >= path[-1]:
                    if sums+n == target:
                        ans.append(path+[n])
                        return
                    elif sums+n<target:
                        helper(sums+n,path+[n])
                    else:
                        return
        helper(0,[])
        return ans
        
Combination Sum II
Combinations
Combination Sum III
Factor Combinations
Combination Sum IV


40. Combination Sum II
Medium

981

50

Favorite

Share
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sums to target.

Each number in candidates may only be used once in the combination.

Note:

All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
Example 1:

Input: candidates = [10,1,2,7,6,1,5], target = 8,
A solution set is:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]


class Solution:
    # @param {integer[]} candidates
    # @param {integer} target
    # @return {integer[][]}
    def combinationSum2(self, candidates, target):
        candidates.sort()
        combinations, stack = [], [(0, 0, [])]
        while stack:
            (total, start, combination) = stack.pop()
            for index in range(start, len(candidates)):
                sum = total + candidates[index]
                if sum < target:
                    if (index == start) or (candidates[index] != candidates[index-1]):
                        # avoid duplicates
                        stack.append((sum, index+1, combination + [candidates[index]]))
                else:
                    if sum == target:
                        combinations.append(combination + [candidates[index]])
                    # no need to try any more
                    break
        return combinations
        
 41. First Missing Positive
Hard

1929

622

Favorite

Share
Given an unsorted integer array, find the smallest missing positive integer.

Example 1:

Input: [1,2,0]
Output: 3
Example 2:

Input: [3,4,-1,1]
Output: 2

class Solution(object):
     def firstMissingPositive(self, nums):
        index = 1
        while True:
            if index not in nums:
                return index
            else:
                index += 1
 Next challenges:
Missing Number
Find the Duplicate Number
Find All Numbers Disappeared in an Array
Couples Holding Hand

43. Multiply Strings
Medium

1121

509

Favorite

Share
Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.

Example 1:

Input: num1 = "2", num2 = "3"
Output: "6"
Next challenges:
Add Binary
Add Strings





69. Sqrt(x)
Easy

847

1471

Favorite

Share
Implement int sqrt(int x).

Compute and return the square root of x, where x is guaranteed to be a non-negative integer.

Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.

Example 1:

Input: 4
Output: 2

class Solution(object):
    def mySqrt(self, x):
        lo, hi = 0, x
        
        while lo <= hi:
            mid = (lo + hi) // 2
            
            if mid * mid > x:
                hi = mid - 1
            elif mid * mid < x:
                lo = mid + 1
            else:
                return mid
        
        # When there is no perfect square, hi is the the value on the left
        # of where it would have been (rounding down). If we were rounding up, 
        # we would return lo
        return hi
  Next challenges:
Pow(x, n)
Valid Perfect Square

70. Climbing Stairs
Easy

2516

90

Favorite

Share
You are climbing a stair case. It takes n steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Note: Given n will be a positive integer.

Example 1:

Input: 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

class Solution:
    def climbStairs(self, n):
        fib, count = [1, 2], 2
        while count < n:
            fib.append(fib[count - 1] + fib[count - 2])
            count += 1
        return fib[n - 1]
 88. Merge Sorted Array
Easy

1241

2985

Favorite

Share
Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

Note:

The number of elements initialized in nums1 and nums2 are m and n respectively.
You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2.
Example:

Input:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

Output: [1,2,2,3,5,6]


from collections import deque
class Solution:
    def merge(self, nums1, m, nums2, n):
        """Do not return anything, modify nums1 in-place instead."""
		
        if (not nums1 or not nums2):
            return
        d1 = deque(nums1[:m])
        d2 = deque(nums2[:n])
        nums1[:] = []
        while d1 and d2:
            if d1[0] <= d2[0]:
                nums1.append(d1.popleft())
            else:
                nums1.append(d2.popleft())
        nums1 += d1 or d2
108. Convert Sorted Array to Binary Search Tree
Easy

1326

138

Favorite

Share
Given an array where elements are sorted in ascending order, convert it to a height balanced BST.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

Example:

Given the sorted array: [-10,-3,0,5,9],

One possible answer is: [0,-3,9,-10,null,5], which represents the following height balanced BST:

      0
     / \
   -3   9
   /   /
 -10  5
 
 class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if len(nums) == 0:
            return None
        elif len(nums) == 1:
            root = TreeNode(nums[0])
            return root
        else:
            if len(nums)%2 == 1:
                h = len(nums)//2
            else:
                h = (len(nums)//2)-1
                
            root = TreeNode(nums[h])
            
            l_subtree = self.sortedArrayToBST(nums[:h])
            r_subtree = self.sortedArrayToBST(nums[h+1:])
            
            root.left = l_subtree
            root.right = r_subtree
            
            return root
            
  371. Sum of Two Integers
Easy

822

1502

Favorite

Share
Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -.

Example 1:

Input: a = 1, b = 2
Output: 3
Example 2:

Input: a = -2, b = 3
Output: 1


class Solution(object):
    def getSum(self, a, b):
        def add(a, b): 
            if not a or not b:
                return a or b
            return add(a^b, (a&b) << 1)

        if a*b < 0: # assume a < 0, b > 0
            if a > 0:
                return self.getSum(b, a)
            if add(~a, 1) == b: # -a == b
                return 0
            if add(~a, 1) < b: # -a < b
                return add(~add(add(~a, 1), add(~b, 1)),1) # -add(-a, -b)

        return add(a, b) # a*b >= 0 or (-a) > b > 0 


104. Maximum Depth of Binary Tree
Easy

1488

59

Favorite

Share
Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Note: A leaf is a node with no children.

Example:

Given binary tree [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
return its depth = 3.

class Solution:
    def maxDepth(self, root):
        if not root:
            return 0
        stack,depth = [(root,1)],0
        while stack:
            cur,cur_depth = stack.pop()
            if cur:
                depth = max(depth,cur_depth)
                if cur.left:
                    stack.append((cur.left,cur_depth+1))
                if cur.right:
                    stack.append((cur.right,cur_depth+1))
        return depth
Next challenges:
Balanced Binary Tree
Minimum Depth of Binary Tree
Maximum Depth of N-ary Tree

50. Pow(x, n)
Medium

930

2274

Favorite

Share
Implement pow(x, n), which calculates x raised to the power n (xn).

Example 1:

Input: 2.00000, 10
Output: 1024.00000

class Solution:
    def myPow(self, x, n):
        if n < 0:
            x = 1/x
            n = -n
        ans = 1
        while n :
            if n % 2:
                ans = x * ans
                n -= 1
            else:
                x = x * x
                n = n // 2
        return ans
        
class Solution:
    def permute(self, nums):
        self.perms = []
        self.helper(nums, [])
        return self.perms
    
    def helper(self, nums, curr):
        #print(nums, curr)
        if len(nums) == 0 and len(curr) > 0:
            self.perms.append(curr)
        else:
            for i in range(len(nums)):
                self.helper(nums[:i] + nums[i+1:], curr + [nums[i]])
        
                
Next challenges:
Next Permutation
Permutations II
Permutation Sequence
Combinations


149. Max Points on a Line
Hard

527

1425

Favorite

Share
Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.

Example 1:

Input: [[1,1],[2,2],[3,3]]
Output: 3
Explanation:
^
|
|        o
|     o
|  o  
+------------->
0  1  2  3  4

Next challenges:
Line Reflection

class Solution(object):
    def maxPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        res = 0
        if not points:
            return res 
        
        n = len(points)
        for i in range(n):
            pairs = {}
            duplicates = 1
            for j in range(i+1, n):
                if points[i][0]==points[j][0] and points[i][1] == points[j][1]:
                    duplicates+=1
                    continue
                dex = points[j][0]-points[i][0]
                dey = points[j][1]-points[i][1]
                d = self.gcd(dex, dey)
                if (dex//d, dey//d,) in pairs:
                    pairs[(dex//d, dey//d)]+=1
                else:
                    pairs[(dex//d, dey//d)]=1
            res = max(res, duplicates)
            for ele in pairs:
                res = max(pairs[ele]+duplicates, res)
        return res 
                
    def gcd(self, a, b):
        if b==0:
            return a
        return self.gcd(b, a%b)

def maxPoints(self, points):
        l = len(points)
        m = 0
        for i in range(l):
        
            dic = {'i': 1}
            same = 0
            for j in range(i+1, l):
                tx, ty = points[j].x, points[j].y
                if tx == points[i].x and ty == points[i].y: 
                    same += 1
                    continue
                if points[i].x == tx: slope = 'i'
                else:slope = (points[i].y-ty) * 1.0 /(points[i].x-tx)
                if slope not in dic: dic[slope] = 1
                dic[slope] += 1
            m = max(m, max(dic.values()) + same)
    return m
    
128. Longest Consecutive Sequence
Hard

2076

101

Favorite

Share
Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

Your algorithm should run in O(n) complexity.

Example:

Input: [100, 4, 200, 1, 3, 2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

class Solution(object):
    def longestConsecutive(self, nums):
        seen, longest = {}, 0
        for n in nums:
          if n not in seen: 
            l, h = seen.get(n-1, (n,n))[0], seen.get(n+1, (n,n))[1]
            seen[n] = [l, h] # [lowest num in known range, highest num in known range]
            longest = max(longest, h - l + 1)
            # update our range ends if needed
            if l < n: seen[l][1] = h
            if h > n: seen[h][0] = l
        return longest
Next challenges:
Binary Tree Longest Consecutive Sequence

54. Spiral Matrix
Medium

1280

439

Favorite

Share
Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.

Example 1:

Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1,2,3,6,9,8,7,4,5]
Example 2:

Input:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]

class Solution(object):
    def spiralOrder(self, matrix):
        res = []
        while matrix:
            res.extend(matrix.pop(0)) # left to right
            if matrix and matrix[0]: # top to dwon
                for row in matrix:
                    res.append(row.pop())
            if matrix: # right to left
                res.extend(matrix.pop()[::-1])
            if matrix and matrix[0]: # bottom to up
                for row in matrix[::-1]:
                    res.append(row.pop(0))
        return res
Next challenges:
Spiral Matrix II
55. Jump Game
Medium

2249

219

Favorite

Share
Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

Example 1:

Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

class Solution(object):
    def canJump(self, nums):
        k = 1
        for i in range(len(nums)-2, -1, -1):
            k = k + 1 if nums[i] < k else 1
        return k == 1
Next challenges:
Jump Game II
45. Jump Game II
Hard

1358

79

Favorite

Share
Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Your goal is to reach the last index in the minimum number of jumps.

Example:

Input: [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2.
    Jump 1 step from index 0 to 1, then 3 steps to the last index.
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        
    Basically it's a shortest-path problem. 
    As an unweighted graph, BFS should be able to solve it in O(|E|).
    But as it's an array with non-negative numbers, we can't jump backward. 
    So we can do better here.
    Suppose we devided the arrays into segments depending on the elment 
    in the array. So for each segment, we find the farthest index we can 
    jump. For example, the first segment is always A[0]. The second will be
    from A[1] to A[A[0]]. The third will be from A[A[0]] to the farthest 
    index we can find in the second segment. We start looking between 
    the end of the last segment and the begin of the next segment.
    """
        
        ans = lastIdx = nextIdx = 0
        while nextIdx < len(nums) - 1:
            ans += 1
            lastIdx, nextIdx = nextIdx, max(i + nums[i] for i in xrange(lastIdx, nextIdx + 1))
        return ans


287. Find the Duplicate Number
Medium

2770

317

Favorite

Share
Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.

Example 1:

Input: [1,3,4,2,2]
Output: 2
Example 2:

Input: [3,1,3,4,2]
Output: 3

class Solution(object):
        def findDuplicate(self, nums):
            """
            :type nums: List[int]
            :rtype: int
            """
            dict1={}
            for x in nums:
                


                if x in dict1:
                    return x
                else:
                    dict1[x]=1
                    
 class Solution(object):
    def topKFrequent(self, nums, k):
        numdict = {} #Counter of elements
        for num in nums:
            numdict[num] = numdict.get(num, 0) + 1
        output = [] #stores all elements sorted by frequency
        for num in sorted(numdict, key = numdict.get, reverse = True):
            output.append(num)
        return output[:k]
 347. Top K Frequent Elements
Medium

1743

111

Favorite

Share
Given a non-empty array of integers, return the k most frequent elements.

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
Example 2:

Input: nums = [1], k = 1
Output: [1]


Next challenges:
Word Frequency
Split Array into Consecutive Subsequences
K Closest Points to Origin


230. Kth Smallest Element in a BST
Medium

1314

44

Favorite

Share
Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

Note: 
You may assume k is always valid, 1 ≤ k ≤ BST's total elements.

Example 1:

Input: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
Output: 1
Example 2:
	
	def kthSmallest(self, root, k):
    stack = []
    curr = root
    while stack or curr:
        if curr:
            stack.append(curr)
            curr = curr.left
        else:
            curr = stack.pop()
            k -= 1
            if k == 0:
                return curr.val
            curr = curr.right
            
 Next challenges:
Second Minimum Node In a Binary Tree


671. Second Minimum Node In a Binary Tree
Easy

409

614

Favorite

Share
Given a non-empty special binary tree consisting of nodes with the non-negative value, where each node in this tree has exactly two or zero sub-node. If the node has two sub-nodes, then this node's value is the smaller value among its two sub-nodes. More formally, the property root.val = min(root.left.val, root.right.val) always holds.

Given such a binary tree, you need to output the second minimum value in the set made of all the nodes' value in the whole tree.

If no such second minimum value exists, output -1 instead.

Example 1:

Input: 
    2
   / \
  2   5
     / \
    5   7

Output: 5
Explanation: Th



454. 4Sum II
Medium

750

61

Favorite

Share
Given four lists A, B, C, D of integer values, compute how many tuples (i, j, k, l) there are such that A[i] + B[j] + C[k] + D[l] is zero.

To make problem a bit easier, all A, B, C, D have same length of N where 0 ≤ N ≤ 500. All integers are in the range of -228 to 228 - 1 and the result is guaranteed to be at most 231 - 1.

Example:

Input:
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]

Output:
2

Explanation:
The two tuples are:
1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0


class Solution(object):
    def fourSumCount(self, A, B, C, D):
        """
        :type A: List[int]
        :type B: List[int]
        :type C: List[int]
        :type D: List[int]
        :rtype: int
        """
        
        res = 0
        counter = {}
        
        for a in A:
            for b in B:
                counter[a+b] = counter.get(a+b, 0) + 1
        
        for c in C:
            for d in D:
                res += counter.get(-c-d, 0)
                
        
        return res

Next challenges:
Maximum Average Subarray II
Peak Index in a Mountain Array
Find Common Characters



Next challenges:
Wiggle Sort
Spiral Matrix III
String Transforms Into Another String

384. Shuffle an Array
Medium

306

704

Favorite

Share
Shuffle a set of numbers without duplicates.

Example:

// Init an array with set 1, 2, and 3.
int[] nums = {1,2,3};
Solution solution = new Solution(nums);

// Shuffle the array [1,2,3] and return its result. Any permutation of [1,2,3] must equally likely to be returned.
solution.shuffle();

// Resets the array back to its original configuration [1,2,3].
solution.reset();

// Returns the random shuffling of array [1,2,3].
solution.shuffle();


import random
class Solution(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums[:]
        self.now = nums[:]
    def reset(self):
        """
        Resets the array to its original configuration and return it.
        :rtype: List[int]
        """
        self.now = self.nums[:] # list(self.nums) also create new array
        return self.now

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        :rtype: List[int]
        """
        for i in xrange(len(self.now) - 1):
            idx = random.randint(i,len(self.now) - 1)
            self.now[i],self.now[idx] = self.now[idx],self.now[i]
        
        return self.now
# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.reset()
# param_2 = obj.shuffle()



another way

import random
def __init__(self, nums):
		self.nums = nums
class Solution(list):

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        """
        return self

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        """
        length = self.__len__()
        return [self[i] for i in random.sample(range(length), length)]
        
        
 378. Kth Smallest Element in a Sorted Matrix
Medium

1365

87

Favorite

Share
Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the kth distinct element.

Example:

matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
],
k = 8,

return 13.


class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:

        res= []
        for i in range (len(matrix)):
            for j in range (len(matrix)):
                res.append(matrix[i][j])
        
        heapq.heapify(res)
        return heapq.nsmallest(k,res)[-1]
 Next challenges:
Find K Pairs with Smallest Sums
Kth Smallest Number in Multiplication Table
Find K-th Smallest Pair Distance
K-th Smallest Prime Fraction


328. Odd Even Linked List
Medium

863

236

Favorite

Share
Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are talking about the node number and not the value in the nodes.

You should try to do it in place. The program should run in O(1) space complexity and O(nodes) time complexity.

Example 1:

Input: 1->2->3->4->5->NULL
Output: 1->3->5->2->4->NULL
Example 2:

Input: 2->1->3->5->6->4->7->NULL
Output: 2->3->6->7->1->5->4->NULL

class Solution:
    def oddEvenList(self, head):
        if head == None:
            return None
        even = head.next
        evenCopy = even
        headCopy = head
        while head.next and even.next:
            head.next = head.next.next
            even.next = even.next.next
            if head.next:
                head = head.next
            if even.next:
                even = even.next
        head.next = evenCopy
        return headCopy
 341. Flatten Nested List Iterator
Medium

1070

452

Favorite

Share
Given a nested list of integers, implement an iterator to flatten it.

Each element is either an integer, or a list -- whose elements may also be integers or other lists.

Example 1:

Input: [[1,1],2,[1,1]]
Output: [1,1,2,1,1]
Explanation: By calling next repeatedly until hasNext returns false, 
             the order of elements returned by next should be: [1,1,2,1,1].
             
             class NestedIterator(object):
    def __init__(self,nestedList):
        self.queue = collections.deque([])
        for elem in nestedList:
            if elem.isInteger():
                self.queue.append(elem.getInteger())
            else:
                newList = NestedIterator(elem.getList())
                while newList.hasNext():
                    self.queue.append(newList.next())
    def hasNext(self):
        if self.queue:
            return True
        return False
    def next(self):
        return self.queue.popleft()
        

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())

Next challenges:
Flatten 2D Vector
Zigzag Iterator
Mini Parser
Array Nesting

Next challenges:
Copy List with Random Pointer
Insertion Sort List
Design Phone Directory

62. A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?


class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        dp = [[1]*m for _ in range(n)]
        
        for i in range(1,n):
            for j in range(1,m):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
                
        return dp[-1][-1]

Next challenges:
Unique Paths II
Minimum Path Sum
Dungeon Game


56. Merge Intervals
Medium

2414

187

Favorite

Share
Given a collection of intervals, merge all overlapping intervals.

Example 1:

Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
Example 2:

Input: [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.


class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        intervals = sorted(intervals, key=lambda interval:interval.start)
        result = []
        interval, l = intervals[0], len(intervals)
        for i in range(1, l):
            interval2 = intervals[i]
            if interval2.start > interval.end:
                result.append(interval)
                interval = interval2
            else:
                interval.end = max（interval.end, interval2.end)
        
        result.append(interval)
        return result
        
        or 
        
        class Solution(object):
    def merge(self, intervals):
        out = []
        for i in sorted(intervals, key=lambda i: i[0]):
            print(i)
            if out and i[0] <= out[-1][1]:
                out[-1][1] = max(out[-1][1], i[1])
            else:
                out += i,
        return out
        
        Next challenges:
Insert Interval
Meeting Rooms
Meeting Rooms II
Teemo Attacking
Add Bold Tag in String
Range Module
Employee Free Time
Interval List Intersections

75. Sort Colors
Medium

1860

167

Favorite

Share
Given an array with n objects colored red, white or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

Note: You are not suppose to use the library's sort function for this problem.

Example:

Input: [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]

def sortColors(self, nums):
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    left = 0
    right = len(nums)-1
    while left<right and nums[left]==0:
        left+=1
    while left<right and nums[right]==2:
        right-=1
    cur =left
    while cur<=right and left<right:
        if nums[cur]==0:
            nums[cur],nums[left]=nums[left],nums[cur]
            left+=1
            cur+=1
        elif nums[cur]==1:
            cur+=1
        else:
            nums[cur],nums[right]=nums[right],nums[cur]
            right-=1

Next challenges:
Sort List
Wiggle Sort
Wiggle Sort II


91. Decode Ways
Medium

1572

1817

Favorite

Share
A message containing letters from A-Z is being encoded to numbers using the following mapping:

'A' -> 1
'B' -> 2
...
'Z' -> 26
Given a non-empty string containing only digits, determine the total number of ways to decode it.

Example 1:

Input: "12"
Output: 2
Explanation: It could be decoded as "AB" (1 2) or "L" (12).

76. Minimum Window Substring
Hard

2627

174

Favorite

Share
Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

Example:

Input: S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"
Note:

If there is no such window in S that covers all characters in T, return the empty string "".
If there is such window, you are guaranteed that there will always be only one unique minimum window in S.






78. Subsets
Medium

2226

54

Favorite

Share
Given a set of distinct integers, nums, return all possible subsets (the power set).

Note: The solution set must not contain duplicate subsets.

Example:

Input: nums = [1,2,3]
Output:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]

class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = [[]]
        for n in nums:
            for i in range(len(res)):
                res.append(res[i] + [n])
        return res


79. Word Search
Medium

2060

106

Favorite

Share
Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

Example:

board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.

class Solution:
    def exist(self, board, word):
        if not board or not board[0]: return False
        m, n = len(board), len(board[0])
        visited = [[False] * n for _ in range(m)]
        
        def dfs(r, c, index):
            if index >= len(word): return True
            if not (0 <= r < m) or not (0 <= c < n) or visited[r][c] or board[r][c] != word[index]: return False
            visited[r][c] = True
            for i, j in [[r+1, c], [r-1, c], [r, c+1], [r, c-1]]:
                if dfs(i, j, index+1):
                    visited[r][c] = False
                    return True
            visited[r][c] = False
            return False        
        return any(dfs(i, j, 0) for i in range(m) for j in range(n))
        
            
class Solution:
    def exist(self, board, word):
        if not board or not board[0]: return False
        m, n = len(board), len(board[0])
        visited = [[False] * n for _ in range(m)]
        
        def dfs(r, c, index):
            if index >= len(word): return True
            if not (0 <= r < m) or not (0 <= c < n) or visited[r][c] or board[r][c] != word[index]: return False
            visited[r][c] = True
            for i, j in [[r+1, c], [r-1, c], [r, c+1], [r, c-1]]:
                if dfs(i, j, index+1):
                    visited[r][c] = False
                    return True
            visited[r][c] = False
            return False        
        return any(dfs(i, j, 0) for i in range(m) for j in range(n))        
        Next challenges:
Word Search II

103. Binary Tree Zigzag Level Order Traversal
Medium

1125

65

Favorite

Share
Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]
Accepted
242,491
Submissions
566,063

class Solution:
    def zigzagLevelOrder(self, root):
        if not root:
            return []
        
        res = []
        queue = []
        
        queue.append((root,1))
        while queue:
            n = len(queue)
            nodeList = []
            for _ in range(n):
                curNode, level = queue.pop(0)
                print(level)
                if level%2 == 1:
                    nodeList.append(curNode.val)
                else:
                    nodeList.insert(0,curNode.val)
                
                if curNode.left: queue.append((curNode.left,level+1))
                if curNode.right: queue.append((curNode.right,level+1))
            
            res.append(nodeList)
        
        return res
        
        105. Construct Binary Tree from Preorder and Inorder Traversal
Medium

1966

54

Favorite

Share
Given preorder and inorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

For example, given

preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
Return the following binary tree:

    3
   / \
  9  20
    /  \
   15   7
   
   class Solution:
    def buildTree(self, preorder, inorder):
        if inorder:
            ind = inorder.index(preorder.pop(0))
            #print(ind)
            root = TreeNode(inorder[ind])
            root.left = self.buildTree(preorder, inorder[0:ind])
            root.right = self.buildTree(preorder, inorder[ind+1:])
            return root

Next challenges:
Binary Tree Postorder Traversal
Is Graph Bipartite?
Smallest String Starting From Leaf
Binary Tree Postorder Traversal
Is Graph Bipartite?
Smallest String Starting From Leaf

Next challenges:
Construct Binary Tree from Inorder and Postorder Traversal

200. Number of Islands
Medium

2984

108

Favorite

Share
Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example 1:

Input:
11110
11010
11000
00000

Output: 1
Example 2:

Input:
11000
11000
00100
00011

Output: 3

#The problem asks the number of connected '1' in the grid.
#So once we find a '1', we can use dfs or bfs to flipped all connected '1' to '0'. The times of #finding a new '1' would be the number of islands.
class Solution:
    def numIslands(self,grid):
        m, n = len(grid), len(grid) and len(grid[0])
        def dfs(i,j):
            if 0 <= i < m and 0 <= j < n and grid[i][j] == '1':
                grid[i][j] = '0'
                print(grid[i][j])
                dfs(i-1, j), dfs(i+1, j), dfs(i, j-1), dfs(i, j+1)
                return 1
            return 0
        return sum(dfs(i,j) for i in range(m) for j in range(n))
     Next challenges:
Surrounded Regions
Walls and Gates
Number of Islands II
Number of Connected Components in an Undirected Graph
Number of Distinct Islands
Max Area of Island


131. Palindrome Partitioning
Medium

1059

42

Favorite

Share
Given a string s, partition s such that every substring of the partition is a palindrome.

Return all possible palindrome partitioning of s.

Example:

Input: "aab"
Output:
[
  ["aa","b"],
  ["a","a","b"]
]
class Solution:
    def partition(self, s):
        res = []
        self.dfs(s, [], res)
        return res

    def dfs(self, s, path, res):
        if not s:
            res.append(path)
            return
        for i in range(1, len(s)+1):
            if self.isPal(s[:i]):
                self.dfs(s[i:], path+[s[:i]], res)

    def isPal(self, s):
        return s == s[::-1]
  Next challenges:
Palindrome Partitioning II


162. Find Peak Element
Medium

949

1455

Favorite

Share
A peak element is an element that is greater than its neighbors.

Given an input array nums, where nums[i] ≠ nums[i+1], find a peak element and return its index.

The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.

You may imagine that nums[-1] = nums[n] = -∞.

Example 1:

Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.

class Solution:
    def findPeakElement(self, nums):
        leng = len(nums)
        if leng == 1:
            return 0
        left, right = 0, leng-1
        while left < right:  # there is always a peak so we use '<' instead of '<='
            mid = (left+right) // 2
            if nums[mid] < nums[mid+1]:
                # 'left' and 'right' could be consecutive
                # 'mid' intends to be 'left', so we need 'left' to plus one to avoid endless loop			
                left = mid + 1
            else:
                right = mid
        return left
        
 Next challenges:
Peak Index in a Mountain Array


134. Gas Station
Medium

848

303

Favorite

Share
There are N gas stations along a circular route, where the amount of gas at station i is gas[i].

You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its next station (i+1). You begin the journey with an empty tank at one of the gas stations.

Return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1.

Note:

If there exists a solution, it is guaranteed to be unique.
Both input arrays are non-empty and have the same length.
Each element in the input arrays is a non-negative integer.
Example 1:

Input: 
gas  = [1,2,3,4,5]
cost = [3,4,5,1,2]

Output: 3

class Solution:
    def canCompleteCircuit(self, gas, cost):
        start = tank = judge = 0
        for i in range(len(gas)):
            tank += gas[i] - cost[i]
            
            judge += gas[i] - cost[i]
            #print("tank",tank,"judge",judge)
            if tank < 0:
                start = i + 1
                tank = 0
        if judge < 0 or start >= len(gas):
            return -1
        else:
            return start
 Next challenges:
Find Permutation
Score After Flipping Matrix
Previous Permutation With One Swap


138. Copy List with Random Pointer
Medium

1773

470

Favorite

Share
A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.

Return a deep copy of the list.



"""W shape link 

insert copied node after each old node
so we get a double length linked list
update random of each new node
restore old link and get new link"""

class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        
        if not head: return head
        
        # insert new copied node after the old one
        p = head
        h = None
        while p:
            q = Node(p.val, p.next, p.random)
            if not h: h = q
            p.next = q
            p = q.next
    
        # update random 
        q = h
        while q: 
            if q.random: 
                q.random = q.random.next
            q = q.next
        
        # restore old link
        p = head
        q = h
        while p and q:
            p.next = q.next
            p = p.next
            if p: q.next = p.next
            q = q.next
            
        return h
        
 or 
 
 class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        
        if not head: return head
        
        d = {}
        p = h = Node(0,None,None)
        q = head
        while q:
            p.next = Node(q.val,None,None)
            p = p.next
            p.next = None
            d[q] = p
            q = q.next
            
        p,q = h.next,head
        while q:
            if q.random: p.random = d[q.random]
            p,q = p.next,q.next

        return h.next


Next challenges:
Clone Graph

146. LRU Cache
Medium

3394

126

Favorite

Share
Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and put.

get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
put(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

The cache is initialized with a positive capacity.

Follow up:
Could you do both operations in O(1) time complexity?

from collections import OrderedDict 
class LRUCache(object):
    def __init__(self, size):
        self.size = size
        self.cache = OrderedDict()

    def get(self, key):
        value = self.cache.pop(key, None)
        if value is None:
            return -1
        self.cache[key] = value
        return value
    
    def set(self, key, value):     
        if not self.cache.pop(key, None) and self.size == len(self.cache):
            self.cache.popitem(last=False)
        self.cache[key] = value
  Next challenges:
LFU Cache
Design In-Memory File System
Design Compressed String Iterator

148.

Sort a linked list in O(n log n) time using constant space complexity.

Example 1:

Input: 4->2->1->3
Output: 1->2->3->4
Example 2:

Input: -1->5->3->4->0
Output: -1->0->3->4->5
class Solution:
    
    def sortList(self, head):
        if not head or not head.next:
            return head
        fast = slow = head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        slow.next, slow = None, slow.next
        head = self.sortList(head)
        slow = self.sortList(slow)
        return self.merge(head, slow)

    def merge(self, a, b):
        if not a:
            return b
        root = ListNode(0)
        root.next, a = a, root
        while a.next:
            if a.next.val > b.val:
                tmp = a.next
                a.next = b
                b = tmp
            a = a.next
        a.next = b
        return root.next


150. Evaluate Reverse Polish Notation
Medium

590

372

Favorite

Share
Evaluate the value of an arithmetic expression in Reverse Polish Notation.

Valid operators are +, -, *, /. Each operand may be an integer or another expression.

Note:

Division between two integers should truncate toward zero.
The given RPN expression is always valid. That means the expression would always evaluate to a result and there won't be any divide by zero operation.
Example 1:

Input: ["2", "1", "+", "3", "*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9

Next challenges:
Insertion Sort List


class Solution(object):
    def evalRPN(self, tokens):
        s = []
        for t in tokens:
            if t in ["+", "-", "*", "/"]:
                b = s.pop()
                a = s.pop()
                if t == "+":
                    s.append(a + b)
                if t == "-":
                    s.append(a - b)
                if t == "*":
                    s.append(a * b)
                if t == "/":
                    s.append(int(float(a)/b))
            else:
                s.append(int(t))
        return s.pop()
        
        
 

Next challenges:
Basic Calculator
Expression Add Operators


       
  152. Maximum Product Subarray
Medium

2389

111

Favorite

Share
Given an integer array nums, find the contiguous subarray within an array (containing at least one number) which has the largest product.

Example 1:

Input: [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
Example 2:

Input: [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.      
  class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_dp = [0]*len(nums)
        min_dp = [0]*len(nums)
        max_dp[0], min_dp[0] = nums[0], nums[0]
        for i in range(1, len(nums)):
            max_dp[i] = max(max_dp[i-1]*nums[i], min_dp[i-1]*nums[i], nums[i])
            min_dp[i] = min(max_dp[i-1]*nums[i], min_dp[i-1]*nums[i], nums[i])
        return max(max_dp)
        
        
        Next challenges:
Maximum Product of Three Numbers
Subarray Product Less Than K


166. Fraction to Recurring Decimal
Medium

517

1131

Favorite

Share
Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.

If the fractional part is repeating, enclose the repeating part in parentheses.

Example 1:

Input: numerator = 1, denominator = 2
Output: "0.5"
Example 2:

Input: numerator = 2, denominator = 1
Output: "2"
	
	class Solution(object):
    def fractionToDecimal(self, numerator, denominator):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        """
        if numerator>0 and denominator<0 or numerator<0 and denominator>0:
            ans = "-"
        else:
            ans = ""
        numerator,denominator = abs(numerator),abs(denominator)
        ans += str(numerator//denominator)
        numerator %= denominator
        if numerator == 0:
            return ans
        ans += "."    
        prevNumerators = {}
        while numerator not in prevNumerators:
            prevNumerators[numerator] = len(ans)
            if numerator == 0:
                return ans
            else:
                ans += str(10*numerator//denominator)
                numerator = 10*numerator % denominator
        patternStartIndex = prevNumerators[numerator]
        return ans[:patternStartIndex]+"("+ans[patternStartIndex:]+")"
        
        
        Next challenges:
Strobogrammatic Number II
Island Perimeter
Convert to Base -2


300. Longest Increasing Subsequence
Medium

2849

68

Favorite

Share
Given an unsorted array of integers, find the length of longest increasing subsequence.

Example:

Input: [10,9,2,5,3,7,101,18]
Output: 4 
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4. 
Note:

There may be more than one LIS combination, it is only necessary for you to return the length.
Your algorithm should run in O(n2) complexity.



class Solution:
    def lengthOfLIS(self, S):
        if not S: return 0
        LS=[1]*len(S) #Longest Subsequences such that the last element is element i of S
        for i in range(len(S)):
            for j in range(i):
                if S[i]>S[j] and LS[j]>=LS[i]: LS[i]=LS[j]+1

        return max(LS)
        
class Solution:
    def lengthOfLIS(self, nums):
        tails = [0] * len(nums)
        size = 0
        for x in nums:
            i, j = 0, size
            while i != j:
                m = int((i + j) / 2)
                if tails[m] < x:
                    i = m + 1
                else:
                    j = m
            tails[i] = x
            size = max(i + 1, size)
        return size
    
    
    
    
    
    
    #def lengthOfLIS(self, S):
        #if not S: return 0
        #LS=[1]*len(S) #Longest Subsequences such that the last element is element i of S
        #print(LS)
        #for i in range(len(S)):
            #for j in range(i):
                #if S[i]>S[j] and LS[j]>=LS[i]: LS[i]=LS[j]+1

        #return max(LS)
Next challenges:
Increasing Triplet Subsequence
Russian Doll Envelopes
Number of Longest Increasing Subsequence
Minimum ASCII Delete Sum for Two Strings


208. Implement Trie (Prefix Tree)
Medium

1781

35

Favorite

Share
Implement a trie with insert, search, and startsWith methods.

Example:

Trie trie = new Trie();

trie.insert("apple");
trie.search("apple");   // returns true
trie.search("app");     // returns false
trie.startsWith("app"); // returns true
trie.insert("app");   
trie.search("app");     // returns true


class TrieNode:
    def __init__(self):
        self.flag = False
        self.children = {}
    
class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        cur = self.root
        for c in word:
            if not c in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.flag = True
        
    def search(self, word):
        res, node = self.childSearch(word)
        if res:
            return node.flag
        return False

    def startsWith(self, prefix):
        return self.childSearch(prefix)[0]
        
    def childSearch(self, word):
        cur = self.root
        for c in word:
            if c in cur.children:
                cur = cur.children[c]
            else:
                return False, None
        return True, cur
  Next challenges:
Add and Search Word - Data structure design
Design Search Autocomplete System
Replace Words
Implement Magic Dictionary



334. Increasing Triplet Subsequence
Medium

947

95

Favorite

Share
Given an unsorted array return whether an increasing subsequence of length 3 exists or not in the array.

Formally the function should:

Return true if there exists i, j, k 
such that arr[i] < arr[j] < arr[k] given 0 ≤ i < j < k ≤ n-1 else return false.
Note: Your algorithm should run in O(n) time complexity and O(1) space complexity.



class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        return self.increasingKlet(nums, 3)

    def increasingKlet(self, nums: List[int], k) -> bool:
        '''
        Approach: start with k-1 very large values, as soon as we 
        find a number bigger than all k-1, return true.
        Time: O(n*k)
        Space: O(k)
        this is the generic solution for this problem
        '''
        small_arr = [math.inf] * (k - 1)

        for num in nums:
            for i in range(k-1):
                if num <= small_arr[i]:
                    small_arr[i] = num
                    break

            if num > small_arr[-1]:
                return True

        return False
        
  Next challenges:
Similar RGB Color
Cousins in Binary Tree
Check If a Number Is Majority Element in a Sorted Array

395. Longest Substring with At Least K Repeating Characters
Medium

821

76

Favorite

Share
Find the length of the longest substring T of a given string (consists of lowercase letters only) such that every character in T appears no less than k times.

Example 1:

Input:
s = "aaabb", k = 3

Output:
3

The longest substring is "aaa", as 'a' is repeated 3 times.

class Solution(object):
    
    def longestSubstring(self, s, k):
        
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        lookup = collections.Counter(s)
        
        for c in lookup:
            print("c: ",c,s.split(c))
            if lookup[c] < k:
                return max(self.longestSubstring(t, k) for t in s.split(c))
        return len(s)
        
        
        Perfect Rectangle
Palindromic Substrings
Employee Free Time


116. Populating Next Right Pointers in Each Node
Medium

1137

96

Favorite

Share
You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.


324. Wiggle Sort II
Medium

677

379

Favorite

Share
Given an unsorted array nums, reorder it such that nums[0] < nums[1] > nums[2] < nums[3]....

Example 1:

Input: nums = [1, 5, 1, 1, 6, 4]
Output: One possible answer is [1, 4, 1, 5, 1, 6].
Example 2:

Input: nums = [1, 3, 2, 2, 3, 1]
Output: One possible answer is [2, 3, 1, 3, 1, 2].

class Solution:
    def wiggleSort(self, nums):
        arr = sorted(nums)
        for i in range(1, len(nums), 2): nums[i] = arr.pop() 
        for i in range(0, len(nums), 2): nums[i] = arr.pop() 
        
  Wiggle Sort
  
  
  
  116. Populating Next Right Pointers in Each Node
Medium

1137

96

Favorite

Share
You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

class Solution:
	def connect(self, root: 'Node') -> 'Node':
		if not root or not root.left:
			return root
		root.left.next=root.right
		if root.next:
			root.right.next=root.next.left
		self.connect(root.left)
		self.connect(root.right)
		return root
		
210. Course Schedule II
Medium

1144

79

Favorite

Share
There are a total of n courses you have to take, labeled from 0 to n-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.

There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.

Example 1:

Input: 2, [[1,0]] 
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished   
             course 0. So the correct course order is [0,1] .
             
             class Solution(object):
    """
        This is similar to finding a cycle in a graph and at the same time,
        doing a Topological sort of the vertices. The approach here uses DFS.
    """
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        #If there are no dependencies among courses, they can 
        #be taken in any order. Weird input/output case tbh.
        if prerequisites == []:
            return [i for i in xrange(numCourses)]
        
        #Convert the edges into an adjecency list representation
        self.adj = [set() for i in xrange(numCourses)]
        
        for i,j in prerequisites:
            self.adj[j].add(i)
        
        #visited set will track vertices seen while exploring the current vertex
        self.visited = set()
        
        #completed set will track vertices that have been completely explored
        self.completed = set()
        
        self.res = []
        
        #For every vertex that has not been explored, visit it
        for i in xrange(len(self.adj)):
            if i not in self.visited and i not in self.completed:
                possible = self.visit(i)
                
                #visit() returns False if a cycle is detected
                if not possible:
                    return []
        
        #Topological sort is the reverse of the order in which the 
        #vertices were explored
        return self.res[::-1]
        
    def visit(self, u):
        #mark the current vertex as visited
        self.visited.add(u)
        possible = True
        
        #For every vertex adjecent to v
        for v in self.adj[u]:
            
            #explore the vertex only if not already explored
            if v not in self.visited and v not in self.completed:
                possible = possible and self.visit(v)
            
            #if this vertex was seen during the exploration of current vertex,
            #then there is a cycle in the graph and we can return False
            if v in self.visited:
                possible = False
                break
            
        #finally, we can mark the current vertex as completely explored
        self.visited.remove(u)
        self.completed.add(u)
        
        #If no cycles were found, we can add current vertex to sort order
        if possible:
            self.res.append(u)
            
        return possible
        
        Course Schedule
Alien Dictionary
Minimum Height Trees
Sequence Reconstruction
Course Schedule III

207. Course Schedule
Medium

2089

97

Favorite

Share
There are a total of n courses you have to take, labeled from 0 to n-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

Example 1:

Input: 2, [[1,0]] 
Output: true
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0. So it is possible.

class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        # backtracking to see if a course is its own ancestor
        edges = {}
        for course, req in prerequisites:
            if course not in edges:
                edges[course] = [req]
            else:
                edges[course].append(req)

        def helper(path,course):
            for req in edges[course]:
                if req in path:
                    return False
                if req in edges and req not in cleared:
                    path.add(req)
                    flag = helper(path,req)
                    path.remove(req)
                    if not flag: return False
                    else: cleared.add(req)
            return True
        
        cleared = set()
        for course in edges.keys():
            if not helper(set(),course):
                return False
        return True
        
        
 class Solution(object):
    """
        This is an instance of finding a cycle in a graph. We can do simple DFS to detect it.
    """
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        if numCourses == 1:
            return True
        
        #Convert the edges into an adjecency list representation
        self.adj = [set() for i in xrange(numCourses)]
        
        for i,j in prerequisites:
            self.adj[i].add(j)
        
        #visited set will track vertices seen while exploring the current vertex
        self.visited = set()
        
        #completed set will track vertices that have been completely explored
        self.completed = set()
        
        #For every vertex that has not been explored, visit it
        for i in xrange(len(self.adj)):
            if i not in self.visited and i not in self.completed:
                possible = self.visit(i)
                if not possible:
                    return False
        
        #when all the vertices have been explored without cycles, we can return True
        return True
    
    def visit(self, v):
        #mark the current vertex as visited
        self.visited.add(v)
        possible = True
        
        #For every vertex adjecent to v
        for u in self.adj[v]:
            
            #explore the vertex only if not already explored
            if u not in self.completed and u not in self.visited:
                possible = possible and self.visit(u)
            
            #if this vertex was seen during the exploration of current vertex,
            #then there is a cycle in the graph and we can return False
            if u in self.visited:
                possible = False
                break
        
        #finally, we can mark the current vertex as completely explored
        self.visited.remove(v)
        self.completed.add(v)
        
        return possible
        
Next challenges:
Graph Valid Tree
Minimum Height Trees
Course Schedule III

91. Decode Ways
Medium

1582

1831

Favorite

Share
A message containing letters from A-Z is being encoded to numbers using the following mapping:

'A' -> 1
'B' -> 2
...
'Z' -> 26
class Solution:
    def numDecodings(self, s):
        ct,ct1,ct2 = 1,0,0
        for ch in s:
            nct = ct * (ch > "0") + ct1 + ct2 * ("0" <= ch <= "6")
            nct1 = ct * (ch == "1")
            nct2 = ct * (ch == "2")
            ct, ct1, ct2 = nct, nct1, nct2
        return ct

Next challenges:
Decode Ways II


227. Basic Calculator II
Medium

804

135

Favorite

Share
Implement a basic calculator to evaluate a simple expression string.

The expression string contains only non-negative integers, +, -, *, / operators and empty spaces . The integer division should truncate toward zero.


class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = s.replace(" ", "")
        pn = [1 if c=="+" else -1 for c in s if c in "+-"] # order of +- signs
        sList = [self.cal(c) for c in s.replace("-", "+").split("+")]

        return sList[0] + sum([sList[i+1]*pn[i] for i in xrange(len(pn))])


    def cal(self, s): # calculate the values of substrings "WITHOUT +-"
        if "*" not in s and "/" not in s:
            return int(s)

        md = [1 if c=="*" else -1 for c in s if c in "*/"] # order of */ signs
        sList = [int(i) for i in s.replace("/", "*").split("*")]

        res, i = sList[0], 0
        while res != 0 and i < len(md):
            if md[i] == 1:
                res *= sList[i+1]
            else:
                res //= sList[i+1]
            i += 1
        return res

Next challenges:
Basic Calculator
Expression Add Operators
Basic Calculator III


322. Coin Change
Medium

2091

81

Favorite

Share
You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

Example 1:

Input: coins = [1, 2, 5], amount = 11
Output: 3 
Explanation: 11 = 5 + 5 + 1
Example 2:

Input: coins = [2], amount = 3
Output: -1

class Solution:
    def coinChange(self, coins, amount):
        ways = [float('inf')] * (amount + 1)
        ways[0] = 0
        
        for c in coins:
            for a in range(len(ways)):
                if c <= a:
                    ways[a] = min(ways[a-c]+1, ways[a])
                    
        
        if ways[amount] == float('inf'):
            return -1
        return ways[amount]   
        
  Next challenges:
Minimum Cost For Tickets


179. Largest Number
Medium

1145

147

Favorite

Share
Given a list of non negative integers, arrange them such that they form the largest number.

Example 1:

Input: [10,2]
Output: "210"
Example 2:

Input: [3,30,34,5,9]
Output: "9534330"
	
	class Solution(object):
    def largestNumber(self, nums):
        if not nums:
            return 0

        def compare(n1, n2):
            if int(n1+n2)>int(n2+n1):
                return -1
            else:
                return 1

        nums = map(str, nums)
        nums.sort(cmp = compare)
        return str(int(''.join(nums)))   
Next challenges:
H-Index
Count of Smaller Numbers After Self
Matrix Cells in Distance Order
