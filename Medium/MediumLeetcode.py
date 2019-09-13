# ~ 5. Longest Palindromic Substring
# ~ Medium

# ~ 4046

# ~ 375

# ~ Favorite

# ~ Share
# ~ Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

# ~ Example 1:

# ~ Input: "babad"
# ~ Output: "bab"
# ~ Note: "aba" is also a valid answer.
# ~ Example 2:

# ~ Input: "cbbd"
# ~ Output: "bb"


class Solution(object):
    def longestPalindrome(self, s):
        """
        type s: str
        rtype: str
        """
        longest = ""
        i = 0
        #here i represent the center position of the string
        while i<len(s):
            l = i
            r = i
			# find the duplicated string
            while(r+1<len(s) and s[r]==s[r+1]):
                
                r = r+1
			# jump to the right side of the duplicated string in next iteration
            i = r+1
            ans = find_longest(s,l,r)
            if len(longest)< len(ans):
                longest = ans
        return longest
    
def find_longest(s,l,r):
    while(l>=0 and r<len(s) and s[l]==s[r]):
        l = l-1
        r = r+1
    return s[l+1:r]
    
    
    
  # ~ 6. ZigZag Conversion
# ~ Medium

# ~ 1137

# ~ 3468

# ~ Favorite

# ~ Share
# ~ The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

# ~ P   A   H   N
# ~ A P L S I I G
# ~ Y   I   R
# ~ And then read line by line: "PAHNAPLSIIGYIR"

# ~ Write the code that will take a string and make this conversion given a number of rows:

# ~ string convert(string s, int numRows);
# ~ Example 1:

# ~ Input: s = "PAYPALISHIRING", numRows = 3
# ~ Output: "PAHNAPLSIIGYIR"


# ~ 8. String to Integer (atoi)
# ~ Medium

# ~ 1035

# ~ 6472

# ~ Favorite

# ~ Share
# ~ Implement atoi which converts a string to an integer.

# ~ The function first discards as many whitespace characters as necessary until the first non-whitespace character is found. Then, starting from this character, takes an optional initial plus or minus sign followed by as many numerical digits as possible, and interprets them as a numerical value.

# ~ The string can contain additional characters after those that form the integral number, which are ignored and have no effect on the behavior of this function.

# ~ If the first sequence of non-whitespace characters in str is not a valid integral number, or if no such sequence exists because either str is empty or it contains only whitespace characters, no conversion is performed.

# ~ If no valid conversion could be performed, a zero value is returned.

# ~ Note:

# ~ Only the space character ' ' is considered as whitespace character.
# ~ Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. If the numerical value is out of the range of representable values, INT_MAX (231 − 1) or INT_MIN (−231) is returned.
# ~ Example 1:

# ~ Input: "42"
# ~ Output: 42
# ~ Example 2:

# ~ Input: "   -42"
# ~ Output: -42
# ~ Explanation: The first non-whitespace character is '-', which is the minus sign.
             # ~ Then take as many numerical digits as possible, which gets 42.
# ~ Example 3:

# ~ Input: "4193 with words"
# ~ Output: 4193
# ~ Explanation: Conversion stops at digit '3' as the next character is not a numerical digit.
# ~ Example 4:

# ~ Input: "words and 987"
# ~ Output: 0
# ~ Explanation: The first non-whitespace character is 'w', which is not a numerical 
             # ~ digit or a +/- sign. Therefore no valid conversion could be performed.
# ~ Example 5:

# ~ Input: "-91283472332"
# ~ Output: -2147483648
# ~ Explanation: The number "-91283472332" is out of the range of a 32-bit signed integer.
             # ~ Thefore INT_MIN (−231) is returned.
class Solution:
    # @return an integer
    def myAtoi(self, str):
        str = str.strip()
        str = re.findall('(^[\+\-0]*\d+)\D*', str)

        try:
            result = int(''.join(str))
            MAX_INT = 2**31 -1
            MIN_INT = -2**31 
            if result > MAX_INT > 0:
                return MAX_INT
            elif result < MIN_INT < 0:
                return MIN_INT
            else:
                return result
        except:
            return 0




        
# ~ 9. Palindrome Number
# ~ Easy

# ~ 1549


class Solution(): 
    def twoSum(self, nums, target):
        lookup = {nums[i]:i for i in range(len(nums))}
        for i in range(len(nums)):
            complement = target - nums[i]
            j = lookup.get(complement)#hash table to search 
            if j != None and j != i: 
                return [i, j]
        return [] 


# ~ 1343

# ~ Favorite

# ~ Share
# ~ Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.

# ~ Example 1:

# ~ Input: 121
# ~ Output: true
# ~ Example 2:

# ~ Input: -121
# ~ Output: false
# ~ Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.

class Solution(object):
    def isPalindrome(self, x):
        if x < 0:
            return False
        p, res = x, 0
        while p:
            res = res * 10 + p % 10
            p /= 10
        return res == x    
        
        
 # ~ 12. Integer to Roman
# ~ Medium

# ~ 644

# ~ 2012

# ~ Favorite

# ~ Share
# ~ Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

# ~ Symbol       Value
# ~ I             1
# ~ V             5
# ~ X             10
# ~ L             50
# ~ C             100
# ~ D             500
# ~ M             1000
# ~ For example, two is written as II in Roman numeral, just two one's added together. Twelve is written as, XII, which is simply X + II. The number twenty seven is written as XXVII, which is XX + V + II.

# ~ Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

# ~ I can be placed before V (5) and X (10) to make 4 and 9. 
# ~ X can be placed before L (50) and C (100) to make 40 and 90. 
# ~ C can be placed before D (500) and M (1000) to make 400 and 900.
# ~ Given an integer, convert it to a roman numeral. Input is guaranteed to be within the range from 1 to 3999.

# ~ Example 1:

# ~ Input: 3
# ~ Output: "III"
# ~ Example 2:

# ~ Input: 4
# ~ Output: "IV"
# ~ Example 3:

# ~ Input: 9
# ~ Output: "IX"
# ~ Example 4:

# ~ Input: 58
# ~ Output: "LVIII"
# ~ Explanation: L = 50, V = 5, III = 3.
                
             
    
    
    
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        maxVal= 0
        # Use two pointer: head pointer and tail pointer
        i = 0;j = len(height)-1
        while i != j:
            currVal = j-i
            if height[i] < height[j]:
                currVal, i, j = currVal * height[i], i+1, j 
            else :
                currVal, i, j = currVal * height[j], i, j-1 
            maxVal = currVal if currVal > maxVal else maxVal
        return maxVal;
        
        
# ~ 12. Integer to Roman
# ~ Medium

# ~ 644

# ~ 2012

# ~ Favorite

# ~ Share
# ~ Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

# ~ Symbol       Value
# ~ I             1
# ~ V             5
# ~ X             10
# ~ L             50
# ~ C             100
# ~ D             500
# ~ M             1000
# ~ For example, two is written as II in Roman numeral, just two one's added together. Twelve is written as, XII, which is simply X + II. The number twenty seven is written as XXVII, which is XX + V + II.

# ~ Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

# ~ I can be placed before V (5) and X (10) to make 4 and 9. 
# ~ X can be placed before L (50) and C (100) to make 40 and 90. 
# ~ C can be placed before D (500) and M (1000) to make 400 and 900.
# ~ Given an integer, convert it to a roman numeral. Input is guaranteed to be within the range from 1 to 3999.

# ~ Example 1:

# ~ Input: 3
# ~ Output: "III"
# ~ Example 2:

# ~ Input: 4
# ~ Output: "IV"
# ~ Example 3:

# ~ Input: 9
# ~ Output: "IX"
# ~ Example 4:

# ~ Input: 58
# ~ Output: "LVIII"
# ~ Explanation: L = 50, V = 5, III = 3.
# ~ Example 5:

# ~ Input: 1994
# ~ Output: "MCMXCIV"
# ~ Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
        
class Solution:
    def intToRoman(self, num):
            values=['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I']
            nums=[1000,900,500,400,100,90,50,40,10,9,5,4,1]
            i=0
            ans=[]
            while(num>0):
                if num-nums[i]>=0:
                    ans.append(values[i])
                    num=num-nums[i]
                else: i=i+1          
            return ''.join(ans)
                 
# ~ 13. Roman to Integer
# ~ Easy

# ~ 1366

# ~ 2776

# ~ Favorite

# ~ Share
# ~ Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

# ~ Symbol       Value
# ~ I             1
# ~ V             5
# ~ X             10
# ~ L             50
# ~ C             100
# ~ D             500
# ~ M             1000
# ~ For example, two is written as II in Roman numeral, just two one's added together. Twelve is written as, XII, which is simply X + II. The number twenty seven is written as XXVII, which is XX + V + II.

# ~ Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

# ~ I can be placed before V (5) and X (10) to make 4 and 9. 
# ~ X can be placed before L (50) and C (100) to make 40 and 90. 
# ~ C can be placed before D (500) and M (1000) to make 400 and 900.
# ~ Given a roman numeral, convert it to an integer. Input is guaranteed to be within the range from 1 to 3999.

# ~ Example 1:

# ~ Input: "III"
# ~ Output: 3
# ~ Example 2:

# ~ Input: "IV"
# ~ Output: 4
# ~ Example 3:

# ~ Input: "IX"
# ~ Output: 9
# ~ Example 4:

# ~ Input: "LVIII"
# ~ Output: 58
# ~ Explanation: L = 50, V= 5, III = 3.
# ~ Example 5:

# ~ Input: "MCMXCIV"
# ~ Output: 1994
# ~ Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
 
             
class Solution(object):
	def romanToInt(self, s):
		"""
		:type s: str
		:rtype: int
		"""
		return_val=0
		pair={"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
		for i in range(len(s)):
			return_val+=pair[s[i]]

			if i==0:continue

			# Subtract the amount already added (times 2) if next number is bigger
			if pair[s[i]]>pair[s[i-1]]:
				return_val-=pair[s[i-1]]*2
		return return_val
		


# ~ 14. Longest Common Prefix
# ~ Easy



# ~ Favorite

# ~ Share
# ~ Write a function to find the longest common prefix string amongst an array of strings.

# ~ If there is no common prefix, return an empty string "".

# ~ Example 1:

# ~ Input: ["flower","flow","flight"]
# ~ Output: "fl"
# ~ Example 2:

# ~ Input: ["dog","racecar","car"]
# ~ Output: ""
# ~ Explanation: There is no common prefix among the input strings.
		
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        lcp = ""
        for s in zip(*strs):
            if (s[0],) * len(s) == s:
                lcp += s[0]
            else:
                break
        return lcp
        
        
 # ~ 17. Letter Combinations of a Phone Number
# ~ Medium

# ~ 2430

# ~ 325

# ~ Favorite

# ~ Share
# ~ Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.

# ~ A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.



# ~ Example:

# ~ Input: "23"
# ~ Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].


# ~ 18. 4Sum
# ~ Medium

# ~ 1183

# ~ 237

# ~ Favorite

# ~ Share
# ~ Given an array nums of n integers and an integer target, are there elements a, b, c, and d in nums such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.

# ~ Note:

# ~ The solution set must not contain duplicate quadruplets.

# ~ Example:

# ~ Given array nums = [1, 0, -1, 0, -2, 2], and target = 0.

# ~ A solution set is:
# ~ [
  # ~ [-1,  0, 0, 1],
  # ~ [-2, -1, 1, 2],
  # ~ [-2,  0, 0, 2]
# ~ ]

class Solution:
    def fourSum(self, nums, target):
        
        
        # Sort the initial list
        nums.sort()

        # HashMap for the solution, to avoid duplicates
        solution = {}

        # i = 0 ..... n-1
        for i in range( len(nums) ):
            #j = i+1 ..... n-1
            for j in range( i+1, len( nums ) ):
                
            # Two pointer approach to find the remaining two elements
                start = j+1
                end = len(nums) - 1
                while ( start < end ):
                    temp = nums[i] + nums[j] + nums[start] + nums[end]
                    
                    if ( temp == target ):
                        solution[ nums[i], nums[j], nums[start], nums[end] ] = True
                        start +=1
                        end -=1
                    elif temp < target:
                        start+=1
                    else:
                        end -=1
        
        return solution.keys()
        
       


# ~ 21. Merge Two Sorted Lists
# ~ Easy

# ~ 2477

# ~ 357

# ~ Favorite

# ~ Share
# ~ Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.

# ~ Example:

# ~ Input: 1->2->4, 1->3->4
# ~ Output: 1->1->2->3->4->4

# Time:  O(n)
# Space: O(1)
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    # iteratively
    def mergeTwoLists(self, l1, l2):
        dummy = cur = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 or l2
        return dummy.next
        
 # ~ 26. Remove Duplicates from Sorted Array
# ~ Easy

# ~ 1658

# ~ 3556

# ~ Favorite

# ~ Share
# ~ Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length.

# ~ Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

# ~ Example 1:

# ~ Given nums = [1,1,2],

# ~ Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively.

# ~ It doesn't matter what you leave beyond the returned length.
        
 class Solution:
    def removeDuplicates(self, nums):
        i=0
        while i<len(nums)-1:
            if nums[i]==nums[i+1]:
                nums.remove(nums[i])
            else:i+=1
        return len(nums)
        
 
# ~ 27. Remove Element
# ~ Easy

# ~ 906

# ~ 1923

# ~ Favorite

# ~ Share
# ~ Given an array nums and a value val, remove all instances of that value in-place and return the new length.

# ~ Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

# ~ The order of elements can be changed. It doesn't matter what you leave beyond the new length.

# ~ Example 1:

# ~ Given nums = [3,2,2,3], val = 3,

# ~ Your function should return length = 2, with the first two elements of nums being 2.

# ~ It doesn't matter what you leave beyond the returned length. 
 class Solution(object):
    def removeElement(self, nums, val):
        for i in range(nums.count(val)):
            nums.pop(nums.index(val))
        return len(nums)
        
        
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if needle=="":
            return 0
        
        len_needle = len(needle)
        
        for i in range(0, len(haystack)-len_needle+1):
            if haystack[i:i+len_needle] == needle:
                return i
        return -1
        
        
  
# ~ 35. Search Insert Position
# ~ Easy

# ~ 1450

# ~ 193

# ~ Favorite

# ~ Share
# ~ Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

# ~ You may assume no duplicates in the array.

# ~ Example 1:

# ~ Input: [1,3,5,6], 5
# ~ Output: 2
            
class Solution:
    def searchInsert(self, nums, target):
        if not nums: return 0
        for i in range(len(nums)):
            if nums[i]==target or nums[i]>target:return i
        return len(nums)
        
# ~ 98. Validate Binary Search Tree
# ~ Medium

# ~ 2183

# ~ 327

# ~ Favorite

# ~ Share
# ~ Given a binary tree, determine if it is a valid binary search tree (BST).

# ~ Assume a BST is defined as follows:

# ~ The left subtree of a node contains only nodes with keys less than the node's key.
# ~ The right subtree of a node contains only nodes with keys greater than the node's key.
# ~ Both the left and right subtrees must also be binary search trees.
 

# ~ Example 1:

    # ~ 2
   # ~ / \
  # ~ 1   3

# ~ Input: [2,1,3]
# ~ Output: true


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def isValidBST(self, root):
        return self.helper(root, float('-inf'), float('inf'))
    
    def helper(self, node, minVal, maxVal):
        if node == None:
            return True
        
        if node.val <= minVal or node.val >= maxVal:
            return False
        
        left = self.helper(node.left, minVal, node.val)
        right = self.helper(node.right, node.val, maxVal)
        
        return left and right         
        
        
 stack, prev =  [], None
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if prev is not None and root is not None and root.val <= prev.val:
                return False
            prev, root = root, root.right
            

# Python program to demonstrate insert operation in binary search tree  
  
# A utility class that represents an individual node in a BST 
class Node: 
    def __init__(self,key): 
        self.left = None
        self.right = None
        self.val = key 
  
# A utility function to insert a new node with the given key 
def insert(root,node): 
    if root is None: 
        root = node 
    else: 
        if root.val < node.val: 
            if root.right is None: 
                root.right = node 
            else: 
                insert(root.right, node) 
        else: 
            if root.left is None: 
                root.left = node 
            else: 
                insert(root.left, node) 
  
# A utility function to do inorder tree traversal 
def inorder(root): 
    if root: 
        inorder(root.left) 
        print(root.val) 
        inorder(root.right) 
  
  
# Driver program to test the above functions 
# Let us create the following BST 
#      50 
#    /      \ 
#   30     70 
#   / \    / \ 
#  20 40  60 80 
r = Node(50) 
insert(r,Node(30)) 
insert(r,Node(20)) 
insert(r,Node(40)) 
insert(r,Node(70)) 
insert(r,Node(60)) 
insert(r,Node(80)) 
  
# Print inoder traversal of the BST 
inorder(r)


https://www.geeksforgeeks.org/make-binary-search-tree/

Tree Traversals (Inorder, Preorder and Postorder)
Unlike linear data structures (Array, Linked List, Queues, Stacks, etc) which have only one logical way to traverse them, trees can be traversed in different ways. Following are the generally used ways for traversing trees.

Example Tree
Example Tree

Depth First Traversals:
(a) Inorder (Left, Root, Right) : 4 2 5 1 3
(b) Preorder (Root, Left, Right) : 1 2 4 5 3
(c) Postorder (Left, Right, Root) : 4 5 2 3 1

Breadth First or Level Order Traversal : 1 2 3 4 5
Please see this post for Breadth First Traversal.



# Python program to for tree traversals 
  
# A class that represents an individual node in a 
# Binary Tree 
class Node: 
    def __init__(self,key): 
        self.left = None
        self.right = None
        self.val = key 
  
  
# A function to do inorder tree traversal 
def printInorder(root): 
  
    if root: 
  
        # First recur on left child 
        printInorder(root.left) 
  
        # then print the data of node 
        print(root.val), 
  
        # now recur on right child 
        printInorder(root.right) 
  
  
  
# A function to do postorder tree traversal 
def printPostorder(root): 
  
    if root: 
  
        # First recur on left child 
        printPostorder(root.left) 
  
        # the recur on right child 
        printPostorder(root.right) 
  
        # now print the data of node 
        print(root.val), 
  
  
# A function to do preorder tree traversal 
def printPreorder(root): 
  
    if root: 
  
        # First print the data of node 
        print(root.val), 
  
        # Then recur on left child 
        printPreorder(root.left) 
  
        # Finally recur on right child 
        printPreorder(root.right) 
  
  
# Driver code 
root = Node(1) 
root.left      = Node(2) 
root.right     = Node(3) 
root.left.left  = Node(4) 
root.left.right  = Node(5) 
print "Preorder traversal of binary tree is"
printPreorder(root) 
  
print "\nInorder traversal of binary tree is"
printInorder(root) 
  
print "\nPostorder traversal of binary tree is"
printPostorder(root) 


94. Binary Tree Inorder Traversal
Medium

1811

77

Favorite

Share
Given a binary tree, return the inorder traversal of its nodes' values.

Example:

Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,3,2]
Follow up: Recursive solution is trivial, could you do it iteratively?

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# iteratively
class Solution:
    def inorderTraversal(self, root):
        results, stack = [], []
        cur = root
        while cur or stack: # KEY: current node not none "or" stack is not empty
            while cur is not None:
                stack.append(cur)
                cur = cur.left # check to the leftmost leave
                print(cur)
            node = stack.pop()
            results.append(node.val) 
            cur = node.right # check if any right leave

        return results
        
# ~ 589. N-ary Tree Preorder Traversal
# ~ Easy

# ~ 273

# ~ 38

# ~ Favorite

# ~ Share
# ~ Given an n-ary tree, return the preorder traversal of its nodes' values.

# ~ For example, given a 3-ary tree:

 



 

# ~ Return its preorder traversal as: [1,3,5,6,2,4].

class Solution(object):
    def preorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        # iterative
        res = []
        q = [root]
        while q:
            cur = q.pop(0)
            if cur:
                res.append(cur.val)
                q = cur.children + q
        return res
       
      
      
      
# ~ 429. N-ary Tree Level Order Traversal
# ~ Easy

# ~ 294

# ~ 34

# ~ Favorite

# ~ Share
# ~ Given an n-ary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

# ~ For example, given a 3-ary tree:
	# ~ We should return its level order traversal:

# ~ [
     # ~ [1],
     # ~ [3,2,4],
     # ~ [5,6]
# ~ ]



class Solution:
	def levelOrder(self, root):
		output = []
		if root is not None:
			output.append([root.val])
			self.trav(root.children, 0, output)
			output.pop()
		return output

	def trav(self, node, deep, output):
		deep += 1
		if (len(output) - 1 < deep): output.append([])
		for x in node:
			output[deep].append(x.val)
			if x.children is not None:
				self.trav(x.children, deep, output)
				
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        
        if not root:
            return []
        
        if not root.children:
            return [root.val]
        
        ret = [root.val]
        for c in root.children:
            ret += self.preorder(c)
        return ret


# ~ 101. Symmetric Tree
# ~ Easy

# ~ 2487

# ~ 53

# ~ Favorite

# ~ Share
# ~ Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

# ~ For example, this binary tree [1,2,2,3,4,4,3] is symmetric:

    # ~ 1
   # ~ / \
  # ~ 2   2
 # ~ / \ / \
# ~ 3  4 4  3
 

# ~ But the following [1,2,2,null,3,null,3] is not:

    # ~ 1
   # ~ / \
  # ~ 2   2
   # ~ \   \
   # ~ 3    3
   
200. Number of Islands
Medium

2925

104

Favorite

Share
Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
 You may assume all four edges of the grid are all surrounded by water.   
  class Solution(object):
    def numIslands(self, grid):
        def sink(i, j):
            if 0 <= i < len(grid) and 0 <= j < len(grid[i]) and grid[i][j] == '1':
                grid[i][j] = '0'
                list(map(sink, (i+1, i-1, i, i), (j, j, j+1, j-1)))  # map in python3 return iterator
                return 1
            return 0
        return sum(sink(i, j) for i in range(len(grid)) for j in range(len(grid[i])))

https://leetcode.com/problems/lru-cache/discuss/317200/Python-solution-explaining-itself
from collections import Counter
class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        a=[]
        l=len(p)
        cp=Counter(p)
        cs=Counter(s[:l-1])
        print(cs)
        i=0
        while i+l<=len(s):
            cs[s[i+l-1]]+=1
            if cs==cp:
                a.append(i)
            cs[s[i]]-=1
            if cs[s[i]]==0:
                del cs[s[i]]
            i+=1
        return a
        
        
class Solution(object):
    # Iteratively
    def swapPairs(self, head):
        dummy = p = ListNode(0)
        dummy.next = head
        while head and head.next:
            tmp = head.next
            head.next = tmp.next
            tmp.next = head
            p.next = tmp
            head = head.next
            p = tmp.next
        return dummy.next

Given a linked list, rotate the list to the right by k places, where k is non-negative.

Example 1:

Input: 1->2->3->4->5->NULL, k = 2
Output: 4->5->1->2->3->NULL
Explanation:
rotate 1 steps to the right: 5->1->2->3->4->NULL
rotate 2 steps to the right: 4->5->1->2->3->NULL
Example 2:

Input: 0->1->2->NULL, k = 4
Output: 2->0->1->NULL
Explanation:
rotate 1 steps to the right: 2->0->1->NULL
rotate 2 steps to the right: 1->2->0->NULL
rotate 3 steps to the right: 0->1->2->NULL
rotate 4 steps to the right: 2->0->1->NULL

class Solution:
	def rotateRight(self, head, k):
		if not head:
			return None
		v = []
		while head:
			v.append(head.val)
			head = head.next
		k = k % len(v)
		for i in range(k):
			v.insert(0, v.pop())
		res = ListNode(v[0])
		node = res
		for i in range(1, len(v)):
			node.next = ListNode(v[i])
			node = node.next
		return res
		
82. Remove Duplicates from Sorted List II
Medium

953

83

Favorite

Share
Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.

Example 1:

Input: 1->2->3->3->4->4->5
Output: 1->2->5
Example 2:

Input: 1->1->1->2->3
Output: 2->3


class Solution(object):
    def deleteDuplicates(self, head):
        dummy = pre = ListNode(0)
        dummy.next = head
        while head and head.next:
            if head.val == head.next.val:
                while head and head.next and head.val == head.next.val:
                    head = head.next
                head = head.next
                pre.next = head
            else:
                pre = pre.next
                head = head.next
        return dummy.next
        
86. Partition List
Medium

754

203

Favorite

Share
Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.

You should preserve the original relative order of the nodes in each of the two partitions.

Example:

Input: head = 1->4->3->2->5->2, x = 3
Output: 1->2->2->4->3->5


class Solution(object):
    def partition(self, head, x):
        h1 = l1 = ListNode(0)
        h2 = l2 = ListNode(0)
        while head:
            if head.val < x:
                l1.next = head
                l1 = l1.next
            else:
                l2.next = head
                l2 = l2.next
            head = head.next
        l2.next = None
        l1.next = h2.next
        return h1.next
        
Next challenges:
Sort Colors
Candy Crush
Long Pressed Name


92. Reverse Linked List II
Medium

1371

97

Favorite

Share
Reverse a linked list from position m to n. Do it in one-pass.

Note: 1 ≤ m ≤ n ≤ length of list.

Example:

Input: 1->2->3->4->5->NULL, m = 2, n = 4
Output: 1->4->3->2->5->NULL

class Solution(object):
    def reverseBetween(self, head, m, n):
        # Edge
        if m == n: return head
        if not head or not m or not n: return None
        
        # Set starting point
        dummy = ListNode(0)
        dummy.next = head
        start = dummy
        for i in range(m - 1):
            start = start.next
            
        # Set ending point
        end = cur = start.next        
        
        prev = None
        # reverse
        for i in range(n - m + 1): 
            next = cur.next
            cur.next = prev
            prev = cur
            cur = next

        # Connect
        start.next = prev
        end.next = cur
        return dummy.next
        
109. Convert Sorted List to Binary Search Tree
Medium

1143

71

Favorite

Share
Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

Example:

Given the sorted linked list: [-10,-3,0,5,9],

One possible answer is: [0,-3,9,-10,null,5], which represents the following height balanced BST:

      0
     / \
   -3   9
   /   /
 -10  5
 
 # Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # @param {ListNode} head
    # @return {TreeNode}
    def sortedListToBST(self, head):
        if not head:
            return None
        if not head.next:
            return TreeNode(head.val)

        dummyHead = ListNode(0)
        dummyHead.next = head
        slow, fast = dummyHead, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        root = TreeNode(slow.next.val)
        root.right = self.sortedListToBST(slow.next.next)
        slow.next = None
        root.left = self.sortedListToBST(head)

        return root
        
 
 
 445. Add Two Numbers II
Medium

809

108

Favorite

Share
You are given two non-empty linked lists representing two non-negative integers. The most significant digit comes first and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Follow up:
What if you cannot modify the input lists? In other words, reversing the lists is not allowed.

Example:

Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 8 -> 0 -> 7       
 class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        s1 = []
        s2 = []
        while (l1 or l2):
            if l1:
                s1.append(l1)
                l1 = l1.next
            if l2:
                s2.append(l2)
                l2 = l2.next
        carry = 0 
        resList = None
        while (s1 or s2 or carry):
            p = s1.pop().val if s1 else 0
            q = s2.pop().val if s2 else 0
            intsum = p + q + carry
            carry = int(intsum/10) 
            newdigit = ListNode(intsum%10)
            newdigit.next = resList
            resList = newdigit
        return resList 
