226. Invert Binary Tree
Easy

1940

32

Favorite

Share
Invert a binary tree.

Example:

Input:

     4
   /   \
  2     7
 / \   / \
1   3 6   9
Output:

     4
   /   \
  7     2
 / \   / \
9   6 3   1

class Solution:
    def invertTree(self, root):
        if root == None:
            return None
        r =  self.invertTree(root.right)
        l =  self.invertTree(root.left)
        root.right =l
        root.left= r
        return root


Next challenges:
Closest Binary Search Tree Value II
Construct Binary Search Tree from Preorder Traversal
Recover a Tree From Preorder Traversal

448. Find All Numbers Disappeared in an Array
Easy

1810

169

Favorite

Share
Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.

Find all the elements of [1, n] inclusive that do not appear in this array.

Could you do it without extra space and in O(n) runtime? You may assume the returned list does not count as extra space.

Example:

Input:
[4,3,2,7,8,2,3,1]

Output:
[5,6]


class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # You're looking the difference between the set of numbers between 1..n and the 
		# set of numbers provided, removing duplicates (python's set() will remove dups for you)
        return list(set(range(1,len(nums) + 1)) - set(nums))
  
  
  class Solution:
    def findDisappearedNumbers(self, n: List[int]) -> List[int]:
    	N = set(n)
    	return [i for i in range(1, len(n) + 1) if i not in N]
    	      
        Next challenges:
Find All Duplicates in an Array

543. Diameter of Binary Tree
Easy

1609

99

Favorite

Share
Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

Example:
Given a binary tree 
          1
         / \
        2   3
       / \     
      4   5    
Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].

Note: The length of path between two nodes is represented by the number of edges between them.

class Solution(object):
    def __init__(self):
		# Global var to keep track of the max as we recurse
        self.max = 0
    
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
		# Check for edge cases
        if not root or (not root.left and not root.right):
            return 0
        
		# self.max will be set in here
        self.getLength(root, 0)
        
        return self.max
        
        
    def getLength(self, root, currLength):
		# base case: will return the currentLength of the recursive call
        if not root:
            return currLength
            
		# Increment the length if it passes the base case
        currLength += 1
        
		# Get the right and left lengths from the current node.
		# It's important to note here that I am calling the recursive funciton with currLength == 0.
		# This is to make sure that if the max length is not at the root, it is caught.
        leftLength = self.getLength(root.left, 0)
        rightLength = self.getLength(root.right, 0)

		# If a max diameter is found at the current node, then set the new max! Woohoo
        if leftLength + rightLength > self.max:
            self.max = leftLength + rightLength
            
		# Return the max of the left and right branches from the current node and add the currLength
		# in order to bubble up the total length through the recursive calls.
        return max(leftLength, rightLength) + currLength
        
       Next challenges:
Leaf-Similar Trees
Range Sum of BST
Delete Nodes And Return Forest


437. Path Sum III
Easy

2098

118

Favorite

Share
You are given a binary tree in which each node contains an integer value.

Find the number of paths that sum to a given value.

The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).

The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.

Example:

root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

Return 3. The paths that sum to 8 are:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11


581. Shortest Unsorted Continuous Subarray
Easy

1613

74

Favorite

Share
Given an integer array, you need to find one continuous subarray that if you only sort this subarray in ascending order, then the whole array will be sorted in ascending order, too.

You need to find the shortest such subarray and output its length.

Example 1:
Input: [2, 6, 4, 8, 10, 9, 15]
Output: 5
Explanation: You need to sort [6, 4, 8, 10, 9] in ascending order to make the whole array sorted in ascending order.
Note:
Then length of the input array is in range [1, 10,000].
The input array may contain duplicates, so ascending order here means <=.


class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        sNums = sorted(nums)
        start = end = 0
        
        for i in range(len(nums)):
            if nums[i] != sNums[i]:
                start = i
                break

        for i in range(len(nums)-1, 0, -1):
            if nums[i] != sNums[i]:
                end = i
                break
        
        return end - start+1 if end - start else 0
        
        Next challenges:
Longest Continuous Increasing Subsequence
Available Captures for Rook
Relative Sort Array




Next challenges:
Path Sum
Path Sum II
Path Sum IV
Longest Univalue Path
