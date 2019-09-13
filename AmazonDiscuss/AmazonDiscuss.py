Date: August 2019
Position: SDE1

behavoir question
queue and stack
database, dynamoDB
545. Boundary of Binary Tree (premium)

# Python3 program for binary traversal of binary tree 
  
# A binary tree node 
class Node: 
  
    # Constructor to create a new node 
    def __init__(self, data): 
        self.data = data  
        self.left = None
        self.right = None
  
# A simple function to print leaf nodes of a Binary Tree 
def printLeaves(root): 
    if(root): 
        printLeaves(root.left) 
          
        # Print it if it is a leaf node 
        if root.left is None and root.right is None: 
            print(root.data), 
  
        printLeaves(root.right) 
  
# A function to print all left boundary nodes, except a  
# leaf node. Print the nodes in TOP DOWN manner 
def printBoundaryLeft(root): 
      
    if(root): 
        if (root.left): 
              
            # to ensure top down order, print the node 
            # before calling itself for left subtree 
            print(root.data) 
            printBoundaryLeft(root.left) 
          
        elif(root.right): 
            print (root.data) 
            printBoundaryLeft(root.right) 
          
        # do nothing if it is a leaf node, this way we 
        # avoid duplicates in output 
  
  
# A function to print all right boundary nodes, except 
# a leaf node. Print the nodes in BOTTOM UP manner 
def printBoundaryRight(root): 
      
    if(root): 
        if (root.right): 
            # to ensure bottom up order, first call for 
            # right subtree, then print this node 
            printBoundaryRight(root.right) 
            print(root.data) 
          
        elif(root.left): 
            printBoundaryRight(root.left) 
            print(root.data) 
  
        # do nothing if it is a leaf node, this way we  
        # avoid duplicates in output 
  
  
# A function to do boundary traversal of a given binary tree 
def printBoundary(root): 
    if (root): 
        print(root.data) 
          
        # Print the left boundary in top-down manner 
        printBoundaryLeft(root.left) 
  
        # Print all leaf nodes 
        printLeaves(root.left) 
        printLeaves(root.right) 
  
        # Print the right boundary in bottom-up manner 
        printBoundaryRight(root.right) 
  
  
# Driver program to test above function 
root = Node(20) 
root.left = Node(8) 
root.left.left = Node(4) 
root.left.right = Node(12) 
root.left.right.left = Node(10) 
root.left.right.right = Node(14) 
root.right = Node(22) 
root.right.right = Node(25) 
printBoundary(root) 


Position: SDE1
1 hr- two questions: No behavioural!

https://www.geeksforgeeks.org/subtract-two-numbers-represented-as-linked-lists
https://leetcode.com/problems/sort-colors

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
            
  Amazon | OA 2019 | Subtree with Maximum Average
1
Sithis's avatar
Sithis
Moderator
2790
Last Edit: August 17, 2019 3:09 PM

1.3K VIEWS

Given an N-ary tree, find the subtree with the maximum average. Return the root of the subtree.
A subtree of a tree is any node of that tree plus all its descendants. The average value of a subtree is the sum of its values, divided by the number of nodes. In this problem we only consider nodes which have at least 1 child.

Example 1:

Input:
		 20
	   /   \
	 12     18
  /  |  \   / \
11   2   3 15  8

Output: 18
Explanation:
There are 3 nodes which have children in this tree:
12 => (11 + 2 + 3 + 12) / 4 = 7
18 => (18 + 15 + 8) / 3 = 13.67
20 => (12 + 11 + 2 + 3 + 18 + 15 + 8 + 20) / 8 = 11.125

18 has the maximum average so output 18.
Similar questions:
	
	
Amazon | Is cheese reachable in the maze?

Amazon | Is cheese reachable in the maze?

Amazon | Distance between 2 nodes

Write a function that given a BST, it will return the distance (number of edges) between 2 nodes.

For example, given this tree

         A
        / \
       B   C
      / \   \
     D   E   F
    /         \
   G           H
The distance between G and E is 3: [G -> D -> B -> E]

The distance between G and H is 6: [G -> D -> B -> A -> C -> F -> H]

get depth of p
get depth of q
get LCA of p and q
dist(p,q) = depth(p) + depth(q) - 2 * dist(lca(p,q))
def distance_in_bst(root, p, q)
  depth(root, p) + depth(root, q) - 2 * depth(root, lca(root, p, q).val)
end

def depth(root, p, d=0)
  return if root.nil?
  return d if p == root.val
  depth(root.left, p, d+1) || depth(root.right, p, d+1)
end

def lca(root, p, q)
  if root.val < p
    lca(root.right, p, q)
  elsif root.val > q
    lca(root.left, p, q)
  else
    root
  end
end

Next challenges:
Minimum Depth of Binary Tree
Closest Binary Search Tree Value
Redundant Connection II

class Solution:
    def lowestCommonAncestor(self, root, p, q) :
        if root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right,p,q)
        if root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left,p,q)
        else:
            return root



235. Lowest Common Ancestor of a Binary Search Tree
Easy

1234

88

Favorite

Share
Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

Given binary search tree:  root = [6,2,8,0,4,7,9,null,null,3,5]





https://leetcode.com/problems/maximum-average-subtree



Find Pair With Max Appeal Sum
3
Anonymous User
Anonymous User
Last Edit: August 12, 2019 5:21 AM

740 VIEWS

Find pair with maximum Appeal value.

Input: Array
Output: index {i, j} ( i = j allowed) with maximum Appeal
Appeal = A[i] +A[j] + abs(i-j)


Date problem
0
Anonymous User
Anonymous User
Last Edit: July 27, 2019 7:17 AM

1.4K VIEWS

Question -> "Given to dates, determine whether they were a month apart"

Answer -> I was taken back by how simple the question was. Since the dates given were date objects (I used javascript), you can subtract and convert from milliseconds to days. Slightly tricky part is determining how many days ahead from earliest date is considered a month. I said it should be number of days in earliest month.

Example ->
function(2/15/2018, 2/18/2018) returns true
function(3/4/2019, 4/5/2018) returns false


Number of islands
1
DarkHorse314's avatar
DarkHorse314
1
Last Edit: July 30, 2019 5:13 PM

553 VIEWS

Exact copy:
https://leetcode.com/problems/number-of-islands/

200. Number of Islands
Medium

3055

109

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

class Solution(object):
    def numIslands(self, grid):
        def sink(i, j):
            if 0 <= i < len(grid) and 0 <= j < len(grid[i]) and grid[i][j] == '1':
                grid[i][j] = '0'
                list(map(sink, (i+1, i-1, i, i), (j, j, j+1, j-1)))  # map in python3 return iterator
                return 1
            return 0
        return sum(sink(i, j) for i in range(len(grid)) for j in range(len(grid[i])))
        
        
        
 Determinant of a Matrix
What is Determinant of a Matrix?
Determinant of a Matrix is a special number that is defined only for square matrices (matrices which have same number of rows and columns). Determinant is used at many places in calculus and other matrix related algebra, it actually represents the matrix in term of a real number which can be used in solving system of linear equation and finding the inverse of a matrix.

How to calculate?
The value of determinant of a matrix can be calculated by following procedure –
For each element of first row or first column get cofactor of those elements and then multiply the element with the determinant of the corresponding cofactor, and finally add them with alternate signs. As a base case the value of determinant of a 1*1 matrix is the single value itself.
Cofactor of an element, is a matrix which we can get by removing row and column of that element from that matrix.

Determinant of 2 x 2 Matrix:

 A = \begin{bmatrix} a & b\\  c & d \end{bmatrix}  \begin{vmatrix} A \end{vmatrix}= ad - bc 

22

Determinant of 3 x 3 Matrix:
 A = \begin{bmatrix} a & b & c\\  d & e & f\\  g & h & i \end{bmatrix}  \begin{vmatrix} A \end{vmatrix}= a(ei-fh)-b(di-gf)+c(dh-eg) 



Recommended: Please s


My approach to this problem: https://leetcode.com/playground/EFw7cewr
Time Complexity: O(N)
Space Complexity: O(1)

Algo:

Find the most valuable (value at index i minus distance of i from index 0) index from left to right.
Find the most valuable (value at index i minus distance of i from last index) index from right to left.
Pair them up and return.


def max_appeal_value(arr):

appeal = 0
res = None
best_value, best_index = arr[0], 0
for i, num in enumerate(arr):

	if i - best_index < num - best_value:
		best_value, best_index = num, i
	
	new_appeal = num + arr[best_index] + abs(best_index -i)
	if new_appeal > appeal:
		appeal = new_appeal
		res = [best_index, i]
return res

Python code:
def maxAppealSum (num):
res = 0
result = [0,0]
for i in range(len(num)):
for j in range(len(num)):
if (num[i] + num[j] + abs(i-j)) > res:
res = num[i] + num[j] + abs(i-j)
result[0] = i
result[1] = j


71. Simplify Path
Medium

485

1305

Favorite

Share
Given an absolute path for a file (Unix-style), simplify it. Or in other words, convert it to the canonical path.

In a UNIX-style file system, a period . refers to the current directory. Furthermore, a double period .. moves the directory up a level. For more information, see: Absolute path vs relative path in Linux/Unix

Note that the returned canonical path must always begin with a slash /, and there must be only a single slash / between two directory names. The last directory name (if it exists) must not end with a trailing /. Also, the canonical path must be the shortest string representing the absolute path.

 

Example 1:

Input: "/home/"
Output: "/home"
Explanation: Note that there is no trailing slash after the last directory name.
Example 2:

Input: "/../"
Output: "/"
Explanation: Going one level up from the root directory is a no-op, as the root level is the highest level you can go.
Example 3:

Input: "/home//foo/"
Output: "/home/foo"
Explanation: In the canonical path, multiple consecutive slashes are replaced by a single one.
Example 4:

Input: "/a/./b/../../c/"
Output: "/c"
Example 5:

Input: "/a/../../b/../c//.//"
Output: "/c"
Example 6:

Input: "/a//b////c/d//././/.."
Output: "/a/b/c"
	
class Solution(object):
    def simplifyPath(self, path):
        ls = path.split("/")
        i, ln = 0, len(path)
        if ln == 0:
            return ""
        st = []
        for item in ls:
            if item == "" or item == ".":
                continue
            if item == "..":
                if st:
                    st.pop(-1)
            else:
                st.append(item)


        return "/" + "/".join(st)
        
  Next challenges:
Valid Number
Student Attendance Record I
Last Substring in Lexicographical Order


Position: SDE1
1 hr- two questions: No behavioural!

https://www.geeksforgeeks.org/subtract-two-numbers-represented-as-linked-lists
https://leetcode.com/problems/sort-colors


Input: dict = [apple, boy, cat, dog, element, zack, zill]
Output:
range("a", "z"); // returns  [apple, boy, cat, dog, element, zack, zill]
range("ebc", "zas"); // returns [element, zack]

Amazon | Phone Screen | Dictionary Range Query

Example:

Input: dict = [apple, boy, cat, dog, element, zack, zill]
Output:
range("a", "z"); // returns  [apple, boy, cat, dog, element, zack, zill]
range("ebc", "zas"); // returns [element, zack]



222. Count Complete Tree Nodes
Medium

1151

149

Favorite

Share
Given a complete binary tree, count the number of nodes.

Note:

Definition of a complete binary tree from Wikipedia:
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

Example:

Input: 
    1
   / \
  2   3
 / \  /
4  5 6

Output: 6

Find first unique char in a string

import Counter from collections

def first_unique_char(text):
    counters = Counter(text)
    for ch in text:
          if counters[ch] is 1:
                return ch
                
 or 
 
 def firstUnique(string):
  for char in string:
    if string.count(char) == 1:
      return char
      
      
 Find the first word in a stream in which it is not repeated in the rest of the stream
2
tferreira's avatar
tferreira
6
March 22, 2016 2:45 PM

6.9K VIEWS

Coding exercise:

Find the first word in a stream in which it is not repeated in the rest of the stream. Please note that you are being provided a stream as a source for the characters. The stream is guaranteed to eventually terminate (i.e. return false from a call to the hasNext() method), though it could be very long. You will access this stream through the provided interface methods. A call to hasNext() will return whether the stream contains any more characters to process. A call to getNext() will return the next character to be processed in the stream. It is not possible to restart the stream.

Example:

Input: The angry dog was red. And the cat was also angry.

Output: dog

In this example, the word ‘dog’ is the first word in the stream in which it is not repeated in the stream.

Use one of the following skeletons for your solution:
	
	
	from collections import OrderedDict
def non_repeated(s):
	dict =OrderedDict()
	s = s.lower()
	for i in s.replace('.','').split(' '):
		dict[i]= dict.get(i,0)+1
	
	for i in dict:
		if dict[i] ==1:
			return i
	
s = "The angry dog was red. And the cat was red angry"
print non_repeated(s)


Amazon wants to implement a new backup system, in which files are stored into data tapes.

Amazon wants to implement a new backup system, in which files are stored into data tapes. This new system must follow the following 2 rules:

Never place more than two files on the same tape.
Files cannot be split across multiple tapes.
It's guaranteed that all tapes have the same size and that they will always be able to store the largest file.

Every time this process is executed, we already know the size of each file, and the capacity of the tapes. Having that in mind, we want to design a system that is able to count how many tapes will be required to store the backup in the most efficient way.

The parameter of your function will be a structure that will contain the file sizes and 
the capacity of the tapes. You must return the minimum amount of tapes required to store the files.



Example:


Input: Tape Size = 100; Files: 70, 10, 20

Output: 2

I use two pointers to solve this issue.

Sort the input as ascending
assign two pointers i,j to first item & last item
sum the i, j: if sum is less than capacity, then counter++; i++, j--, else j--
loop with i <= j



public static int GetMinimumTapeCount(List<int> files, int tapeCapacity)
        {
            //sort in ascending
            files.Sort(new MyComparer());

            int count = 0;
            int i = 0;
            int j = files.Count - 1;


            while (i <= j)
            {
                //if the largest file is less than half of capacity
                if (files[j] <= tapeCapacity / 2)
                {
                    count += (j-i)/ 2 + 1;
                    break;
                }

                if (files[i] + files[j] <= tapeCapacity)
                {
                    i++;
                    j--;
                }
                else
                    j--;

                count++;
            }

            return count;
        }



Amazon | Validate Complete Binary Tree


958. Check Completeness of a Binary Tree
Medium

313

7

Favorite

Share
Given a binary tree, determine if it is a complete binary tree.

Definition of a complete binary tree from Wikipedia:
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

class Solution(object):
    def isCompleteTree(self, root):
        q=collections.deque()
        q.append(root)
        while q and q[0]:
            el=q.popleft()
            q.append(el.left)
            q.append(el.right)
        return not any(q)

Next challenges:
Minimum Absolute Difference in BST
Kill Process
Maximum Binary Tree II

Design a data structure that supports insert, delete, search and getRandom in constant time.


class Solution():
    def __init__(self):
        self.data = []
        self.hash = {}
    def insert(self, a):
        if self.search(a):
            self.data.append(a)
            self.hash[a] = len(self.data)-1
    def search(self, a):
        return self.hash.has_key(a)
    def remove(self, a):
        if self.search(a):
            del self.data[self.hash[a]]
            self.hash.pop(a,None)
    def getRandom(self):
        import random
        if len(self.data) > 0:
            return self.data[random.randint(0,len(self.data)-1)]
        return
        
    Or
        
  class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.elements = dict()
        

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.elements:
            return False
        else:
            self.elements[val] = 0
            return True
        

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.elements:
            self.elements.pop(val)
            return True
        else:
            return False
        

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        from random import randint
        
        return self.elements.keys()[randint(0, len(self.elements) - 1)]

Amazon | System Design | A scalable chat application on phone browsing

https://leetcode.com/discuss/interview-question/124613/Amazon-or-System-Design-or-A-scalable-chat-application-on-phone-browsing


Scope / Clarifications:

Registration with Parent & Child
Add / Remove Parent / Child account within family
Login & Logout
Cross Country & Cross States
Unlimited history viewing but scroll up and lazy load the old messages
Unread messages
Push notification when new messages arrived
Support sending multi-media including images, video and sound clip
Assumptions:

Average length of text message is 100 characters, i.e. 1.6 KB if using UTF-16
Since sending raw images consume too much bandwidth, the mobile app will resize the images to < 2000 px in any dimension
Average size of images after resize is 200 KB
Average size of video is 3 MB
Average size of sound clip is 1 MB
Ratio of text : images : sound : video is 10 : 3 : 3 : 1
Components:

Authentication API, for login, logout & registration
Account Management API, for add / remove accounts from family
Message Exchange for real time messaging of all online devices
Storage for multi-media contents
Storage for message histories
Storage for unread messages
Design:

Multiple Authentication Servers per Country / State, load balancers are in front of each Country / State Authentication Server
When First Time Login / Register, the mobile app will contact the default Authentication Server. No matter Login / Register success or not, the response will contain the correct 
Authentication Server's URL for the device next Login / Register, based on geolocation of the devices' IP address.
There is a mapping table inside each Authentication Server to map IP range to corresponding location's 
Authentication Server. The mapping table will be updated periodically.
The device will remember the Authentication Server URL, so that it will contact to the nearest Authentication Server directly on next login.
After login success with Authentication Server, a token generated by shared secret key will be generated and the device will use this key in all the following API call / data retrieval.
Separate servers used for Account Management, Message Exchange, multi-media storage, message histories storage and unread message storage
Same token is used for calling all of the above servers. The verification is simple and scalable, since they use shared secret key, signature based authentication.
There are multiple servers for Account Management, Message Exchange, multi-media storage, message histories storage and unread message storage, the corresponding server URL are assigned to a device once login. The assignment is based on consistent hash.
The message exchange servers will state connected with online device through HTTP connection. 
In order to push message to the devices in real time.
For the message exchange servers, there is an internal gateway within the same region in order to interchange the messages to devices belongs to different exchange servers.
And there is an external gateway per region for cross region messages interchange. So that all the exchange server in the 
world don't need to interconnected with each others.
The unread message storage servers are using in-memory cache, since unread message are suppose to be retrieve frequently and should be faster, compare with that of reading message histories.
Suppose every multi-media resources will have an GUID and an server ID, when device connected storage server cannot find the corresponding resources, the storage server will pull the resource from the correct server, and then serve the content to the client device, so that when the family member within the same region try to access the same content, it will be faster.
https://www.youtube.com/watch?v=WE9c9AZe-DY



0
yunhong's avatar
yunhong
0
May 8, 2016 10:29 AM

6.7K VIEWS

Given a data stream input of positive integers a1, a2, ..., an, ..., summary the numbers seen so far as a list of disjoint intervals.

For example, suppose the input are 1, 3, 7, 2, 6, ..., then the summary will be:
[1, 1]
[1, 1], [3, 3]
[1, 1], [3, 3], [7, 7]
[1, 3], [7, 7]
[1, 3], [6, 7]

Need to do it better than linear time.


352. Data Stream as Disjoint Intervals
Hard

201

64

Favorite

Share
Given a data stream input of non-negative integers a1, a2, ..., an, ..., summarize the numbers seen so far as a list of disjoint intervals.

For example, suppose the integers from the data stream are 1, 3, 7, 2, 6, ..., then the summary will be:

[1, 1]
[1, 1], [3, 3]
[1, 1], [3, 3], [7, 7]
[1, 3], [7, 7]
[1, 3], [6, 7]

https://leetcode.com/problems/data-stream-as-disjoint-intervals/discuss/347973/Concise-python.-O(1)-add-O(nlgn)-get.-No-union-find-no-heap.

To make things faster, you can also keep two dictionaries:

start value : [start, end]
end value : [start, end]
For each newly added value, look for value+1 in start dict, look for value-1 in end dictionary, when find one, update the dictionary.
In the "get" operation, you sort the dictionary to get output intervals. But beause the dictionary size is much smaller than total number of values, it is much faster sorting.

180ms defeats 97%

class SummaryRanges(object):
    def __init__(self):
        self.intervals = []

    def addNum(self, val):
        # find location
        low, high = 0, len(self.intervals) - 1
        while low <= high:
            mid = (low + high) // 2
            elem = self.intervals[mid]
            if elem.start <= val <= elem.end:
                return
            elif elem.start > val:
                high = mid - 1
            else:
                low = mid + 1

        # insert the interval
        pos = min(low, high) + 1
        self.intervals[pos:pos] = [Interval(val, val)]

        # merge with next interval
        if pos + 1 < len(self.intervals) and val == self.intervals[pos + 1].start - 1:
            self.intervals[pos].end = self.intervals[pos + 1].end
            self.intervals[pos + 1:pos + 2] = []

        # merge with prev interval
        if pos - 1 >= 0 and val == self.intervals[pos - 1].end + 1:
            self.intervals[pos - 1].end = self.intervals[pos].end
            self.intervals[pos:pos + 1] = []

    def getIntervals(self):
        return self.intervals
        
  Next challenges:
Summary Ranges
Find Right Interval
Range Module


Convert a string to largest palindrome by changing at most k digits in the string

Amazon | Max Consecutive Ones
1004. Max Consecutive Ones III
Medium

316

6

Favorite

Share
Given an array A of 0s and 1s, we may change up to K values from 0 to 1.

Return the length of the longest (contiguous) subarray that contains only 1s. 

Example 1:

Input: A = [1,1,1,0,0,0,1,1,1,1,0], K = 2
Output: 6
Explanation: 
[1,1,1,0,0,1,1,1,1,1,1]
Bolded numbers were flipped from 0 to 1.  The longest subarray is underlined.

class Solution(object):
    def longestOnes(self, a, n):
    
        if not a:
            return 0

        from collections import deque
        queue = deque()
        count_zeros = 0
        max_len = 0

        i = 0
        while i < len(a):
            if a[i] == 1:
                queue.append(a[i])
                i += 1
            else:
                if count_zeros < n:
                    queue.append(a[i])
                    count_zeros += 1
                    i += 1
                else:
                    left = queue.popleft()
                    if left == 0:
                        count_zeros -= 1
            max_len = max(max_len, len(queue))

        return max_len
        
        
        
        Next challenges:
Longest Substring with At Most K Distinct Characters
Longest Repeating Character Replacement
Max Consecutive Ones
Max Consecutive Ones II

https://leetcode.com/problems/max-consecutive-ones-iii/discuss/247678/Python-3-
Solution%3A-sliding-window-for-zeros'-indexes.-Detailed-explanation-included.

class Solution:
    def longestOnes(self, A, K):
        zero_index = [i for i, v in enumerate(A) if v == 0]
        if K >= len(zero_index):
            return len(A)
        res = 0
        for i in range(0, len(zero_index) - K + 1):
            one_start = zero_index[i-1] + 1 if i > 0 else 0
            one_end = zero_index[i+K] - 1 if i+K < len(zero_index) else len(A) - 1
            res = max(res, one_end - one_start + 1)
        return res
        
  Simple Balanced Parentheses(Phone Interview)      
  def balanced_parens(s):
    if len(s) == 0:
        return True
    stack = collections.deque()
    matches = {')': '(',
               ']': '[',
               '}': '{'}
    for k in s:
        if k in ']})' and len(stack) == 0:
            return False
        elif k in '[{(':
            stack.append(k)
        else:
            v = stack.pop()
            if matches[k] != v:
                return False
    return True
    
    Print all k-sum paths in a binary tree
A binary tree and a number k are given. Print every path in the tree with sum of the nodes in the path as k.
A path can start from any node and end at any node and must be downward only, i.e. they need not be 
root node and leaf node; and negative numbers can also be there in the tree.

Input : k = 5  
        Root of below binary tree:
           1
        /     \
      3        -1
    /   \     /   \
   2     1   4     5                        
        /   / \     \                    
       1   1   2     6    
                       
Output :
3 2 
3 1 1 
1 3 1 
4 1 
1 -1 4 1 
-1 4 2 
5 
1 -1 5 


113. Path Sum II
Medium

1037

36

Favorite

Share
Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.

Note: A leaf is a node with no children.

Example:

Given the below binary tree and sum = 22,

      5
     / \
    4   8
   /   / \
  11  13  4
 /  \    / \
7    2  5   1
Return:

[
   [5,4,11,2],
   [5,8,4,5]
]

class Solution(object):
    def pathSum(self, root, sum):
        if not root: return []
        stack=[(root,[root.val],root.val)]
        res=[]
        while stack:
            root,path,s=stack.pop()
            if not root.right and not root.left:
                if s==sum:
                    res.append(path)
            if root.left:
                stack.append((root.left,path+[root.left.val],s+root.left.val))
            if root.right:
                stack.append((root.right,path+[root.right.val],s+root.right.val))
        return res
        
        
        Next challenges:
Path Sum
Binary Tree Paths
Path Sum IV

What is demand paging?

Demand paging is a system wherein area of memory that are not currently being used are swapped to disk to make room for an application’s need.

Demand paging is a feature in operating systems which allows the OS to bring in to memory a page of a process only when the process accesses it.

What happens when we turn on computer?

When the computer is switched on, it’s of no use because the data stored in the memory(RAM) is garbage and there is no Operating System running. The first thing motherboard does is to initialize its own firmware and get the CPU running. Some of the CPU registers including Instruction Pointer (EIP) have predefined values. In x86 systems the initial value of the EIP is 0xfffffff0 and the instruction stored at this memory location is executed. The instruction is JMP (JUMP) to a Read Only Memory (ROM) which contains the BIOS and its code starts executing.

Functions of BIOS
POST (Power On Self Test) to ensure that the various components present in the system are functioning properly. If video card is missing or not functioning properly, motherboard emit beeps since error cannot be displayed. Beeps are emitted according to Beep Codes of the motherboard and it varies from one motherboard to other. A comprehensive list of beep codes can be found here. If the computer passes the video card test, manufacturer logo is printed on the screen.

It initializes the various hardware devices. It is an important process so as to ensure that all the devices operate smoothly without any conflicts. BIOSes following ACPI create tables describing the devices in the computer.

It looks for an Operating System to load. Typically, the BIOS will search it in Hard Drives, CD-ROMs, floppy disks etc. The actual search order can be configured by the user by changing Boot Order in BIOS settings. If BIOS cannot find a bootable operating system it displays an error message “Non-System Disk Error”.

Generally the operating system is present in the hard disk. We confine our discussion to how operating system boots from the hard disk.

Master Boot Record
The first sector of the hard disk is called Master Boot Record (MBR). The structure of MBR is operating system independent. It is of 512 bytes and it has mainly two components. The first 446 bytes contain a special program called Bootstrap Loader. The next 64 bytes contains a partition table. A partition table stores all the information about the partitions in a hard disk and file system types (a file system describes how data will be stored and retrieved from the partition). A partition table is required to boot up the operating system. The last two bytes of MBR contains a magic number AA55. It is used to classify whether the MBR is valid or not. An invalid magic number indicates that the MBR is corrupt and machine will not be able to boot.

Bootstrap Loader
Bootstrap loader or the boot loader contains the code to load an operating system. Earlier Linux distributions used LILO (LInux Loader) bootloader. Today, most of the distributions use GRUB (GRand Unified Bootloader) which has many advantages over LILO. BIOS loads the bootstrap loader into the memory (RAM) and starts executing the code.

Boot loader of traditional operating systems like Windows 98 used to identify an active partition in the hard disk by looking at the active flag of partition table and loading its boot sector into the memory. Boot sector is the first sector of each partition in contrast to MBR which is the first sector of the hard disk. The boot sector is of 512 bytes in memory and contains code to boot an operating system in that partition. However boot loaders like GRUB and LILO are more robust and boot process is not so straight forward.

Booting an operating system with GRUB is a two stage process: stage 1 and stage 2. In some cases an intermediate stage 1.5 may also be used to load stage 2 from an appropriate file system. Stage 1 is the boot loader code itself and its task is to only call the stage 2 which contains the main code. This is done because of the tiny size of the stage 1 (512 bytes). GRUB stage 2 loads the Linux Kernel and initramfs into the memory.

Kernel is the core component of an operating system. It has complete control of all the things happening in a system. It is the first part of the operating system to load into the memory and remains there throughout the session.

To access a file system it must be first mounted. When kernel is loaded into the memory none of the file system is mounted and hence initial RAM based file system (initramfs) is required by kernel to execute programs even before the root file system is mounted. Kernel executes a init (initialization) program which has pid=1. It is a daemon process and continues to run until the computer is shut down. It also load the modules and drivers required to mount the root file system. Linux stores information about the major file systems in a file /etc/fstab

init
init is the last step of the kernel boot sequence. It looks for the file /etc/inittab to see if there is an entry for initdefault. It is used to determine initial run-level of the system. A run-level is used to decide the initial state of the operating system.
Some of the run levels are:

Level

0 –> System Halt
1 –> Single user mode
3 –> Full multiuser mode with network
5 –> Full multiuser mode with network and X display manager
6 –> Reboot
The above design of init is called SysV- pronounced as System five. Several other implementations of init have been written now. Some of the popular implementations are systemd and upstart. Upstart is being used by ubuntu since 2006. More details of the upstart can be found here.

The next step of init is to start up various daemons that support networking and other services. X server daemon is one of the most important daemon. It manages display, keyboard, and mouse. When X server daemon is started you see a Graphical Interface and a login screen is displayed.

https://www.geeksforgeeks.org/what-happens-when-we-turn-on-computer/


Find the minimum distance between two numbers
Given an unsorted array arr[] and two numbers x and y, find the minimum distance between x and y in arr[]. The array might also contain duplicates. You may assume that both x and y are different and present in arr[].

Examples:
Input: arr[] = {1, 2}, x = 1, y = 2
Output: Minimum distance between 1 and 2 is 1.

Input: arr[] = {3, 4, 5}, x = 3, y = 5
Output: Minimum distance between 3 and 5 is 2.

Input: arr[] = {3, 5, 4, 2, 6, 5, 6, 6, 5, 4, 8, 3}, x = 3, y = 6
Output: Minimum distance between 3 and 6 is 4.

Input: arr[] = {2, 5, 3, 5, 4, 4, 2, 3}, x = 3, y = 2
Output: Minimum distance between 3 and 2 is 1.

 Python3 code to Find the minimum 
# distance between two numbers 
  
def minDist(arr, n, x, y): 
    min_dist = 99999999
    for i in range(n): 
        for j in range(i + 1, n): 
            if (x == arr[i] and y == arr[j] or
            y == arr[i] and x == arr[j]) and min_dist > abs(i-j): 
                min_dist = abs(i-j) 
        return min_dist 
  
  
# Driver code 
arr = [3, 5, 4, 2, 6, 5, 6, 6, 5, 4, 8, 3] 
n = len(arr) 
x = 3
y = 6
print("Minimum distance between ",x," and ", 
     y,"is",minDist(arr, n, x, y)) 
  
# This code is contributed by "Abhishek Sharma 44" 

Time Complexity: O(n^2)


import sys 
  
def minDist(arr, n, x, y): 
    min_dist = sys.maxsize 
  
    #Find the first occurence of any of the two numbers (x or y) 
    # and store the index of this occurence in prev 
    for i in range(n): 
          
        if arr[i] == x or arr[i] == y: 
            prev = i 
            break
   
    # Traverse after the first occurence 
    while i < n: 
        if arr[i] == x or arr[i] == y: 
  
            # If the current element matches with any of the two then 
            # check if current element and prev element are different 
            # Also check if this value is smaller than minimm distance so far 
            if arr[prev] != arr[i] and (i - prev) < min_dist : 
                min_dist = i - prev 
                prev = i 
            else: 
                prev = i 
        i += 1        
   
    return min_dist 
   
# Driver program to test above fnction */ 
arr = [3, 5, 4, 2, 6, 3, 0, 0, 5, 4, 8, 3] 
n = len(arr) 
x = 3
y = 6
print ("Minimum distance between %d and %d is %d\n"%( x, y,minDist(arr, n, x, y))); 
  
# This code is contributed by Shreyanshi Arun. 

Time Complexity: O(n)

92
prashant3's avatar
prashant3
126
Last Edit: October 22, 2018 1:53 PM

66.1K VIEWS

Ways to approach a general Design problem.

Use Cases Generation: Gather all the possible use cases

Constraints and Analysis: How many users, how much data etc.

Basic Design: Most basic design. Few users case.

Bottlenecks: Find the bottlenecks and solve them.

Scalability: A large number of users. 4 and 5 step will go in loop till we get a satisfactory answer

Current Scenario

Use cases for this problem.
Parking can be single-level or multilevel.
Types of vehicles that can be parked, separate spaces for each type of vehicle.
Entry and exit points.
Constraints
Number of vehicles that can be accommodated of any type.
Basic Design/High-Level Components
Vehicle/Type of vehicle.
Entry and Exit points.
Different spots for vehicles.
Bottlenecks
Capacity breach for any type of vehicle.
Scalability
Scalable from single-level to multi-level
Scalable from Bike only parking to accommodate all kinds of vehicles.
Keeping these in minds, APIs can be designed in the language of your preference.


Binary Tree largest Sum


Binary Tree Maximum Path Sum
Hard

1916

143

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

124. Binary Tree Maximum Path Sum
Hard

1916

143

Favorite

Share
Given a non-empty binary tree, find the maximum path sum.

For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.

Example 1:



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



https://medium.com/jay-tillu/what-is-gradle-why-google-choose-it-as-official-build-tool-for-android-adafbff4034


Help me solving this String based problem from amazon interview

Given a paragraph of text, write a program to find the first shortest sub-segment that contains each of the given k words at least once. A segment is said to be shorter than other if it contains less number of words.

Ignore characters other than [a-z][A-Z] in the text. Comparison between the strings should be case-insensitive.

If no sub-segment is found then the program should output “NO SUBSEGMENT FOUND”.

Input format :

First line of the input contains the text.
Next line contains k , the number of words given to be searched.
Each of the next k lines contains a word.

Output format :

Print first shortest sub-segment that contains given k words , ignore special characters, numbers.If no sub-segment is found it should return “NO SUBSEGMENT FOUND”

Sample Input :

This is a test. This is a programming test. This is a programming test in any language.
4
this
a
test
programming

Sample Output :

a programming test This

Explanation :
In this test case segment "a programming test. This" contains given four words. You have to print without special characters, numbers so output is "a programming test This". Another segment "This is a programming test." also contains given four words but have more number of words.

Constraint :

Total number of character in a paragraph will not be more than 200,000.
0 < k <= no. of words in paragraph.
0 < Each word length < 15

def scan(para, keys):
    words = para.lower().replace('.', '').split()
    keys_pos = {x: [] for x in keys}
    print(keys_pos)
    win_begin = 0
    win_end = 0
    min_seg = (-1, len(words))

    for i, word in enumerate(words):
        if word not in keys_pos:
            continue
        win_end = i

        pos = keys_pos[word]
        pos.append(i)
        if len(pos) > 1 and pos[0] == win_begin:
            del pos[0]
            win_begin = min(x[0] for x in keys_pos.values() if len(x) > 0)

        if all(len(x) > 0 for x in keys_pos.values()):
            valid_seg = (win_begin, win_end)
            if valid_seg[1] - valid_seg[0] < min_seg[1] - min_seg[0]:
                min_seg = valid_seg

    if min_seg[0] == -1:
        return None
    return ' '.join(words[min_seg[0]:min_seg[1]+1])

if __name__ == '__main__':
    para = "This is a test. This is a programming test. This is a programming test in any language."
    keys = ["this", "a", "test", "programming"]
    print(scan(para, keys))


Amazon | Number of distinct subsequences

Jamie is walking along a number line that starts at point 0 and ends at point n. She can move either one step to the left or one step to the right of her current location , with the exception that she cannot move left from point 0 or right from point n. In other words, if Jamie is standing at point i,she can move to either i-1 or i+1 as long as her destination exists in the inclusive range [0,n]. She has a string ,s , of movement instruction consisting of the letters 1 and r , where 1 is an instruction to move one step left and r is an instruction to move one step right.
Jamie followed the instructions in s one by one and in order .For Example if s=‘rrlr’,she performs the following sequence of moves :one step right ->one step right ->one step left -> one step right .Jamie wants to move from point x to point y following some subsequence of string s instruction and wonders how many distinct possible subsequence of string s will get her from point x to point y. recall that a subsequence of a string is obtained by deleting zero or more characters from string .

it has four parameters
A String , s giving a sequence of eduction using the characters l( i.e. move left one unit ) and r (i.e. move right one unit)
An integer n, denoting the length of the number line.
An integer x, denoting jamie’s starting point on the number line
An integer y , denoting Jamie’s enidng point on the number line.
The function must return an integer denoting the total number of distinct subsequence of string s that will lead Jamie from point x to point y as this value cab be quite large .

Sample Input
rrlrlr
6
1
2

out put =7

https://leetcode.com/discuss/interview-question/124943/Amazon-or-Number-of-distinct-subsequences

class Solution():
    def count_unique_subsequence(self, s, start, dest, n):
        
        def solve_recur(s, index, curr, dest, n, path, result):
            print("result", result,index,curr)
            if curr == dest:
                result.add(path)

            if index == len(s):
                return

            solve_recur(s, index + 1, curr, dest, n, path, result)
            if s[index] == 'l':
                if curr > 0:
                    solve_recur(s, index + 1, curr - 1, dest, n, path + 'l', result)
            else:
                if curr + 1 < n:
                    solve_recur(s, index + 1, curr + 1, dest, n, path + 'r', result)
        
        
        result = set([])
        solve_recur(s, 0, start, dest, n, '', result)
        return len(result)

s = Solution()
res=s.count_unique_subsequence('rrlrlr', 1, 2, 6)
print(res)


Maximize the number of TV shows that can be watched

Given a TV show guide which is a list of start and end times for each TV show being aired on a given day, return the 
list which has the maximize shows that can be watched without conflicts.

programming = [(8, 11), (4, 7), (1, 4), (3, 5), (5, 9), (0, 6), (3, 8), (6, 10)]
https://leetcode.com/discuss/interview-question/124748/Maximize-the-number-of-TV-shows-that-can-be-watched
def schedule_greedy(prog):
    def not_overlap(entry1, entry2):
        if entry1[0] >= entry2[1]:
            return True
        elif entry1[1] <= entry2[0]:
            return True
        else:
            return False

    sorted_prog = sorted(prog, key=lambda x: x[1]) # sort by end time
    result = []
    while sorted_prog:
        smallest_entry = sorted_prog.pop(0)
        result.append(smallest_entry)
        sorted_prog = [entry for entry in sorted_prog if not_overlap(smallest_entry, entry)]
    return result

print(schedule_greedy(programming))
# output
# [(1, 4), (4, 7), (8, 11)]


Michelle has created a word game for her students. The word game begins with Michelle writing a string and a number, K, on the board.
The students must find a substring of size K such that there is exactly one character that is repeated one;
in other words, there should be k - 1 distinct characters in the substring.

Write an algorithm to help the students find the correct answer. If no such substring can be found, return an empty list;
if multiple such substrings exist, return all them, without repetitions. The order in which the substrings are does not matter.

Input:
The input to the function/method consists of two arguments - inputString, representing the string written by the teacher;
num an integer representing the number, K, written by the teacher on the board.

Output:
Return a list of all substrings of inputString with K characters, that have k-1 distinct character i.e.
exactly one character is repeated, or an empty list if no such substring exist in inputString.
The order in which the substrings are returned does not matter.

Constraints:
The input integer can only be greater than or equal to 0 and less than or equal to 26 (0 <= num <= 26)
The input string consists of only lowercase alphabetic characters.

Example
Input:
inputString = awaglk
num = 4

Output:
[awag]

Explanation:
The substrings are {awag, wagl, aglk}
The answer is awag as it has 3 distinct characters in a string of size 4, and only one character is repeated twice



763. Partition Labels
Medium

1191

62

Favorite

Share
A string S of lowercase letters is given. We want to partition this string into as many parts as possible so that each letter appears in at most one part, and return a list of integers representing the size of these parts.

Example 1:
Input: S = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits S into less parts.

class Solution(object):
    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        
        counter = collections.Counter(S)
        #print(counter[0])
        seen, mark, start = set(), 0, 0
        res = []

        for i, c in enumerate(S):
            #print (i,c)
            if c not in seen:
                seen.add(c)
                print(counter[c])
                mark += counter[c]
            mark -= 1
            if mark == 0:
                res.append(i - start + 1)
                start = i + 1

        return res
        
  Amazon | Phone screen | Words intersection
  
  q = ['apple', 'peach', 'pear', 'watermelon', 'strawberry']
w = ['pineapple', 'peach', 'watermelon', 'kiwi']
t = list(set(q) & set(w))



Find Max Bandwidth
0
rst's avatar
rst
0
February 9, 2018 11:23 PM

2.1K VIEWS

For n tv channels, given show start time, end time & bandwidth needed for each channels, 
find the maximum bandwidth required at peak. a show represented as [1,30,2] meaning [show-start-time, show-end-time, bandwidth-needed].

e.g. n =3 channels, 
[[1,30, 2],[31,60, 4],[61,120, 3],
[1,20,2],[21,40,4],[41,60,5],[61,120,3],
[1,60,4],[61,120,4]]

Ans: 13, for time slot between 41-60 each channel need 4,5,4 bandwidth respectively. 13 is highest (peek/max) bandwidth.

Note
Min-size-of-show = 2 (min)
Max-duration-for-show = 720 (min) same as 24hours
Max-bandwidth-per-show = 100 (mbps)
n
Some channels can decide not to broadcast any show for given time-slot, 
which mean there will be 0 bandwidth required for that channel for given time-slot.

Same as other people have suggested. O(n) extra space.

step 1) First we need to create a timeline in the system. 
The timeline will be composed of two type of events - positive bandwidth (at start of interval) and negative bandwidth (at end of interval).
 Each point on the time line will look like (time, bandwitdh). For instance, you can break (1, 10, 2) as two intervals - (1, 2) & (10, -2)

step 2) sort the points based on the time component.

step3) keep the count of the sum as you iterate through points. The max sum will give you the max bandwidth requirement.

def get_max_bandwidth(bandwidths):
    points = []
    for start, end, band in bandwidths:
        points.append((start, band))
        points.append((end, -1 * band))
    
    points.sort(key=lambda x: x[0])
   
    max_value = 0
    curr_value = 0
    for _, band in points:
        curr_value += band
        max_value = max(curr_value, max_value)
    
    return max_value
        
        
        
        222. Count Complete Tree Nodes
Medium

1169

149

Favorite

Share
Given a complete binary tree, count the number of nodes.

Note:

Definition of a complete binary tree from Wikipedia:
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

Example:

Input: 
    1
   / \
  2   3
 / \  /
4  5 6

Output: 6

def countNodes(self, root):
    if not root:
        return 0
    h1, h2 = self.height(root.left), self.height(root.right)
    if h1 > h2: # right child is full 
        return self.countNodes(root.left) +  2 ** h2 
    else: # left child is full 
        return 2 ** h1 + self.countNodes(root.right)

# the height of the left-most leaf node
def height1(self, root):
    h = 0
    while root:
        h += 1
        root = root.left
    return h
    
def height(self, root):
    if not root:
        return 0
    return self.height(root.left) + 1
    
    Next challenges:
Closest Binary Search Tree Value



221. Maximal Square
Medium

1521

35

Favorite

Share
Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

206. Reverse Linked List
Easy

2711

69

Favorite

Share
Reverse a singly linked list.

Example:

Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL

class Solution(object):
    def reverseList(self, head):
        
        if head is None:
                return None
        previousHead = None
        nextHead = head.next
        while not nextHead is None:
            head.next = previousHead
            previousHead = head
            head = nextHead
            nextHead = head.next
        head.next = previousHead
        return head
        
#class Solution(object):
    #def reverseList(self, head):
        #rev = None
        #while head: 
            #head.next, rev, head = rev, head, head.next
        #return rev
    
  141. Linked List Cycle
Easy

1755

218

Favorite

Share
Given a linked list, determine if it has a cycle in it.

To represent a cycle in the given linked list, we use an integer pos which represents the position (0-indexed) in the linked list where tail connects to. If pos is -1, then there is no cycle in the linked list.

 

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where tail connects to the second node.
  
    
    class Solution(object):
    def hasCycle(self, head):
        marker1 = head
        marker2 = head
        while marker2!=None and marker2.next!=None:
            marker1 = marker1.next
            marker2 = marker2.next.next
            if marker2==marker1:
                return True
        return False

Example:

Input: 

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

Output: 4

class Solution(object):
    # O(m*n) space, one pass  
    def maximalSquare(self, matrix):
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
        
        
    
    
    A threaded binary tree defined as follows:

"A binary tree is threaded by making all right child pointers that would normally be null point to the in-order successor of the node (if it exists), and all left child pointers that would normally be null point to the in-order predecessor of the node."[1]

This definition assumes the traversal order is the same as in-order traversal of the tree. 
However, pointers can instead (or in addition) be added to tree nodes, rather than replacing linked 
lists thus defined are also commonly called "threads", and can be used to enable traversal in any order(s) desired.
 For example, a tree whose nodes represent information about people might be sorted by name, but have extra threads allowing quick 
traversal in order of birth date, weight, or any other known characteristic. 


# Iterative function for inorder tree traversal 
def MorrisTraversal(root): 
      
    # Set current to root of binary tree 
    current = root  
      
    while(current is not None): 
          
        if current.left is None: 
            print current.data, 
            current = current.right 
        else: 
            # Find the inorder predecessor of current 
            pre = current.left 
            while(pre.right is not None and pre.right != current): 
                pre = pre.right 
   
            # Make current as right child of its inorder predecessor 
            if(pre.right is None): 
                pre.right = current 
                current = current.left 
                  
            # Revert the changes made in if part to restore the  
            # original tree i.e., fix the right child of predecessor 
            else: 
                pre.right = None
                print current.data, 
                current = current.right 
              
# Driver program to test the above funct


Possible Words using given characters in Python
Given a dictionary and a character array, print all valid words that are possible using characters from the array.
Note: Repetitions of characters is not allowed.
Examples:

Input : Dict = ["go","bat","me","eat","goal","boy", "run"]
        arr = ['e','o','b', 'a','m','g', 'l']
Output : go, me, goal. 

This problem has existing solution please refer Print all valid words that are possible using Characters of Array link. We will this problem in python very quickly using Dictionary Data Structure. Approach is very simple :

Traverse list of given strings one by one and convert them into dictionary using Counter(input) method of collections module.
Check if all keys of any string lies within given set of characters that means this word is possible to create.
filter_none
edit
play_arrow

brightness_4
# Function to print words which can be created 
# using given set of characters 
  
  
  
def charCount(word): 
    dict = {} 
    for i in word: 
        dict[i] = dict.get(i, 0) + 1
    return dict
  
  
def possible_words(lwords, charSet): 
    for word in lwords: 
        flag = 1
        chars = charCount(word) 
        for key in chars: 
            if key not in charSet: 
                flag = 0
            else: 
                if charSet.count(key) != chars[key]: 
                    flag = 0
        if flag == 1: 
            print(word) 
  
if __name__ == "__main__": 
    input = ['goo', 'bat', 'me', 'eat', 'goal', 'boy', 'run'] 
    charSet = ['e', 'o', 'b', 'a', 'm', 'g', 'l'] 
    possible_words(input, charSet) 
    
    
    
    
    
    
    Amazon | OOD | Design a chess gam

We need to clarify a bunch of things first, for example:

Do we need a graphical environment? it is totally possible to code the game with no graphics at all
Do we need an artificial intelligence machine or is it just going to be human to human?
Do we want to support network gaming (ey,why not?)
Once all this has been clarified, we can dive into the problem.

This question can be modelled in great detail using object orientation, here is my approach, basic objects required are

Game: Stores the whole game
Movement: Stores each of the players movements
Piece abstract class represents each of the pieces
Coordinate each of the 64 squares in the chess game
Knight, Tower, Pawn... all the pieces are implementations of the abstract class Piece


I will try to define it as simply as possible.

The board is an array of arrays, each array having length 8.

The board's state is defined by the locations of the pieces, the initial setup is fairly straightforward (a pawn in each of the 1st row array for white, the other white pieces in their order in the 0th row, the opposite for black, ie the pawns in 6th row and other pieces in 7th row).

You must keep track of whose turn it is, it switches between black and white.

There are a set of valid moves for each piece during a turn, these are defined by where the piece can go within the array. Pieces my land on pieces of the other color but not their own color, pieces they may not move through other pieces (except knights and kings when castling).

Each piece has it's own rules, for example a pawn's possible moves are of the form

[current_x][current_y] => [current_x][current_y+1] (only if there is no piece of same color at [current_x][current_y+1])
or [current_x][current_y] => [current_x][current_y+2] if current position == starting position
or [current_x][current_y] => [current_x+1][current_y+1] (if opposing piece occupies that location)
or [current_x][current_y] => [current_x+1][current_y-1] (if opposing piece occupies that location)

if a piece ends its move on the piece of another color, that other piece is removed from the game.

Every turn you must check to see if the other color's king is in check (meaning that one of your pieces can move to its location next turn... and any move is invalid if it causes your own king to be in check). If this is the case, the other player 
is notified. If all possible moves by the other player still lead to their king being in check, then the first player is declare the victor. If a player is not in check, yet all moves would put them in check the game is a stale mate. Likewise 
if the game has been a loop of a set of moves the game is a stale mate.

Find Minimum Continuous Subsequence: targetList & availabletTagsList are two lists of string.

Input:
targetList = {"cat", "dog"};
availableTagsList = { "cat", "test", "dog", "get", "spain", "south" };
Output: [0, 2] //'cat' in position 0; 'dog' in position 2

Input:
targetList = {"east", "in", "south"};
availableTagsList = { "east", "test", "east", "in", "east", "get", "spain", "south" };
Output: [2, 6] //'east' in position 2; 'in' in position 3; 'south' in position 6 (east in position 4 is not outputted as it is coming after 'in')

Input:
targetList = {"east", "in", "south"};
availableTagsList = { "east", "test", "south" };
Output: [0] //'in' not present in availableTagsList


Amazon | Phone screen | How to handle large log data?
break the problem down to 3 main parts:

Collection
Transport
Storage
Discuss and nail down the specs:
Discussion (with white board drawing) could be something like:
Many things to clarify that affects the system design:

Scale requirement is in terabytes/per day but:

do we have a estimate on the peak load? Peak load defines the boundary for our system, we need to take them in to account.
Availability requirements - what would be the acceptable uptime?

Security - should we be concerned about security?

at collection side
at transport
storage
Reliability requirements

Collection failures
Transport failures
Storage failures
Any acceptable data loss?
Redundancy / cost trade off


Hashmap put and get operation time complexity is O(1) with assumption that 
key-value pairs are well distributed across the buckets. It means hashcode implemented is good. 
In above Letter Box example, If say hashcode() method is poorly implemented and returns hashcode 'E' always, In this case

In general, time complexity is O(h) where h is height of BST. Insertion: For inserting element 0, 
it must be inserted as left child of 1. Therefore, we need to traverse all elements (in order 3, 2, 1) 
to insert 0 which has worst case complexity of O(n). 
In general, time complexity is O(h).

BASIS FOR COMPARISON	QUICK SORT	MERGE SORT
The partition of elements in the array

The splitting of a array of elements is in any ratio, not necessarily divided into half.	The splitting of a array of elements is in any ratio, not necessarily divided into half.
Worst case complexity

O(n2)	O(nlogn)
Works well on

It works well on smaller array	It operates fine on any size of array
Speed of execution

It work faster than other sorting algorithms for small data set like Selection sort etc	It has a consistent speed on any size of data
Additional storage space requirement

Less(In-place)	More(not In-place)
Efficiency

Inefficient for larger arrays	More efficient
Sorting method

Internal	External
Stability

Not Stable	Stable
Preferred for

for Arrays	for Linked Lists
Locality of reference

good


def urlify(in_string, in_string_length):
    return ''.join('%20' if c == ' ' else c for c in in_string[:in_string_length])


print(urlify("Mr  John Smith",20))



Given a complete (virtual) binary tree, return true/false if the given target node exists in the tree or not. Here, the virtual means the tree nodes are numbered assuming the tree is a complete binary tree.

Example:

                1
		    /        \ 
		 2              3
       /   \           /  \ 
     4   (5)nil      6  (7)nil
   
doesNodeExist(root, 4); // true
doesNodeExist(root, 7); // false, given the node on #7 is a nil node


 I own a parking garage that provides valet parking service.
 * When a customer pulls up to the entrance they are either rejected
 * because the garage is full, or they are given a ticket they can
 * use to collect their car, and the car is parked for them.
 *
 * Given a set of different parking bays (Small, Medium, Large),
 * write a control program to accept/reject cars (also small, medium or large)
 * as they arrive, and issue/redeem tickets.
 *
 * Garage layout is 1 small bay, 1 medium bay, and 2 large bays: [1,1,2]
 *
 * First sequence Actions:
 * [(arrival, small),
 * (arrival, large),
 * (arrival, medium),
 * (arrival, large),
 * (arrival, medium)]
 *
 * Expected output: [1, 2, 3, 4, reject]
 *
 * Second sequence Actions:
 * [(arrival, small),
 * (arrival, large),
 * (arrival, medium),
 * (arrival, large),
 * (depart, 3),
 * (arrival, medium)]
 *
 * Expected output: [1, 2, 3, 4, 5]
 
 
 ParkingLot is a class.

ParkingSpace is a class.

ParkingSpace has an Entrance.

Entrance has a location or more specifically, distance from Entrance.

ParkingLotSign is a class.

ParkingLot has a ParkingLotSign.

ParkingLot has a finite number of ParkingSpaces.

HandicappedParkingSpace is a subclass of ParkingSpace.

RegularParkingSpace is a subclass of ParkingSpace.

CompactParkingSpace is a subclass of ParkingSpace.

ParkingLot keeps array of ParkingSpaces, and a separate array of vacant ParkingSpaces in order of distance from its Entrance.

ParkingLotSign can be told to display "full", or "empty", or "blank/normal/partially occupied" by calling .Full(), .Empty() or .Normal()

Parker is a class.

Parker can Park().

Parker can Unpark().

Valet is a subclass of Parker that can call ParkingLot.FindVacantSpaceNearestEntrance(), which returns a ParkingSpace.

Parker has a ParkingSpace.

Parker can call ParkingSpace.Take() and ParkingSpace.Vacate().

Parker calls Entrance.Entering() and Entrance.Exiting() and ParkingSpace notifies ParkingLot when it is taken or vacated so that ParkingLot can determine if it is full or not. If it is newly full or newly empty or newly not full or empty, it should change the ParkingLotSign.Full() or ParkingLotSign.Empty() or ParkingLotSign.Normal().

HandicappedParker could be a subclass of Parker and CompactParker a subclass of Parker and RegularParker a subclass of Parker. (might be overkill, actually.)

In this solution, it is possible that Parker should be renamed to be Car.
 */
 
 
 https://www.educative.io/courses/grokking-the-object-oriented-design-interview/gxM3gRxmr8Z
 
 
 
 A parking lot or car park is a dedicated cleared area that is intended for parking vehicles. In most countries where cars are a major mode of transportation, parking lots are a feature of every city and suburban area. Shopping malls, sports stadiums, megachurches, and similar venues often feature parking lots over large areas.


A Parking Lot
System Requirements
We will focus on the following set of requirements while designing the parking lot:

The parking lot should have multiple floors where customers can park their cars.

The parking lot should have multiple entry and exit points.

Customers can collect a parking ticket from the entry points and can pay the parking fee at the exit points on their way out.

Customers can pay the tickets at the automated exit panel or to the parking attendant.

Customers can pay via both cash and credit cards.

Customers should also be able to pay the parking fee at the customer’s info portal on each floor. If the customer has paid at the info portal, they don’t have to pay at the exit.

The system should not allow more vehicles than the maximum capacity of the parking lot. If the parking is full, the system should be able to show a message at the entrance panel and on the parking display board on the ground floor.

Each parking floor will have many parking spots. The system should support multiple types of parking spots such as Compact, Large, Handicapped, Motorcycle, etc.

The Parking lot should have some parking spots specified for electric cars. These spots should have an electric panel through which customers can pay and charge their vehicles.

The system should support parking for different types of vehicles like car, truck, van, motorcycle, etc.

Each parking floor should have a display board showing any free parking spot for each spot type.

The system should support a per-hour parking fee model. For example, customers have to pay $4 for the first hour, $3.5 for the second and third hours, and $2.5 for all the remaining hours.

Use case diagram
Here are the main Actors in our system:

Admin: Mainly responsible for adding and modifying parking floors, parking spots, entrance, and exit panels, adding/removing parking attendants, etc.

Customer: All customers can get a parking ticket and pay for it.

Parking attendant: Parking attendants can do all the activities on the customer’s behalf, and can take cash for ticket payment.

System: To display messages on different info panels, as well as assigning and removing a vehicle from a parking spot.

Here are the top use cases for Parking Lot:

Add/Remove/Edit parking floor: To add, remove or modify a parking floor from the system. Each floor can have its own display board to show free parking spots.
Add/Remove/Edit parking spot: To add, remove or modify a parking spot on a parking floor.
Add/Remove a parking attendant: To add or remove a parking attendant from the system.
Take ticket: To provide customers with a new parking ticket when entering the parking lot.
Scan ticket: To scan a ticket to find out the total charge.
Credit card payment: To pay the ticket fee with credit card.
Cash payment: To pay the parking ticket through cash.
Add/Modify parking rate: To allow admin to add or modify the hourly parking rate.




   
# Python program to count  
# full nodes in a Binary Tree 
# using iterative approach 
  
# A node structure 
class Node: 
    # A utility function to create a new node 
    def __init__(self ,key): 
        self.data = key 
        self.left = None
        self.right = None
  
# Iterative Method to count full nodes of binary tree 
def getfullCount(root): 
    # Base Case 
    if root is None: 
        return 0
      
    # Create an empty queue for level order traversal 
    queue = [] 
  
    # Enqueue Root and initialize count 
    queue.append(root) 
          
    count = 0 #initialize count for full nodes 
    while(len(queue) > 0): 
        node = queue.pop(0) 
  
        # if it is full node then increment count 
        if node.left is not None and node.right is not None: 
            count = count+1
  
        # Enqueue left child 
        if node.left is not None: 
            queue.append(node.left) 
  
        # Enqueue right child 
        if node.right is not None: 
            queue.append(node.right) 
              
    return count 
  
# Driver Program to test above function 
root = Node(2) 
root.left = Node(7) 
root.right = Node(5) 
root.left.right = Node(6) 
root.left.right.left = Node(1) 
root.left.right.right = Node(11) 
root.right.right = Node(9) 
root.right.right.left = Node(4) 
  
  
print "%d" %(getfullCount(root)) 

Output:
 2
Time Complexity: O(n)
Auxiliary Space : O(n) where, n is number of nodes in given binary tree

Method: Recursive

The idea is to traverse the tree in postorder. If the current node is full, we increment result by 1 and add returned values of left and right subtrees.
C++JavaPython3C#
filter_none
edit
play_arrow
brightness_4
# Python program to count full  
# nodes in a Binary Tree 
class newNode():  
  
    def __init__(self, data):  
        self.data = data 
        self.left = None
        self.right = None
          
          
# Function to get the count of   
# full Nodes in a binary tree  
def getfullCount(root): 
  
    if (root == None): 
        return 0
      
    res = 0
    if (root.left and root.right): 
        res += 1
      
    res += (getfullCount(root.left) + 
            getfullCount(root.right))  
    return res  
  
          
# Driver code  
if __name__ == '__main__': 
    """ 2  
    / \  
    7 5  
    \ \  
    6 9  
    / \ /  
    1 11 4  
    Let us create Binary Tree as shown  
    """
      
    root = newNode(2)  
    root.left = newNode(7)  
    root.right = newNode(5)  
    root.left.right = newNode(6)  
    root.left.right.left = newNode(1)  
    root.left.right.right = newNode(11)  
    root.right.right = newNode(9)  
    root.right.right.left = newNode(4)  
      
    print(getfullCount(root)) 
  
# This code is contributed by SHUBHAMSINGH10 

Output:
 2
Time Complexity: O(n)
Auxiliary Space : O(n)
where, n is number of nodes in given binary tree






Note: targetList will contain Distinct string objects.

I have an algorithm but I am struggling with the use of right data structure. Any ideas ?

amazonDiscuss.py
Displaying amazonDiscuss.py.

    
    140. Word Break II
Hard

1128

263

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
        
 Amazon | Phone Screen | Find Node in a Sorted Binary Tree
5
Anonymous User
Anonymous User
Last Edit: August 1, 2019 10:46 AM

490 VIEWS

Position: SDE2

A binary tree is level order sorted (Each level is sorted). Find a given node. Could you do better than O(n). Eg. Could you find 10 in below eg faster than O(n)

         1
       /   \
     2      3     
    / \    / \
  8    9 10   11
       
  def ifNodeExists(node, key): 
  
    if (node == None):  
        return False
  
    if (node.data == key):  
        return True
  
    """ then recur on left sutree """
    res1 = ifNodeExists(node.left, key)  
  
    """ now recur on right subtree """
    res2 = ifNodeExists(node.right, key)  
  
    return res1 or res2 


71. Simplify Path
Medium

490

1309

Favorite

Share
Given an absolute path for a file (Unix-style), simplify it. Or in other words, convert it to the canonical path.

In a UNIX-style file system, a period . refers to the current directory. Furthermore, a double period .. moves the directory up a level. For more information, see: Absolute path vs relative path in Linux/Unix

Note that the returned canonical path must always begin with a slash /, and there must be only a single slash / between two directory names. The last directory name (if it exists) must not end with a trailing /. Also, the canonical path must be the shortest string representing the absolute path.

 

Example 1:

Input: "/home/"
Output: "/home"
Explanation: Note that there is no trailing slash after the last directory name.


class Solution(object):
    def simplifyPath(self, path):
        ls = path.split("/")
        ln = len(path)
        if ln == 0:
            return ""
        st = []
        for item in ls:
            if item == "" or item == ".":
                continue
            if item == "..":
                if st:
                    st.pop(-1)
            else:
                st.append(item)


        return "/" + "/".join(st)
        
  Iterators in Python
Iterator in python is any python type that can be used with a ‘for in loop’. Python lists, tuples, dicts and sets are all examples of inbuilt iterators. These types are iterators because they implement following methods. In fact, 
any object that wants to be an iterator must implement following methods.

__iter__ method that is called on initialization of an iterator. This should return an object that has a next or __next__ (in Python 3) method.
next ( __next__ in Python 3) The iterator next method should return the next value for the iterable. When an iterator is used with a ‘for in’ loop, the for loop implicitly calls next() on the iterator object. This method should raise a StopIteration to signal the end of the iteration.

# A simple Python program to demonstrate 
# working of iterators using an example type 
# that iterates from 10 to given value 
  
# An iterable user defined type 
class Test: 
  
    # Cosntructor 
    def __init__(self, limit): 
        self.limit = limit 
  
    # Called when iteration is initialized 
    def __iter__(self): 
        self.x = 10
        return self
  
    # To move to next element. In Python 3, 
    # we should replace next with __next__ 
    def next(self): 
  
        # Store current value ofx 
        x = self.x 
  
        # Stop iteration if limit is reached 
        if x > self.limit: 
            raise StopIteration 
  
        # Else increment and return old value 
        self.x = x + 1; 
        return x 
  
# Prints numbers from 10 to 15 
for i in Test(15): 
    print(i) 
  
# Prints nothing 
for i in Test(5): 
    print(i) 
I can think of two options :
[1] Hash tables :

Store all records into array.
first hashtable is first_name + last_name -> index, where index represents index for that record in above array.
second hashtable for last_name -> list of indexes , where list represents all the records in array with same last name.
this seems space consuming solution but look up is O(1) for both searches.
[2] Usign Tries:

Store all records into array.
Store all first_name and last_name in a single trie, when word ends, have that node point to a list of indexes which indicates records corresponding to that word.
For last name search, search the word using trie and get the list of indexes.
for first_name+last_name search, get two list of indexes independently and find the intersection to retrieve unique record.
first_name+last_name search is slower O(n) compare to [1], but trie will use way less space considering lot of names are common. Also it gives us first_name search as well ( it wasn't asked in question. )
we could also break it down into two tries ( first_name+last_name, last_name ) if we want.
Any other ideas ?

Phone Dictionary
1
Anonymous User
Anonymous User
Last Edit: August 16, 2019 9:11 AM

488 VIEWS

Telephone directory application with insert and search mechanism, data in app memory.
First name(FN), last name(LN) and telephone number(TN) will be inserted.
Can be searched by either first name, last name combo, or last name.
Constraint on data is FN, LN combination will be unique.

Example:-
F1 L1 T1
F2 L1 T2
F1 L2 T3
F2 L2 T4

Ask what whould be best DS to build it, and assume that it has Ms of data.

Happy job hunting!


Boundary Traversal of binary tree

We break the problem in 3 parts:
1. Print the left boundary in top-down manner.
2. Print all leaf nodes from left to right, which can again be sub-divided into two sub-parts:
…..2.1 Print all leaf nodes of left sub-tree from left to right.
…..2.2 Print all leaf nodes of right subtree from left to right.
3. Print the right boundary in bottom-up manner.

We need to take care of one thing that nodes are not printed again. e.g. The left most node is also the leaf node of the tree.Grid
We break the problem in 3 parts:
1. Print the left boundary in top-down manner.
2. Print all leaf nodes from left to right, which can again be sub-divided into two sub-parts:
…..2.1 Print all leaf nodes of left sub-tree from left to right.
…..2.2 Print all leaf nodes of right subtree from left to right.
3. Print the right boundary in bottom-up manner.

We need to take care of one thing that nodes are not printed again. e.g. The left most node is also the leaf node of the tree.

Based on the above cases, below is the implementation:

filter_none
edit
play_arrow

brightness_4
# Python3 program for binary traversal of binary tree 
  
# A binary tree node 
class Node: 
  
    # Constructor to create a new node 
    def __init__(self, data): 
        self.data = data  
        self.left = None
        self.right = None
  
# A simple function to print leaf nodes of a Binary Tree 
def printLeaves(root): 
    if(root): 
        printLeaves(root.left) 
          
        # Print it if it is a leaf node 
        if root.left is None and root.right is None: 
            print(root.data), 
  
        printLeaves(root.right) 
  
# A function to print all left boundary nodes, except a  
# leaf node. Print the nodes in TOP DOWN manner 
def printBoundaryLeft(root): 
      
    if(root): 
        if (root.left): 
              
            # to ensure top down order, print the node 
            # before calling itself for left subtree 
            print(root.data) 
            printBoundaryLeft(root.left) 
          
        elif(root.right): 
            print (root.data) 
            printBoundaryLeft(root.right) 
          
        # do nothing if it is a leaf node, this way we 
        # avoid duplicates in output 
  
  
# A function to print all right boundary nodes, except 
# a leaf node. Print the nodes in BOTTOM UP manner 
def printBoundaryRight(root): 
      
    if(root): 
        if (root.right): 
            # to ensure bottom up order, first call for 
            # right subtree, then print this node 
            printBoundaryRight(root.right) 
            print(root.data) 
          
        elif(root.left): 
            printBoundaryRight(root.left) 
            print(root.data) 
  
        # do nothing if it is a leaf node, this way we  
        # avoid duplicates in output 
  
  
# A function to do boundary traversal of a given binary tree 
def printBoundary(root): 
    if (root): 
        print(root.data) 
          
        # Print the left boundary in top-down manner 
        printBoundaryLeft(root.left) 
  
        # Print all leaf nodes 
        printLeaves(root.left) 
        printLeaves(root.right) 
  
        # Print the right boundary in bottom-up manner 
        printBoundaryRight(root.right) 
  
  
# Driver program to test above function 
root = Node(20) 
root.left = Node(8) 
root.left.left = Node(4) 
root.left.right = Node(12) 
root.left.right.left = Node(10) 
root.left.right.right = Node(14) 
root.right = Node(22) 
root.right.right = Node(25) 
printBoundary(root) 
  
# This code is contributed by  
# Nikhil Kumar Singh(nickzuck_007) 


Design a hotel check in check out system

enum RoomType to hold room types that can be category by bed number ad types such as King, queue, two-bed

class: Room with variables: roomNumber, roomType, isSmoking, reserveStartTime, reserveStartTime,CheckingTime and checkoutTime
costructor with parameter roomType and isSmoking
set/get methods
method isAvailable(), reserver(), checkin() and checkout()

interface Hotel with member variable List rooms to hold rooms
and below method
addRoom,(Room r);
List avaibleRooms();
List avaibleRooms(RoomType type, boolean smoking)
reserveRoom(Date startTime, Date endTime);
checkin(Room room, Date checkinTime)
checkout(Room room, Date checkoutTime)

HotelImpl to implement Hotel

HotelApp to intitiate Hotel and addRoom(), then let user to reserveRoom, Checkin and checkout

Amazon | Onsite | Find if an essay is plagiarized
0
nnasiruddin's avatar
nnasiruddin
12
Last Edit: July 27, 2019 8:04 AM

920 VIEWS

Given two essays (Essay 1 and Essay 2). Write a function which will check if two or more sentences in essay 2 are copied from essay 1.


 OOD | Design a snake and ladders games
16
baishali13's avatar
baishali13
16
Last Edit: July 27, 2019 8:00 AM

7.8K VIEWS

I was asked to describe the classes that would be used.
The key idea here was to try and evaluate a candidate's abstraction skills and I was provided feedback to improve upon the same.

It's driving me crazy how lots of people create so many classes for this simple game. Why make a dozen classes when you only really need a 1 class and several simple data structures? Flat is better than nested. Simple is better than complicated. Short is better than long.

Forgive me if I'm guessing the rules of the game.

THE BOARD:
Represent each square on the board in a 1D array of length 100 (when you wanted to create a gui for this board, simply divide the board into rows of ten and map that to the gui). Having a 1D abstraction of the board will make it a LOT easier to deal with moving pieces and OOB errors.

The board will be filled with zeros initially. We'll go through and randomly change some of these squares to positive and negative numbers:
Positive numbers on the board represent the start of LADDERS.
Negative numbers on the board represents the head of SNAKES.
When you run into a square on a board, if it's not zero, you add that number to the current position of that piece at the end of the move.
We have to do some thorough checking in the generate_snakes() and generate_ladders() methods.
We need to make sure that the snakes and ladders don't start/end on the same squares OR shoot you off the board OR transport you to a different square on the same row, etc..

THE PIECES:
There will be two arrays that hold the positions of each of the players pieces. When a piece reached 100, that piece has reached the finish line. When ALL of a players pieces reach 100, then that player wins the game.

IN PYTHON:
Snakes_And_Ladders.py
"""
def init_board():
board = np.zeros(100)

def generate_snakes(num_of_snakes):
# Change some random board squares to negative numbers

def generate_ladders(num_of_ladders):
# Change some random board squares to positives numbers

def generate_players_pieces(number_of_pieces):
pieces_one = np.zeros(number_of_pieces)
pieces_two = np.zeros(number_of_pieces)

while playing:

move player 1
check for win
move player 2
check for win
""""
THE MOVE FUNCTION:
The move function will just except the following inputs from the player:

"ENTER BUTTON" to roll the dice
Select which piece to move
This just selects the index of that players array of pieces. Obviously you need to check for invalid selections.

Design Object Oriented class structure for a stop watch similar to the one that appears on a phone.

It should include features for the stopwatch, pause, cancel, reset and laps.

Design a file sharing system
While on site interview, for system design i was asked to create a file sharing system.
Mention the follwoing:

Load balancer for traffic income;
Databases as follows:
Tables keep track of all users;
Tables to keep track of what every user shares(index to whom he shares each file);
Partition your databases as you need to chunk the data to different database nodes;
Tables to keep track of partitions for each file( if you want to return it you need to know the databses where the parts are and also the partition indexes as well so you can recombine the data);
You can come with even more improvements for scalabilty here.

Return the words from dictionary the range query.

Example:

Input: dict = [apple, boy, cat, dog, element, zack, zill]
Output:
range("a", "z"); // returns  [apple, boy, cat, dog, element, zack, zill]
range("ebc", "zas"); // returns [element, zack]

import bisect

class RangeDictionary(object):

	def __init__(self, dic):
		self.wordDictionary = sorted(dic)

	def add_word_to_dictionary(self, word):
		bisect.insort(self.wordDictionary, word)

	def get_words_in_range(self, start, end):
		startInd = bisect.bisect(self.wordDictionary, start)
		endInd = bisect.bisect(self.wordDictionary, end)

		return self.wordDictionary[startInd:endInd]

System Design -  E-Commerce Marketplace analytics
1
Anonymous User
Anonymous User
January 28, 2019 11:35 AM

1.7K VIEWS

Consider a marketplace like Amazon where millions of products belonging to several thousand
categories are sold. Explain your approach building a system that will provide the following and also consider cross functional teams that could be involved to achieve this.

-Top 100 products based on User rating given a category
-Top 100 products based on Sales given a category

The design itself can focus on how the backend of the application will be designed and the
systems that belong to other teams can just be quoted about their need and how they would be
utilized.

my quick analysis:

We can use order processing micro-service to create events/post messages [orderId, productId, categoryId] into a highly available distributed queue, let some other stream processing micro services consume and apply rules and compute rank for each category and write into some reporting tables [we can use distibuted cache to speed up the process ] .
Another way if the load is too much to handle, we can have partitions of product, categories and let the different process run parallely to compute and rank the top products for each category.

Probable Tech stack: Apache Kafka, Memcache, Micro services

Top 20 items sold in last hour
2
Anonymous User
Anonymous User
Last Edit: July 27, 2019 7:48 AM

1.2K VIEWS

Make Amazon.com's top 20 items sold in last hour. The only interface given to you is a function telling you which item is sold, at the time it is sold.

// Invoked as items are being sold
void (Item item);

// Can be called whenever
vector GetTop20SoldInLastHour();


Generate all permutation of a set in Python
Permutation is an arrangement of objects in a specific order. 
Order of arrangement of object is very important. The number of permutations on a set of n elements is given by  n!.  For example, there are 2! = 2*1 = 2 permutations of {1, 2}, namely {1, 2} and {2, 1}, and 3! = 3*2*1 = 6 permutations of {1, 2, 3}, namely {1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2} and {3, 2, 1}.


def toString(List): 
    return ''.join(List) 
  
# Function to print permutations of string 
# This function takes three parameters: 
# 1. String 
# 2. Starting index of the string 
# 3. Ending index of the string. 
def permute(a, l, r): 
    if l==r: 
        print toString(a) 
    else: 
        for i in xrange(l,r+1): 
            a[l], a[i] = a[i], a[l] 
            permute(a, l+1, r) 
            a[l], a[i] = a[i], a[l] # backtrack 
  
# Driver program to test the above function 
string = "ABC"
n = len(string) 
a = list(string) 
permute(a, 0, n-1) 
  
  
  
  
  # Python function to print permutations of a given list 
def permutation(lst): 
  
    # If lst is empty then there are no permutations 
    if len(lst) == 0: 
        return [] 
  
    # If there is only one element in lst then, only 
    # one permuatation is possible 
    if len(lst) == 1: 
        return [lst] 
  
    # Find the permutations for lst if there are 
    # more than 1 characters 
  
    l = [] # empty list that will store current permutation 
  
    # Iterate the input(lst) and calculate the permutation 
    for i in range(len(lst)): 
       m = lst[i] 
       print("m",m)
       
       
       
 # A Python program to print all  
# permutations using library function 
from itertools import permutations 
  
# Get all permutations of [1, 2, 3] 
perm = permutations([1, 2, 3]) 
  
# Print the obtained permutations 
for i in list(perm): 
    print 
  
       # Extract lst[i] or m from the list.  remLst is 
       # remaining list 
       remLst = lst[:i] + lst[i+1:] 
       print("remLst",remLst)
  
       # Generating all permutations where m is first 
       # element 
       for p in permutation(remLst): 
           l.append([m] + p) 
    return l 
  
  
# Driver program to test above function 
data = list('123') 
for p in permutation(data): 
    print (p)
    
    
    
    
    Amazon Locker

Requirements :-

Assign a closest locker to a person given current co-ordinates( where customer wants )
After order is delivered by courier service to customer specified amazon locker, a 6 digit code will be sent to customer .
Item will be placed in Amazon locker for 3 days
After 3 days, if Item is not picked up from the locker, refund process will be initiated
Amazon locker will be assigned to customer based on the size of the locker ( S,M,L,XL,XXL)
Only items that are eligible to be put in locker can be assigned to amazon locker .i.e Not all items can be placed inside locker (like furniture can't be put inside amazon locker)
Access to Amazon locker will depend on Store's opening and closing time.(Since Amazon locker are placed inside stores,malls etc)
Items can be returned to Amazon locker as well .
Items that are eligible Amazon locker item, can only be returned by customer
Once the Door is closed. User's code will be expired. (User will not be able to open the locker now)
Questions I will ask from interviewer :

Who are my customers ? probable answer would be courier guys (For delivery : FedX,Bluedart and For accepting order : Customer who buys)
How many are they?
Is this service to avail globally or to certain cities within a country ?
How many user request may come in a single minute ?
Scenario :

Customer added (amazon locker eligible) item in a cart from amazon.com
there will be cart Microservice that will forward the request to Amazon locker location tracking service based on current location of the user, item size requested
Internal server api call will contains following parameters
List find_Locker(item_id,size_requested,customer_latitude,customer_longitude)

This will find all the lockers based on the size_requested by the customer
All the lockers can be put in a data structure like (K_dimension tree), to effectively search lockers, based on k dimensions
where dimesnsions could be location,size,availibility

Each locker Obect have the property
Locker {
Locker_id
size,
locker_status
}

locker_status {
Booked
Free

}

on next screen user will get the list of nearest amazon lockers available with their address details and their closing and opening timings.

User selects one of the amazon locker from the given lockers location list

api request like (item_id, locker_id,payment_status,expected_delivery_date) will be made from the client side.
6a. Now the locker_id status will be changed to BOOKED, only when payment Status = OK
6a i. when the locker is booked, an RDMS transaction will be commited and locker_status will be changed to BOOKED
Note : we will use RDMS database (MySQL) because locker booking status needs consistency. so MYSQL transactions will provide ACID properties with BEGIN Transaction
and COMMIT feature for booking a current locker. SO that in distributed environment , other customers can not book the same locker.
6a ii. when payment_status is NOT_OK, locker_id will not be booked . error message will be spawned from the (amazon locker booking service) microservice directly to theclient

when the locker is booked. locker_staus will be : BOOKED of given locker_ID

Given locker_id will not be sent to AMAZON LOCKER.

AMAZON LOCKER -> is the actual locker at a particular store
AMAZON LOCKER will generate a two 6 digit code for a given locker_ID

one for BlueDart delivery service and will send code to
one for Customer , code will be generated after Bluedart delivery service has placed the order and closed the door.
Amazon Locker will send this code to Amazon App server.

Code generated by Amazon Locker will be sent through messaging queue.

Messaging queue(like Rabbit MQ) will send the code to app server and client(subscriber)

App server will store the code in their db as well

After customer has taken out the order from the order . Locker status will be changed to FREE and same status will be deliverd by AMAZON locker system to app server, which will in turn will update the locker_id to db.

IF AMAZON LOCKER is closed from 3 days. a request to app server is made to delivery service to pickup the item from the given locker id. 




Prime Numbers
A prime number is a whole number greater than 1, which is only divisible by 1 and itself. First few prime numbers are : 2 3 5 7 11 13 17 19 23 …..



Some interesting fact about Prime numbers

Two is the only even Prime number.
Every prime number can represented in form of 6n+1 or 6n-1 except 2 and 3, where n is natural number.
Two and Three are only two consecutive natural numbers which are prime too.
Goldbach Conjecture: Every even integer greater than 2 can be expressed as the sum of two primes.
GCD of all other natural numbers with a prime is always one.
Wilson Theorem : Wilson’s theorem states that a natural number p > 1 is a prime number if and only if
    (p - 1) ! ≡  -1   mod p 
OR  (p - 1) ! ≡  (p-1) mod p
Fermat’s Little Theorem: If n is a prime number, then for every a, 1 <= a < n,
an-1 ≡ 1 (mod n)
 OR 
an-1 % n = 1 
Prime Number Theorem : The probability that a given, randomly chosen number n is prime is inversely proportional to its number of digits, or to the logarithm of n.
Lemoine’s Conjecture : Any odd integer greater than 5 can be expressed as a sum of an odd prime (all primes other than 2 are odd) and an even semiprime. A semiprime number is a product of two prime numbers.
 This is called Lemoine’s conjecture.
 
  A optimized school method based  
# Python3 program to check 
# if a number is prime 
  
  
def isPrime(n) : 
    # Corner cases 
    if (n <= 1) : 
        return False
    if (n <= 3) : 
        return True
  
    # This is checked so that we can skip  
    # middle five numbers in below loop 
    if (n % 2 == 0 or n % 3 == 0) : 
        return False
  
    i = 5
    while(i * i <= n) : 
        if (n % i == 0 or n % (i + 2) == 0) : 
            return False
        i = i + 6
  
    return True
  
  
# Driver Program  
  
if(isPrime(11)) : 
    print(" true") 
else : 
    print(" false") 
      
if(isPrime(15)) : 
    print(" true") 
else :  
    print(" false") 
      
      
# This code is contributed  
# by Nikita Tiwari. 

Output :
true
false
Time complexity of this solution is O(√n)

Amazon Locker

Requirements :-

Assign a closest locker to a person given current co-ordinates( where customer wants )
After order is delivered by courier service to customer specified amazon locker, a 6 digit code will be sent to customer .
Item will be placed in Amazon locker for 3 days
After 3 days, if Item is not picked up from the locker, refund process will be initiated
Amazon locker will be assigned to customer based on the size of the locker ( S,M,L,XL,XXL)
Only items that are eligible to be put in locker can be assigned to amazon locker .i.e Not all items can be placed inside locker (like furniture can't be put inside amazon locker)
Access to Amazon locker will depend on Store's opening and closing time.(Since Amazon locker are placed inside stores,malls etc)
Items can be returned to Amazon locker as well .
Items that are eligible Amazon locker item, can only be returned by customer
Once the Door is closed. User's code will be expired. (User will not be able to open the locker now)
Questions I will ask from interviewer :

Who are my customers ? probable answer would be courier guys (For delivery : FedX,Bluedart and For accepting order : Customer who buys)
How many are they?
Is this service to avail globally or to certain cities within a country ?
How many user request may come in a single minute ?
Scenario :

Customer added (amazon locker eligible) item in a cart from amazon.com
there will be cart Microservice that will forward the request to Amazon locker location tracking service based on current location of the user, item size requested
Internal server api call will contains following parameters
List find_Locker(item_id,size_requested,customer_latitude,customer_longitude)

This will find all the lockers based on the size_requested by the customer
All the lockers can be put in a data structure like (K_dimension tree), to effectively search lockers, based on k dimensions
where dimesnsions could be location,size,availibility

Each locker Obect have the property
Locker {
Locker_id
size,
locker_status
}

locker_status {
Booked
Free

}

on next screen user will get the list of nearest amazon lockers available with their address details and their closing and opening timings.

User selects one of the amazon locker from the given lockers location list

api request like (item_id, locker_id,payment_status,expected_delivery_date) will be made from the client side.
6a. Now the locker_id status will be changed to BOOKED, only when payment Status = OK
6a i. when the locker is booked, an RDMS transaction will be commited and locker_status will be changed to BOOKED
Note : we will use RDMS database (MySQL) because locker booking status needs consistency. so MYSQL transactions will provide ACID properties with BEGIN Transaction
and COMMIT feature for booking a current locker. SO that in distributed environment , other customers can not book the same locker.
6a ii. when payment_status is NOT_OK, locker_id will not be booked . error message will be spawned from the (amazon locker booking service) microservice directly to theclient

when the locker is booked. locker_staus will be : BOOKED of given locker_ID

Given locker_id will not be sent to AMAZON LOCKER.

AMAZON LOCKER -> is the actual locker at a particular store
AMAZON LOCKER will generate a two 6 digit code for a given locker_ID

one for BlueDart delivery service and will send code to
one for Customer , code will be generated after Bluedart delivery service has placed the order and closed the door.
Amazon Locker will send this code to Amazon App server.

Code generated by Amazon Locker will be sent through messaging queue.

Messaging queue(like Rabbit MQ) will send the code to app server and client(subscriber)

App server will store the code in their db as well

After customer has taken out the order from the order . Locker status will be changed to FREE and same status will be deliverd by AMAZON locker system to app server, which will in turn will update the locker_id to db.

IF AMAZON LOCKER is closed from 3 days. a request to app server is made to delivery service to pickup the item from the given locker id.

222. Count Complete Tree Nodes
Medium

1172

149

Favorite

Share
Given a complete binary tree, count the number of nodes.

Note:

Definition of a complete binary tree from Wikipedia:
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

Example:

Input: 
    1
   / \
  2   3
 / \  /
4  5 6

Output: 6

class Solution:
    
    def countNodes(self, root):
        if not root:
            return 0
        h1, h2 = self.height(root.left), self.height(root.right)
        if h1 > h2: # right child is full 
            return self.countNodes(root.left) +  2 ** h2 
        else: # left child is full 
            return 2 ** h1 + self.countNodes(root.right)

    # the height of the left-most leaf node
    def height1(self, root):
        h = 0
        while root:
            h += 1
            root = root.left
        return h

    def height(self, root):
        if not root:
            return 0
        return self.height(root.left) + 1




Here is one solution comes to my mind-

Read a chunk of data (say 1 GB).
count the excess number of open parenthesis. i.e. "((()())" -> 1, "(()))" -> -1
Do step 1 & 2 until either the whole file is done or return false if you find a negative result after finishing any intermediate step 2.
After finishing the whole file, if you have any count either positive or negative, return false.

https://stackoverflow.com/questions/4500813/distributed-algorithm-to-compute-the-balance-of-the-parentheses

You can break the string into chunks and process each separately, assuming you can read and send to the other machines in parallel. You need two numbers for each string.

The minimum nesting depth achieved relative to the start of the string.

The total gain or loss in nesting depth across the whole string.

With these values, you can compute the values for the concatenation of many chunks as follows:

minNest = 0
totGain = 0
for p in chunkResults
  minNest = min(minNest, totGain + p.minNest)
  totGain += p.totGain
return new ChunkResult(minNest, totGain)

String concatenation

Got this question:
given a list of unique strings, if the last char at string A match first char at string B then you can append them together: good+dog -> goodog . Now return the longest possible string (length of concatenated String, not the string number).

Example: {good, dog, doog, xyhhdgy} --> dogoodoog

Is there anyway can do better than O(n^n)?

I did solved the problem the standard DFS, added all possible result into a list and then iterate the find the longest. However the interviewer said there is a small trick (by selecting the first
 word rather than try them all). Is anyone have idea what is it?
 
 My first approach to this will be use a sliding window of sorts to solve it in 0(N). Find below my pseudocode

Make sure list of strings is in a dynamic structure e.g. ArrayList
Loop through the ArrayList (while will be ideal), compare last character of present item with first character of next item.
If there is a match, concatenate the 2 strings, replace with the present index and delete that next item.
If there is no match then go to the next item.
Exit the while loop if the beginning pointer and end pointer are 
both at the end
Return the longest string at this point (can be
 done if you were storing it in a variable during the while loop or simply loop through your arraylist and select the longest).
 
 
 import collections
class Sol(object):	
	def find_longest_connected_string(self, wordsArr):
		self.longestString = ''

		self.c = collections.Counter(wordsArr)
		print(self.c)

		for word in wordsArr:
			self.c[word] -= 1
			self.backtrack_dfs(word)
			self.c[word] += 1

		return self.longestString

	def backtrack_dfs(self, runningString):
		if len(runningString) > len(self.longestString):
			self.longestString = runningString

		for word in self.c.keys():
			if runningString[-1] == word[0] and self.c[word] > 0:
				self.c[word] -= 1
				self.backtrack_dfs(runningString + word[1:])
				self.c[word] += 1
				
r1=Sol()			
wordsArr1=["good", "dog", "doog", "xyhhdgy"]		
print(r1.find_longest_connected_string(wordsArr1))


Linked List Cycle & Secret Santa
0
iamgodzilla's avatar
iamgodzilla
104
Last Edit: August 7, 2019 1:49 PM

1.6K VIEWS

Position: SDE2

Tell me about an interesting project you worked on.
Below questions were open ended, no function signature was given

https://leetcode.com/problems/reverse-linked-list
https://leetcode.com/problems/linked-list-cycle
Secret Santa. -> Design an api that'd return the secret santa gift list with a constraint that pe
rson cannot gift himself and should handle all the edge cases.

206. Reverse Linked List
Easy

2715

69

Favorite

Share
Reverse a singly linked list.

Example:

Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
Follow up:

A linked list can be reversed either iteratively or recursively. Could you implement both?

class Solution(object):
    def reverseList(self, head):
        
        if head is None:
                return None
        previousHead = None
        nextHead = head.next
        while not nextHead is None:
            head.next = previousHead
            previousHead = head
            head = nextHead
            nextHead = head.next
        head.next = previousHead
        return head
        
#class Solution(object):
    #def reverseList(self, head):
        #rev = None
        #while head: 
            #head.next, rev, head = rev, head, head.next
        #return rev
    
141. Linked List Cycle
Easy

1757

218

Favorite

Share
Given a linked list, determine if it has a cycle in it.

To represent a cycle in the given linked list, we use an integer pos which represents the position (0-indexed) in the linked list where tail connects to. If pos is -1, then there is no cycle in the linked list.

 

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where tail connects to the second node.

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        marker1 = head
        marker2 = head
        while marker2!=None and marker2.next!=None:
            marker1 = marker1.next
            marker2 = marker2.next.next
            if marker2==marker1:
                return True
        return False


Cab Booking And Scheduling


Cab Booking And Scheduling

Let's say, XYZ is a user and wants to travel by cars to the different locations on different scheduled dates and time. Below are the different scenarios:

Cabs available : Cab A, Cab B, Cab C

The user sees the available cabs - Cab A, Cab B, Cab C and schedules booking as below.

User XYZ books Cab A and schedules for 18/01/2018 15:30 PM
User XYZ books CabB and schedules for 20/01/2018 12:30 PM
User XYZ books CabC and schedules for 20/01/2018 16:30 PM
The user can cancel the scheduled cab before the arrival of the cab at the scheduled time.
https://www.c-sharpcorner.com/article/cab-booking-and-scheduling-by-using-command-design-pattern-and-scheduler/

Design food delivery system

Design OO food delivery app catering to use cases -

User can search different restaurant

User can select a restaurant

User sees a menu

Restaurant can change the menu any time

User adds an item from menu

User orders the food

User can track the order in real time

User can cancel the order

User pays for the orderDesign OO food delivery app with C# & Design Patterns
https://github.com/gmershad/FoodDeliveryApp

Word Break II variation


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
        
        
        
         First word whose reverse was present in the string
         
         
       # Python implementation of the approach 
  
# Function that returns true if s1  
# is equal to reverse of s2 
def isReverseEqual(s1, s2): 
  
    # If both the strings differ in length 
    if len(s1) != len(s2): 
        return False
      
    l = len(s1) 
  
    for i in range(l): 
  
        # In case of any character mismatch 
        if s1[i] != s2[l-i-1]: 
            return False
    return True
  
# Function to return the first word whose  
# reverse is also present in the array 
def getWord(str, n): 
  
    # Check every string 
    for i in range(n-1): 
  
        # Pair with every other string 
        # appearing after the current string 
        for j in range(i+1, n): 
  
            # If first string is equal to the 
            # reverse of the second string 
            if (isReverseEqual(str[i], str[j])): 
                return str[i] 
      
    # No such string exists 
    return "-1"
  
  
# Driver code 
if __name__ == "__main__": 
    str = ["geeks", "for", "skeeg"] 
    print(getWord(str, 3)) 
  
# This code is contributed by 
# sanjeev2552 
Output:
geeks

https://www.geeksforgeeks.org/first-string-from-the-given-array-whose-reverse-is-also-present-in-the-same-array/


Maximize profit by installing billboards on road
0
Anonymous User
Anonymous User
Last Edit: July 21, 2019 7:33 AM

116 VIEWS

There is a huge road. Given are the following

Array D that stores the distance from a starting point where billboard can be installed.
Array C that stores the profit. C[i] -> profit if the billboard is installed at distance D[i].
dist -> minimum distance to maintain between the billboards.
Assume you can install any number of billboards while maintaining a given minimum distance 'dist'
 between each of them. Find the maximum profit you can achieve.

Let maxRev[i], 1 <= i <= M, be the maximum revenue generated from beginning to i miles on the highway. Now for each mile on the highway, we need to check whether this mile has the option for any billboard, if not then the maximum revenue generated till that mile would be same as maximum revenue generated till one mile before. But if that mile has the option for billboard then we have 2 options:
1. Either we will place the billboard, ignore the billboard in previous t miles, and add the revenue of the billboard placed.
2. Ignore this billboard. So maxRev[i] = max(maxRev[i-t-1] + revenue[i], maxRev[i-1])

Below is implementation of this approach:

filter_none
edit
play_arrow

brightness_4
# Python3 program to find maximum revenue  
# by placing billboard on the highway with 
# given constarints.  
  
def maxRevenue(m, x, revenue, n, t) : 
      
    # Array to store maximum revenue  
    # at each miles.  
    maxRev = [0] * (m + 1) 
  
    # actual minimum distance between  
    # 2 billboards.  
    nxtbb = 0;  
    for i in range(1, m + 1) : 
          
        # check if all billboards are  
        # already placed.  
        if (nxtbb < n) : 
              
            # check if we have billboard for  
            # that particular mile. If not,  
            # copy the previous maximum revenue.  
            if (x[nxtbb] != i) : 
                maxRev[i] = maxRev[i - 1]  
  
            # we do have billboard for this mile.  
            else : 
              
                # We have 2 options, we either take  
                # current or we ignore current billboard.  
  
                # If current position is less than or  
                # equal to t, then we can have only 
                # one billboard.  
                if (i <= t) : 
                    maxRev[i] = max(maxRev[i - 1], 
                                    revenue[nxtbb]) 
  
                # Else we may have to remove  
                # previously placed billboard  
                else : 
                    maxRev[i] = max(maxRev[i - t - 1] + 
                                    revenue[nxtbb],  
                                    maxRev[i - 1]);  
  
                nxtbb += 1
      
        else : 
              
            maxRev[i] = maxRev[i - 1]  
      
    return maxRev[m] 
      
# Driver Code 
if __name__ == "__main__" : 
      
    m = 20
    x = [6, 7, 12, 13, 14] 
    revenue = [5, 6, 5, 3, 1]  
    n = len(x) 
    t = 5
    print(maxRevenue(m, x, revenue, n, t))  
  
# This code is contributed by Ryuga 

https://www.geeksforgeeks.org/highway-billboard-problem/

Find faulty commit
2
gurmeet1992's avatar
gurmeet1992
2
Last Edit: July 27, 2019 7:44 AM

1.3K VIEWS

Position: SDE2

There are two source code repositories app01 and app02 which gets deployed at 7 PM in production every day .
You are given below function to deploy the code from app01 and app02 in production up-to a given commit id.

boolean deploy(int commitId_app01, int commitId_app02)

if we call deploy(10, 12) then every commit from id 1 to 10 from app01 and every commit from 1 to 12 from app02 will be deployed in production
and the function will return either true if deployment is successful or false if deployment failed .

A deployment is made (i.e deploy(n, m)) and deployment failed because of a faulty commit you need to write an algorithm to find the faulty commit
using the deploy method in least number of deployments.

278. First Bad Version
Easy

738

453

Favorite

Share
You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API bool isBadVersion(version) which will return whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.

Example:

Given n = 5, and version = 4 is the first bad version.

call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true

Then 4 is the first bad version. 

class Solution(object):
def firstBadVersion(self, n):
    """
    :type n: int
    :rtype: int
    """
    r = n-1
    l = 0
    while(l<=r):
        mid = l + (r-l)/2
        if isBadVersion(mid)==False:
            l = mid+1
        else:
            r = mid-1
    return l
    
    Next challenges:
Guess Number Higher or Lower



Run Length Encoding in Python
Given an input string, write a function that returns the Run Length Encoded string for the input string.

For example, if the input string is ‘wwwwaaadexxxxxx’, then the function should return ‘w4a3d1e1x6’.

Examples:

Input  :  str = 'wwwwaaadexxxxxx'
Output : 'w4a3d1e1x6'
Recommended: Please try your approach on {IDE} first, before moving on to the solution.
This problem has existing solution please refer Run Length Encoding link. Here we will solve this problem quickly in python using OrderedDict. Approach is very simple, first we create a ordered dictionary which contains characters of input string as key and 0 as their default value, now we run a loop to count frequency of each character and will map it to it’s corresponding key.

filter_none
edit
play_arrow

brightness_4
# Python code for run length encoding 
from collections import OrderedDict 
def runLengthEncoding(input): 
  
    # Generate ordered dictionary of all lower 
    # case alphabets, its output will be  
    # dict = {'w':0, 'a':0, 'd':0, 'e':0, 'x':0} 
    dict=OrderedDict.fromkeys(input, 0) 
  
    # Now iterate through input string to calculate  
    # frequency of each character, its output will be  
    # dict = {'w':4,'a':3,'d':1,'e':1,'x':6} 
    for ch in input: 
        dict[ch] += 1
  
    # now iterate through dictionary to make  
    # output string from (key,value) pairs 
    output = '' 
    for key,value in dict.iteritems(): 
         output = output + key + str(value) 
    return output 
   
# Driver function 
if __name__ == "__main__": 
    input='wwwwaaadexxxxxx'
    print runLengthEncoding(input) 
Output:

'w4a3d1e1x6'

Cache with TTL
1
Anonymous User
Anonymous User
Last Edit: August 20, 2019 4:06 AM

665 VIEWS

Implement Time To Live(TTL) in LRU cache. What could be the most optimal way ?

146. LRU Cache
Medium

3466

135

Favorite

Share
Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and put.

get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
put(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

The cache is initialized with a positive capacity.

Follow up:
Could you do both operations in O(1) time complexity?

Example:

LRUCache cache = new LRUCache( 2 /* capacity */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4


from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.dict = OrderedDict()

    def get(self, key):
        if key not in self.dict:
            return -1
        self.dict.move_to_end(key)
        return self.dict[key]

    def put(self, key,value):
        if key in self.dict:
            self.dict.pop(key)
            
        self.dict[key] = value
        if len(self.dict) > self.cap:
            self.dict.popitem(last=False) 

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

# Python3 implementation to sort 
# the given matrix 
  
SIZE = 10
  
# Function to sort the given matrix 
def sortMat(mat, n) : 
      
    # Temporary matrix of size n^2 
    temp = [0] * (n * n) 
    k = 0
  
    # Copy the elements of matrix   
    # one by one into temp[] 
    for i in range(0, n) : 
          
        for j in range(0, n) : 
              
            temp[k] = mat[i][j] 
            k += 1
  
    # sort temp[] 
    temp.sort() 
      
    # copy the elements of temp[]  
    # one by one in mat[][] 
    k = 0
      
    for i in range(0, n) : 
          
        for j in range(0, n) : 
            mat[i][j] = temp[k] 
            k += 1
  
  
# Function to print the given matrix 
def printMat(mat, n) : 
      
    for i in range(0, n) : 
          
        for j in range( 0, n ) : 
              
            print(mat[i][j] , end = " ") 
              
        print() 
      
      
# Driver program to test above 
mat = [ [ 5, 4, 7 ], 
        [ 1, 3, 8 ], 
        [ 2, 9, 6 ] ] 
n = 3
  
print( "Original Matrix:") 
printMat(mat, n) 
  
sortMat(mat, n) 
  
print("\nMatrix After Sorting:") 
printMat(mat, n) 
  
  
# This code is contributed by Nikita Tiwari. 

Time Complexity: O(n2log2n).
Auxiliary Space: O(n2).

SSL (Secure Sockets Layer) is the standard security technology for establishing an encrypted link between a web server and a browser. This link ensures that all data passed between the web server and browsers remain private and integral. SSL is an industry standard and is used by millions of 
websites in the protection of their online transactions with their customers.
https://www.ssl.com/faqs/faq-what-is-ssl/




from collections import OrderedDict
def non_repeated(s):
	dict =OrderedDict()
	s = s.lower()
	for i in s.replace('.','').split(' '):
		dict[i]= dict.get(i,0)+1
	
	for i in dict:
		if dict[i] ==1:
			return i
	
s = "The angry dog was red. And the cat was red angry"
print non_repeated(s)
First unique integer

387. First Unique Character in a String
Easy

1162

85

Favorite

Share
Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.

Examples:

s = "leetcode"
return 0.

s = "loveleetcode",
return 2.

class Solution:
    def firstUniqChar(self, s):
        d = {}
        for l in s:
            if l not in d: d[l] = 1
            else: d[l] += 1
        
        index = -1
        for i in range(len(s)):
            print(d[s[i]])
            if d[s[i]] == 1:
                index = i
                break
        
        return index
        
 Design an HR web portal for Amazon's recruiting team
17
TrajanWOWS's avatar
TrajanWOWS
17
Last Edit: June 16, 2019 3:50 PM

1.9K VIEWS

image


I could think of following features along with possible technical stack based on the individual feature we plan to implement.

Possible Features :

i) List of candidates applied.
ii) Look into individual candidate profile in details.
iii) Sending the profile to hiring manager for proceedings.
iv) Once it's done then send a link for the initial evaluation round.
v) Check the present status and then organizing further rounds.
vi) Resume reading capabilities from the candidate details section.
vii) Interview comments sections for each round. ( atleast 5/6 ) of tabular input fields.

User Persona :
SysAdmin
Recruiting Manager / Sr. Hr
Executives
Hiring Manager
Interviewers

Candidate Profile Satatus:

Ready to be hired.
Interviewed (Round Specific 1/2/3 etc..)
Rejected
Selected
Offered Given.
Offer Accepted.
Offer Rejected
Log monitoring and management :

Different alerting and email notification mechanism.

Technological Stack Point of view :
UI : React/Angular with nginx as hosting app server with docker based deployement.
Microservices layer : Could be in python. For async task based operations it could be set up with celery.
API Gateway : to plugin all set of microservices endpoint.
DB : MySQL for the metadata management. And for the documents tracking any no sql like mongodb.
Log monitoring : Elastic stack.


# Python code t get difference of two lists 
# Using set() 
def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 
  
# Driver Code 
li1 = [10, 15, 20, 25, 30, 35, 40] 
li2 = [25, 40, 35] 
print(Diff(li1, li2)) 

# Python code t get difference of two lists 
# Not using set() 
def Diff(li1, li2): 
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2] 
    return li_dif 
  
# Driver Code 
li1 = [10, 15, 20, 25, 30, 35, 40] 
li2 = [25, 40, 35] 
li3 = Diff(li1, li2) 
print(li3)

Amazon | Data Engineer Role | Database Design Question


I was asked this question for amazon data engineer role. How to design data model for Customer, Products, Wishlist, Orders, Cart. The web site receives 
about 2 million hits every second. Also how would I do ETL on the tables.

2 million hits implies that it's difficult to use SQL databases (even Amazon Aurora caps at 200000 writes per second). We will likely need to use a K-V store like Voldemort or DynamoDB. Each customer can be identified using an UUID and have a json representing wishlist, orders, cart, and other customer data (like addresses, name, preferences, etc.) Note that you should not store the customer's user id and password in the same K-V store. Ideally, consistency guarantees for such login details should be strict, so SQL is more suitable. This because you can't have staleness, when for eg, a password is changed. Products can be also be 
identified using UUID and have their own jsons containing data like price, reviews, etc. Any customer json can reference product UUIDs in the cart, wishlist, etc.

In order to perform analytics, since it will likely be offline processing, 
you may choose to replicate the data in the K-V store to another slower but more scalable database. This option is considered since you'd want to save the QPS of K-V store for customer queries as much as possible. In case the K-V store is already very scalable, you can disregard this OLAP database. ETL can be done using MapReduce. Specifically, for queries like "What are the age groups interested in this product", "How popular is the product in this area", "How many customers shopped today", etc. mapreduce can be used to perform mapping, filtering, and reducing on the K-V data to gather insights. You can drill down on a single query and talk about it more.

For e.g, let's consider: what are the age groups interested in this product.
Mapper maps the customer json to the customer age, if the customer's wishlist, cart, or orders have the product (or set of related products). Once this is done, the intermediate data store (on the mapper perhaps) will have data tuples like this (customer UUID: age). Mapper can sort this data before Reduce step, so that all identical UUIDs come together. Then, each mapper can send customers within a certain UUID range to certain reducers, to maintain locality which can then perform a basic count operation by reduction, while taking care to deduplicate entries containing same UUIDs (unless you want to collect data about how often the same customer wants the product in that time frame).

388. Longest Absolute File Path
Medium

465

962

Favorite

Share
Suppose we abstract our file system by a string in the following manner:

The string "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents:

dir
    subdir1
    subdir2
        file.ext
The directory dir contains an empty sub-directory subdir1 and a sub-directory subdir2 containing a file file.ext.

lass Solution:
    def lengthLongestPath(self, input):
        l = input.split('\n')
        cdir = []
        m = 0
        
        for d in l:
            l_cdir = len(cdir)
            lvl = d.count('\t')
            if l_cdir > lvl:
                cdir = cdir[:-(l_cdir-lvl)]
            cdir.append(d)
            if '.' in cdir[len(cdir)-1]:
                com = '/'.join(cdir).replace('\t', '')
                m = max(m, len(com))
                # print(com)
        
        return m
Next challenges:
Find Anagram Mappings
Rotated Digits
Verifying an Alien Dictionary


1032. Stream of Characters
Hard

120

32

Favorite

Share
Implement the StreamChecker class as follows:

StreamChecker(words): Constructor, init the data structure with the given words.
query(letter): returns true if and only if for some k >= 1, the last k characters queried (in order from oldest to newest, including this letter just queried) spell one of the words in the given list.
 

Example:

StreamChecker streamChecker = new StreamChecker(["cd","f","kl"]); // init the dictionary.
streamChecker.query('a');          // return false
streamChecker.query('b');          // return false
streamChecker.query('c');          // return false
streamChecker.query('d');          // return true, because 'cd' is in the wordlist
streamChecker.query('e');          // return false
streamChecker.query('f');          // return true, because 'f' is in the wordlist
streamChecker.query('g');          // return false
streamChecker.query('h');          // return false
streamChecker.query('i');          // return false
streamChecker.query('j');          // return false
streamChecker.query('k');          // return false
streamChecker.query('l');          // return true, because 'kl' is in the wordlist
 

Note:

1 <= words.length <= 2000
1 <= words[i].length <= 2000
Words will only consist of lowercase English letters.
Queries will only consist of lowercase English letters.
The number of queries is at most 40000.
Accepted

200. Number of Islands
Medium

3062

109

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
class Solution(object):
    def numIslands(self, grid):
        def sink(i, j):
            if 0 <= i < len(grid) and 0 <= j < len(grid[i]) and grid[i][j] == '1':
                grid[i][j] = '0'
                list(map(sink, (i+1, i-1, i, i), (j, j, j+1, j-1)))  # map in python3 return iterator
                return 1
            return 0
        return sum(sink(i, j) for i in range(len(grid)) for j in range(len(grid[i])))

Determinant of a Matrix
What is Determinant of a Matrix?
Determinant of a Matrix is a special number that is defined only for square matrices (matrices which have same number of rows and columns). Determinant is used at many places in calculus and other matrix related algebra, it actually represents the matrix in term of a real number which can be used in solving system of linear equation and finding the inverse of a matrix.

How to calculate?
The value of determinant of a matrix can be calculated by following procedure –
For each element of first row or first column get cofactor of those elements and then multiply the element with the determinant of the corresponding cofactor, and finally add them with alternate signs. As a base case the value of determinant of a 1*1 matrix is the single value itself.
Cofactor of an element, is a matrix which we can get by removing row and column of that element from that matrix.

Determinant of 2 x 2 Matrix:

 A = \begin{bmatrix} a & b\\  c & d \end{bmatrix}  \begin{vmatrix} A \end{vmatrix}= ad - bc 

22

Determinant of 3 x 3 Matrix:
 A = \begin{bmatrix} a & b & c\\  d & e & f\\  g & h & i \end{bmatrix}  \begin{vmatrix} A \end{vmatrix}= a(ei-fh)-b(di-gf)+c(dh-eg) 

https://www.geeksforgeeks.org/determinant-of-a-matrix/

Convert BST to Greater Tree
0
Anonymous User
Anonymous User
Last Edit: August 6, 2019 3:36 PM

903 VIEWS

https://leetcode.com/problems/convert-bst-to-greater-tree

Interviewer only wanted a solution wirh constant space and I was unable to complete the problem. He did say that he saw where i was going with the solution. I don't think you can solve this in constant space on the spot if you've never heard about Morris Traversal.

Morris in-order traversal using threading
Main article: Threaded binary tree
A binary tree is threaded by making every left child pointer (that would otherwise be null) point to the in-order predecessor of the node (if it exists) and every right child pointer (that would otherwise be null) point to the in-order successor of the node (if it exists).

Advantages:

Avoids recursion, which uses a call stack and consumes memory and time.
The node keeps a record of its parent.
Disadvantages:

The tree is more complex.
We can make only one traversal at a time.
It is more prone to errors when both the children are not present and both values of nodes point to their ancestors.
Morris traversal is an implementation of in-order traversal that uses threading:[6]

Create links to the in-order successor.
Print the data using these links.
Revert the changes to restore original tree.
# Python program to do inorder traversal without recursion and  
# without stack Morris inOrder Traversal 
  
# A binary tree node 
class Node: 
      
    # Constructor to create a new node 
    def __init__(self, data): 
        self.data = data  
        self.left = None
        self.right = None
  
# Iterative function for inorder tree traversal 
def MorrisTraversal(root): 
      
    # Set current to root of binary tree 
    current = root  
      
    while(current is not None): 
          
        if current.left is None: 
            print current.data, 
            current = current.right 
        else: 
            # Find the inorder predecessor of current 
            pre = current.left 
            while(pre.right is not None and pre.right != current): 
                pre = pre.right 
   
            # Make current as right child of its inorder predecessor 
            if(pre.right is None): 
                pre.right = current 
                current = current.left 
                  
            # Revert the changes made in if part to restore the  
            # original tree i.e., fix the right child of predecessor 
            else: 
                pre.right = None
                print current.data, 
                current = current.right 
              
# Driver program to test the above function 
"""  
Constructed binary tree is 
            1 
          /   \ 
        2      3 
      /  \ 
    4     5 
"""
root = Node(1) 
root.left = Node(2) 
root.right = Node(3) 
root.left.left = Node(4) 
root.left.right = Node(5) 



Rat in a Maze | Backtracking-2
We have discussed Backtracking and Knight’s tour problem in Set 1. Let us discuss Rat in a Maze as another example problem that can be solved using Backtracking.
A Maze is given as N*N binary matrix of blocks where source block is the upper left most block i.e., maze[0][0] and destination block is lower rightmost block i.e., maze[N-1][N-1]. A rat starts from source and has to reach the destination. The rat can move only in two directions: forward and down.
In the maze matrix, 0 means the block is a dead end and 1 means the block can be used in the path from source to destination. Note that this is a simple version of the typical Maze problem. For example, a more complex version can be that the rat can move in 4 directions and a more complex version can be with a limited number of moves.

Following is an example maze.

 Gray blocks are dead ends (value = 0). 


Following is binary matrix representation of the above maze.

                {1, 0, 0, 0}
                {1, 1, 0, 1}
                {0, 1, 0, 0}
                {1, 1, 1, 1}
Following is a maze with highlighted solution path.


brightness_4
# Python3 program to solve Rat in a Maze  
# problem using backracking  
  
# Maze size 
N = 4
  
# A utility function to print solution matrix sol 
def printSolution( sol ): 
      
    for i in sol: 
        for j in i: 
            print(str(j) + " ", end ="") 
        print("") 
  
# A utility function to check if x, y is valid 
# index for N * N Maze 
def isSafe( maze, x, y ): 
      
    if x >= 0 and x < N and y >= 0 and y < N and maze[x][y] == 1: 
        return True
      
    return False
  
""" This function solves the Maze problem using Backtracking.  
    It mainly uses solveMazeUtil() to solve the problem. It  
    returns false if no path is possible, otherwise return  
    true and prints the path in the form of 1s. Please note 
    that there may be more than one solutions, this function 
    prints one of the feasable solutions. """
def solveMaze( maze ): 
      
    # Creating a 4 * 4 2-D list 
    sol = [ [ 0 for j in range(4) ] for i in range(4) ] 
      
    if solveMazeUtil(maze, 0, 0, sol) == False: 
        print("Solution doesn't exist"); 
        return False
      
    printSolution(sol) 
    return True
      
# A recursive utility function to solve Maze problem 
def solveMazeUtil(maze, x, y, sol): 
      
    # if (x, y is goal) return True 
    if x == N - 1 and y == N - 1: 
        sol[x][y] = 1
        return True
          
    # Check if maze[x][y] is valid 
    if isSafe(maze, x, y) == True: 
        # mark x, y as part of solution path 
        sol[x][y] = 1
          
        # Move forward in x direction 
        if solveMazeUtil(maze, x + 1, y, sol) == True: 
            return True
              
        # If moving in x direction doesn't give solution  
        # then Move down in y direction 
        if solveMazeUtil(maze, x, y + 1, sol) == True: 
            return True
          
        # If none of the above movements work then  
        # BACKTRACK: unmark x, y as part of solution path 
        sol[x][y] = 0
        return False
  
# Driver program to test above function 
if __name__ == "__main__": 
    # Initialising the maze 
    maze = [ [1, 0, 0, 0], 
             [1, 1, 0, 1], 
             [0, 1, 0, 0], 
             [1, 1, 1, 1] ] 
               
    solveMaze(maze) 
  
# This code is contributed by Shiv Shankar 
  
MorrisTraversal(root) 
  
# This code is contributed by Naveen Aili 

Encryption is a two-way function; what is encrypted can be decrypted with the proper key. Hashing, however, is a one-way function that scrambles plain text to produce a unique message digest.
 With a properly designed algorithm, 
 
 In CPython, the global interpreter lock, or GIL, is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecodes at once. This lock is necessary mainly 
 
 
 # Python3 code to find minimum steps to reach  
# to specific cell in minimum moves by Knight  
class cell: 
      
    def __init__(self, x = 0, y = 0, dist = 0): 
        self.x = x 
        self.y = y 
        self.dist = dist 
          
# checks whether given position is  
# inside the board 
def isInside(x, y, N): 
    if (x >= 1 and x <= N and 
        y >= 1 and y <= N):  
        return True
    return False
      
# Method returns minimum step to reach 
# target position  
def minStepToReachTarget(knightpos,  
                         targetpos, N): 
      
    #all possible movments for the knight 
    dx = [2, 2, -2, -2, 1, 1, -1, -1] 
    dy = [1, -1, 1, -1, 2, -2, 2, -2] 
      
    queue = [] 
      
    # push starting position of knight 
    # with 0 distance 
    queue.append(cell(knightpos[0], knightpos[1], 0)) 
      
    # make all cell unvisited  
    visited = [[False for i in range(N + 1)]  
                      for j in range(N + 1)] 
      
    # visit starting state 
    visited[knightpos[0]][knightpos[1]] = True
      
    # loop untill we have one element in queue  
    while(len(queue) > 0): 
          
        t = queue[0] 
        queue.pop(0) 
          
        # if current cell is equal to target  
        # cell, return its distance  
        if(t.x == targetpos[0] and 
           t.y == targetpos[1]): 
            return t.dist 
              
        # iterate for all reachable states  
        for i in range(8): 
              
            x = t.x + dx[i] 
            y = t.y + dy[i] 
              
            if(isInside(x, y, N) and not visited[x][y]): 
                visited[x][y] = True
                queue.append(cell(x, y, t.dist + 1)) 
  
# Driver Code      
if __name__=='__main__':  
    N = 30
    knightpos = [1, 1] 
    targetpos = [30, 30] 
    print(minStepToReachTarget(knightpos, 
                               targetpos, N)) 
      
# This code is contributed by  
# Kaustav kumar Chanda because CPython's memory management is not thread-safe.



Maximum element in min heap
Given a min heap, find the maximum element present in the heap.

# Python 3 implementation of 
# above approach 
  
# Function to find the maximum  
# element in a max heap 
def findMaximumElement(heap, n): 
      
    maximumElement = heap[n // 2] 
      
    for i in range(1 + n // 2, n): 
        maximumElement = max(maximumElement,  
                             heap[i]) 
    return maximumElement 
  


Given a continous stream of characters, find the first non repeating character at any given point => https://www.geeksforgeeks.org/find-first-non-repeating-character-stream-characters/
Given a tree with left, right and next pointer, update next pointer to point to the right node at same level => https://leetcode.com/problems/populating-next-right-pointers-in-each-node/. Told the interviewer that I'd already done the question
Given two arrays containing numbers, find the difference of closest greatest of each number from left and right ?
3 2 1 7 5
0 3 2 0 7 => from left (here for number 1 since 2 and 3 are both greater, we pick the closest viz. 2)
7 7 7 0 0 => from right
7 4 5 0 7 => difference



Location: Bangalore
Position: SDE-II
Duration: 60 minutes

Given a continous stream of characters, find the first non repeating character at any given point => https://www.geeksforgeeks.org/find-first-non-repeating-character-stream-characters/
Given a tree with left, right and next pointer, update next pointer to point to the right node at same level => https://leetcode.com/problems/populating-next-right-pointers-in-each-node/. Told the interviewer that I'd already done the question
Given two arrays containing numbers, find the difference of closest greatest of each number from left and right ?
3 2 1 7 5
0 3 2 0 7 => from left (here for number 1 since 2 and 3 are both greater, we pick the closest viz. 2)
7 7 7 0 0 => from right
7 4 5 0 7 => difference

# Python program to print the first non-repeating character 
NO_OF_CHARS = 256
  
# Returns an array of size 256 containg count 
# of characters in the passed char array 
def getCharCountArray(string): 
    count = [0] * NO_OF_CHARS 
    for i in string: 
        count[ord(i)]+=1
    return count 
  
# The function returns index of first non-repeating 
# character in a string. If all characters are repeating 
# then returns -1 
def firstNonRepeating(string): 
    count = getCharCountArray(string) 
    index = -1
    k = 0
  
    for i in string: 
        if count[ord(i)] == 1: 
            index = k 
            break
        k += 1
  
    return index 
  
# Driver program to test above function 
string = "geeksforgeeks"
index = firstNonRepeating(string) 
if index==1: 
    print "Either all characters are repeating or string is empty"
else: 
    print "First non-repeating character is " + string[index] 
  
# This code is contributed by Bhavya Jain 

K’th Non-repeating Character

 Python 3 program to find k'th  
# non-repeating character in a string 
MAX_CHAR = 256
  
# Returns index of k'th non-repeating  
# character in given string str[] 
def kthNonRepeating(str, k): 
  
    n = len(str) 
  
    # count[x] is going to store count of   
    # character 'x' in str. If x is not  
    # present, then it is going to store 0. 
    count = [0] * MAX_CHAR 
  
    # index[x] is going to store index of  
    # character 'x' in str. If x is not  
    # present or x is repeating, then it  
    # is going to store a value (for example,  
    # length of string) that cannot be a valid 
    # index in str[] 
    index = [0] * MAX_CHAR 
  
    # Initialize counts of all characters  
    # and indexes of non-repeating characters. 
    for i in range( MAX_CHAR): 
        count[i] = 0
        index[i] = n # A value more than any  
                     # index in str[] 
  
    # Traverse the input string 
    for i in range(n): 
          
        # Find current character and  
        # increment its count 
        x = str[i] 
        count[ord(x)] += 1
  
        # If this is first occurrence, then  
        # set value in index as index of it. 
        if (count[ord(x)] == 1): 
            index[ord(x)] = i 
  
        # If character repeats, then remove  
        # it from index[] 
        if (count[ord(x)] == 2): 
            index[ord(x)] = n 
  
    # Sort index[] in increasing order. This step 
    # takes O(1) time as size of index is 256 only 
    index.sort() 
  
    # After sorting, if index[k-1] is value,  
    # then return it, else return -1. 
    return index[k - 1] if (index[k - 1] != n) else -1
  
# Driver code 
if __name__ == "__main__": 
    str = "geeksforgeeks"
    k = 3
    res = kthNonRepeating(str, k) 
    if(res == -1): 
        print("There are less than k", 
              "non-repeating characters") 
    else: 
        print("k'th non-repeating character is",  
                                       str[res]) 
  
# This code is contributed 
# by ChitraNayal 

116. Populating Next Right Pointers in Each Node
Medium

1161

100

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

Populating Next Right Pointers in Each Node

class Solution(object):
    def connect(self, root):
        """
        :type root: TreeLinkNode
        :rtype: nothing
        """
        
        if not root:
            return None
        cur  = root
        next = root.left

        while cur.left :
            cur.left.next = cur.right
            if cur.next:
                cur.right.next = cur.next.left
                cur = cur.next
            else:
                cur = next
                next = cur.left
        return root
739. Daily Temperatures
Medium

1542

45

Favorite

Share
Given a list of daily temperatures T, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead.

For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].

Note: The length of temperatures will be in the range [1, 30000]. Each temperature will be an integer in the range [30, 100].


739. Daily Temperatures
Medium

1542

45

Favorite

Share
Given a list of daily temperatures T, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead.

For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].

Note: The length of temperatures will be in the range [1, 30000]. Each temperature will be an integer in the range [30, 100].

class Solution(object):
    def dailyTemperatures(self, T):
        
        if len(T)==0: return []
        stack = []
        ans = [0]*len(T)
        pos = 0
        for i in range(len(T)):
            while stack and stack[-1][1] < T[i]:
                #print(stack[-1][1],T[i])
                idx, _ = stack.pop()
                #print(idx)
                ans[idx] = i - idx
            stack.append((pos,T[i]))
            pos += 1
        return ans
        
  30. Substring with Concatenation of All Words
Hard

606

987

Favorite

Share
You are given a string, s, and a list of words, words, that are all of the same length. Find all starting indices of substring(s) in s that is a concatenation of each word in words exactly once and without any intervening characters.

Example 1:

Input:
  s = "barfoothefoobarman",
  words = ["foo","bar"]
Output: [0,9]
Explanation: Substrings starting at index 0 and 9 are "barfoor" and "foobar" respectively.
The output order does not matter, returning [9,0] is fine too.
Example 2:

Input:
  s = "wordgoodgoodgoodbestword",
  words = ["word","good","best","word"]
Output: []

class Solution(object):
    def findSubstring(self, s, words):
        n, m, r = len(words), len(words[0]) if words else 0, []
        counter = collections.Counter(words)
        #print(counter)

        for i in xrange(m):
            localCout = collections.defaultdict(int)
            #print(localCout)
            window = collections.deque()

            for j in xrange(i, len(s), m):
                word = s[j:j + m]
                print(word)
                if word in counter:
                    localCout[word] += 1
                    window.append(word)

                    while localCout[word] > counter[word]:
                        localCout[window.popleft()] -= 1

                    if len(window) == n:
                        r.append(j - (n - 1) * m)
                else:
                    localCout.clear()
                    window.clear()
        return r
        
  472. Concatenated Words
Hard

306

72

Favorite

Share
Given a list of words (without duplicates), please write a program that returns all concatenated words in the given list of words.
A concatenated word is defined as a string that is comprised entirely of at least two shorter words in the given array.

Example:
Input: ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]

Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]

Explanation: "catsdogcats" can be concatenated by "cats", "dog" and "cats"; 
 "dogcatsdog" can be concatenated by "dog", "cats" and "dog"; 
"ratcatdogcat" can be concatenated by "rat", "cat", "dog" and "cat".


Next challenges:
Range Sum Query - Immutable
Best Time to Buy and Sell Stock with Transaction Fee
Making A Large Island

class Solution(object):
    def findAllConcatenatedWordsInADict(self, words):
        W, res = set(words), []
        #print(W)
        
        for w in words:
            matches = [len(w)]
            #print(matches)
            for i in range(len(w)-1, 0, -1):
                if any(w[i:mi] in W for mi in matches): 
                    #print(i)
                    matches.append(i)
            #print("matches[1:]",matches[1:])
            if any(w[:mi] in W for mi in matches[1:]): 
                #print(res)
                res.append(w)
                        
        return res
        
        
   Movies on Flight
18
Anonymous User
Anonymous User
Last Edit: August 13, 2019 2:37 AM

8.2K VIEWS

I had 2 question on my online assesment, however I remeber only the first one. My code passed only 10 test out of 13. I did a sorting and then found the best pair with 2 for loops

Question:
You are on a flight and wanna watch two movies during this flight.
You are given int[] movie_duration which includes all the movie durations.
You are also given the duration of the flight which is d in minutes.
Now, you need to pick two movies and the total duration of the two movies is less than or equal to (d - 30min).
Find the pair of movies with the longest total duration. If multiple found, return the pair with the longest movie.

Example 1:

Input: movie_duration = [90, 85, 75, 60, 120, 150, 125], d = 250
Output: [90, 125]
Explanation: 90min + 125min = 215 is the maximum number within 220 (250min - 30min)     
        
        def flightDetails(arr, k):
    k-=30
    arr = sorted(arr)
    left = 0
    right = len(arr)-1
    max_val = 0
    while left<right:
        if arr[left]+arr[right]<=k:
            if max_val < arr[left]+arr[right]:
                max_val = arr[left]+arr[right]
                i = left
                j = right
            left+=1
        else:
            right-=1
    return(arr[i],arr[j])

arr = [90, 85, 75, 60, 120, 150, 125]
k = 250
print(flightDetails(arr,k))



Expression Tree
Expression tree is a binary tree in which each internal node corresponds to operator and each leaf node corresponds to operand so for example expression tree for 3 + ((5+9)*2) would be:
expressiontre

Inorder traversal of expression tree produces infix version of given postfix expression (same with preorder traversal it gives prefix expression)

Evaluating the expression represented by expression tree:

Let t be the expression tree
If  t is not null then
      If t.value is operand then  
                Return  t.value
      A = solve(t.left)
      B = solve(t.right)
 
      // calculate applies operator 't.value' 
      // on A and B, and returns value
      Return calculate(A, B, t.value)   
Construction of Expression Tree:
Now For constructing expression tree we use a stack. We loop through input expression and do following for every character.
1) If character is operand push that into stack
2) If character is operator pop two values from stack make them its child and push current node again.
At the end only element of stack will be root of expression tree.

Recommended: Please solve it on “PRACTICE” first, before moving on to the solution.
Below is the implementation :
	
	
	
# Python program for expression tree 
  
# An expression tree node 
class Et: 
  
    # Constructor to create a node 
    def __init__(self , value): 
        self.value = value 
        self.left = None
        self.right = None
  
# A utility function to check if 'c' 
# is an operator 
def isOperator(c): 
    if (c == '+' or c == '-' or c == '*'
        or c == '/' or c == '^'): 
        return True
    else: 
        return False
  
# A utility function to do inorder traversal 
def inorder(t): 
    if t is not None: 
        inorder(t.left) 
        print t.value, 
        inorder(t.right) 
  
# Returns root of constructed tree for 
# given postfix expression 
def constructTree(postfix): 
    stack = [] 
  
    # Traverse through every character of input expression 
    for char in postfix : 
  
        # if operand, simply push into stack 
        if not isOperator(char): 
            t = Et(char) 
            stack.append(t) 
  
        # Operator 
        else: 
  
            # Pop two top nodes 
            t = Et(char) 
            t1 = stack.pop() 
            t2 = stack.pop() 
                
            # make them children 
            t.right = t1 
            t.left = t2 
              
            # Add this subexpression to stack 
            stack.append(t) 
  
    # Only element  will be the root of expression tree 
    t = stack.pop() 
     
    return t 
  
# Driver program to test above 
postfix = "ab+ef*g*-"
r = constructTree(postfix) 
print "Infix expression is"
inorder(r)


Form words with letters
10
Anonymous User
Anonymous User
Last Edit: August 14, 2019 7:22 AM

2.3K VIEWS

Position: SDE1

Given a list of words and list of letters. Words can be formed by using letters given and each letter can only be used once. Find maximum number of letters that can be used from the given list to form the words from the word list.

Example 1:

Input: words = ["dog", "og", "cat"], letters = ['o', 'g', 'd']
Output: ["dog"]
Explanation: Using the given letters we can form ["dog"] and ["og"] but output the longer word
Example 2:

Input: words =[ "dog", "do", "go"], letters = ["d", "o", "g", "o"]
Output: ["do", "go"]
Explanation: Here we can either form ["dog"] or ["do", "go"] but pick the one which requires more letters.
Was asked to solve in O(n). But could not come up with the required solution. Any leads on how to solve this?

200. Number of Islands
Medium

3070

109

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

class Solution(object):
    def numIslands(self, grid):
        def sink(i, j):
            if 0 <= i < len(grid) and 0 <= j < len(grid[i]) and grid[i][j] == '1':
                grid[i][j] = '0'
                list(map(sink, (i+1, i-1, i, i), (j, j, j+1, j-1)))  # map in python3 return iterator
                return 1
            return 0
        return sum(sink(i, j) for i in range(len(grid)) for j in range(len(grid[i])))
        
        
        
        The key process in quickSort is partition(). Target of partitions is, given an array and an element x of array as pivot, put x at its correct position in sorted array and put all smaller elements (smaller than x) before x, and put all greater elements (greater than x) after x. 
        All this should be done in linear time.
        
  Partition Algorithm
There can be many ways to do partition, following pseudo code adopts the method given in CLRS book. The logic is simple, we start from the leftmost element and keep track of index of smaller (or equal to) elements as i. While traversing, if we find a smaller element, we swap current element with arr[i]. Otherwise we ignore current element.

/* low  --> Starting index,  high  --> Ending index */
quickSort(arr[], low, high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[pi] is now
           at right place */
        pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);  // Before pi
        quickSort(arr, pi + 1, high); // After pi
    }
}
Pseudo code for partition()

/* This function takes last element as pivot, places
   the pivot element at its correct position in sorted
    array, and places all smaller (smaller than pivot)
   to left of pivot and all greater elements to right
   of pivot */
partition (arr[], low, high)
{
    // pivot (Element to be placed at right position)
    pivot = arr[high];  
 
    i = (low - 1)  // Index of smaller element

    for (j = low; j <= high- 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j] < pivot)
        {
            i++;    // increment index of smaller element
            swap arr[i] and arr[j]
        }
    }
    swap arr[i + 1] and arr[high])
    return (i + 1)
}

Merge Sort
Like QuickSort, Merge Sort is a Divide and Conquer algorithm. It divides input array in two halves, calls itself for the two halves and then merges the two sorted halves. The merge() function is used for merging two halves. The merge(arr, l, m, r) is key process that assumes that arr[l..m] and arr[m+1..r] are sorted and merges the two sorted sub-arrays into one. See following C implementation for details.

MergeSort(arr[], l,  r)
If r > l
     1. Find the middle point to divide the array into two halves:  
             middle m = (l+r)/2
     2. Call mergeSort for first half:   
             Call mergeSort(arr, l, m)
     3. Call mergeSort for second half:
             Call mergeSort(arr, m+1, r)
     4. Merge the two halves sorted in step 2 and 3:
             Call merge(arr, l, m, r)
             
     BASIS FOR COMPARISON	QUICK SORT	MERGE SORT
The partition of elements in the array
The splitting of a array of elements is in any ratio, not necessarily divided into half.	The splitting of a array of elements is in any ratio, not necessarily divided into half.
Worst case complexity
O(n2)	O(nlogn)
Works well on
It works well on smaller array	It operates fine on any size of array
Speed of execution
It work faster than other sorting algorithms for small data set like Selection sort etc	It has a consistent speed on any size of data
Additional storage space requirement
Less(In-place)	More(not In-place)
Efficiency
Inefficient for larger arrays	More efficient
Sorting method
Internal	External
Stability
Not Stable	Stable
Preferred for
for Arrays	for Linked Lists
Locality of reference
good	poor



Basics of Hash Tables
TUTORIALPROBLEMS
Hashing is a technique that is used to uniquely identify a specific object from a group of similar objects. Some examples of how hashing is used in our lives include:

In universities, each student is assigned a unique roll number that can be used to retrieve information about them.
In libraries, each book is assigned a unique number that can be used to determine information about the book, such as its exact position in the library or the users it has been issued to etc.
In both these examples the students and books were hashed to a unique number.

Assume that you have an object and you want to assign a key to it to make searching easy. To store the key/value pair, you can use a simple array like a data structure where keys (integers) can be used directly as an index to store values. However, in cases where the keys are large and cannot be used directly as an index, you should use hashing.

In hashing, large keys are converted into small keys by using hash functions. The values are then stored in a data structure called hash table. The idea of hashing is to distribute entries (key/value pairs) uniformly across an array. Each element is assigned a key (converted key). By using that key you can access the element in O(1) time. Using the key, the algorithm (hash function) computes an index that suggests where an entry can be found or inserted.

Hashing is implemented in two steps:

An element is converted into an integer by using a hash function. This element can be used as an index to store the original element, which falls into the hash table.
The element is stored in the hash table where it can be quickly retrieved using hashed key.

hash = hashfunc(key)
index = hash % array_size

In this method, the hash is independent of the array size and it is then reduced to an index (a number between 0 and array_size − 1) by using the modulo operator (%).

Hash function
A hash function is any function that can be used to map a data set of an arbitrary size to a data set of a fixed size, which falls into the hash table. The values returned by a hash function are called hash values, hash codes, hash sums, or simply hashes.

To achieve a good hashing mechanism, It is important to have a good hash function with the following basic requirements:

Easy to compute: It should be easy to compute and must not become an algorithm in itself.

Uniform distribution: It should provide a uniform distribution across the hash table and should not result in clustering.

Less collisions: Collisions occur when pairs of elements are mapped to the same hash value. These should be avoided.

Note: Irrespective of how good a hash function is, collisions are bound to occur. Therefore, to maintain the performance of a hash table, it is important to manage collisions through various collision resolution techniques.

Need for a good hash function

Let us understand the need for a good hash function. Assume that you have to store strings in the hash table by using the hashing technique {“abcdef”, “bcdefa”, “cdefab” , “defabc” }.

To compute the index for storing the strings, use a hash function that states the following:

The index for a specific string will be equal to the sum of the ASCII values of the characters modulo 599.

As 599 is a prime number, it will reduce the possibility of indexing different strings (collisions). It is recommended that you use prime numbers in case of modulo. The ASCII values of a, b, c, d, e, and f are 97, 98, 99, 100, 101, and 102 respectively. 
Since all the strings contain the same characters with different permutations, the sum will 599.

The hash function will compute the same index for all the strings and the strings will be stored in the hash table in the following format. As the index of all the strings is the same, you can create a list on that index and insert all the strings in that list.



Minimum Window Substring
1
Anonymous User
Anonymous User
Last Edit: July 3, 2019 8:38 AM

601 VIEWS

Position: SDE2

First 10 mins about my profile, type of applications developed, why amazon..

https://leetcode.com/problems/minimum-window-substring


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
        #print(count,required.count(0))
        while right < len(s):
            while count > 0 and right < len(s):
                # move right till s[left : right] has all characters in t
                required[ord(s[right])] -= 1
                #print(required[ord(s[right])])
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
        
  Private Constructors and Singleton Classes in Java
Let’s first analyze the following question:

Can we have private constructors ?

As you can easily guess, like any method we can provide access specifier to the constructor. If it’s made private, then it can only be accessed inside the class.

Do we need such ‘private constructors ‘ ?

There are various scenarios where we can use private constructors. The major ones are

Internal Constructor chaining
Singleton class design pattern
What is a Singleton class?

As the name implies, a class is said to be singleton if it limits the number of objects of that class to one.

We can’t have more than a single object for such classes.

Singleton classes are employed extensively in concepts like Networking and Database Connectivity.

Design Pattern of Singleton classes:

The constructor of singleton class would be private so there must be another way to get the instance of that class. This problem is resolved using a class member 
instance and a factory method to return the class member.

/ Java program to demonstrate implementation of Singleton  
// pattern using private constructors. 
import java.io.*; 
  
class MySingleton 
{ 
    static MySingleton instance = null; 
    public int x = 10; 
    
    // private constructor can't be accessed outside the class 
    private MySingleton() {  } 
   
    // Factory method to provide the users with instances 
    static public MySingleton getInstance() 
    { 
        if (instance == null)         
             instance = new MySingleton(); 
   
        return instance; 
    }  
} 
  
// Driver Class 
class Main 
{ 
   public static void main(String args[])     
   { 
       MySingleton a = MySingleton.getInstance(); 
       MySingleton b = MySingleton.getInstance(); 
       a.x = a.x + 10; 
       System.out.println("Value of a.x = " + a.x); 
       System.out.println("Value of b.x = " + b.x); 
   }     
} 
Output:

Value of a.x = 20
Value of b.x = 20

136. Single Number
Easy

2808

105

Favorite

Share
Given a non-empty array of integers, every element appears twice except for one. Find that single one.

Note:

Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

Example 1:

Input: [2,2,1]
Output: 1

class Solution:
    def singleNumber(self, nums):
    	L, d = len(nums), {}
    	for n in nums:
    		if n in d: del d[n]
    		else: d[n] = 1
    	return list(d)[0]
		
		
28. Implement strStr()
Easy

1022

1476

Favorite

Share
Implement strStr().

Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

Example 1:

Input: haystack = "hello", needle = "ll"
Output: 2
Example 2:

Input: haystack = "aaaaa", needle = "bba"
Output: -1

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
        
  268. Missing Number
Easy

1080

1476

Favorite

Share
Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

Example 1:

Input: [3,0,1]
Output: 2
Example 2:

Input: [9,6,4,2,3,5,7,0,1]
Output: 8      
    class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        req_sum = len(nums)*(len(nums)+1)/2
        act_sum = sum(nums)
        return req_sum-act_sum
        
  Given A, B, C, find any string of maximum length that can be created such that no 3 consecutive characters are same. There can be at max A 'a', B 'b' and C 'c'.

Example 1:

Input: A = 1, B = 1, C = 6
Output: "ccbccacc"
Example 2:

Input: A = 1, B = 2, C = 3
Output: "acbcbc"
Related questions:
	
	class Solution(object):
    def reorganizeString(self, S):
        a = sorted(sorted(S), key=S.count)
        #print(a)
        h = len(a) / 2
        a[1::2], a[::2] = a[:h], a[h:]
        return ''.join(a) * (a[-1:] != a[-2:-1])

https://leetcode.com/problems/reorganize-string
https://leetcode.com/problems/distant-barcodes
https://leetcode.com/problems/rearrange-string-k-distance-apart (premium)
1054. Distant Barcodes
Medium

149

10

Favorite

Share
In a warehouse, there is a row of barcodes, where the i-th barcode is barcodes[i].

Rearrange the barcodes so that no two adjacent barcodes are equal.  You may return any answer, and it is guaranteed an answer exists.

 

Example 1:

Input: [1,1,1,2,2,2]
Output: [2,1,2,1,2,1]
Example 2:

Input: [1,1,1,1,2,2,3,3]
Output: [1,3,1,3,2,1,2,1]

class Solution:
    # 1 1 1 1 2 2 3 3
    # 1 . 1 . 1 . 1 .
    # . 2 . 2 . 3 . 3
    # time On
    # space On
    def rearrangeBarcodes(self, A):
        
        
        count = collections.Counter(A)
        A.sort(key=lambda a: (count[a], a))
        A[1::2], A[::2] = A[0:len(A) / 2], A[len(A) / 2:]
        return A
        
  rom collections import Counter

def fn(A, B, C):
    counter = Counter(dict(A=A, B=B, C=C))

    N = A + B + C
    ret = [None] * N

    for i in xrange(N):
        for char, freq in counter.most_common():
            if freq > 0 and ret[i-2:i+1].count(char) < 2:
                ret[i] = char
                counter[char] -= 1
                break
        else:
            print('IMPOSSIBLE')
            return

    ret = ''.join(ret)
    print(ret)
    
   Coin sum & maximum average node of N-ary tree
1
Anonymous User
Anonymous User
Last Edit: July 27, 2019 7:05 AM

546 VIEWS

Similar to Coin sum, like an unbounded knapsack.
Similar to Maximum Binary Tree, except it was to find the maximum average node excluding the last leaf node of a N-ary tree.

class Solution:
    def coinChange(self, coins, amount):
        ways = [float('inf')] * (amount + 1)
        #print(ways)
        ways[0] = 0
        
        for c in coins:
            for a in range(len(ways)):
                if c <= a:
                    
                    ways[a] = min(ways[a-c]+1, ways[a])
                    #print(min(ways[a-c]+1, ways[a]))
                    
                    
        
        if ways[amount] == float('inf'):
            return -1
        print(ways[amount])
        return ways[amount]      


322. Coin Change
Medium

2159

83

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

654. Maximum Binary Tree
Medium

1232

160

Favorite

Share
Given an integer array with no duplicates. A maximum tree building on this array is defined as follow:

The root is the maximum number in the array.
The left subtree is the maximum tree constructed from left part subarray divided by the maximum number.
The right subtree is the maximum tree constructed from right part subarray divided by the maximum number.
Construct the maximum tree by the given array and output the root node of this tree.

Example 1:
Input: [3,2,1,6,0,5]
Output: return the tree root node representing the following tree:

      6
    /   \
   3     5
    \    / 
     2  0   
 class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None

        i = nums.index(max(nums))

        node = TreeNode(nums[i])

        node.left = self.constructMaximumBinaryTree(nums[:i])
        node.right = self.constructMaximumBinaryTree(nums[i + 1:])

        return node
  Next challenges:
Maximum Binary Tree II


class Solution:
    def insertIntoMaxTree(self, root, val):
        if not root:
            return TreeNode(val)
        if val>root.val:
            t=TreeNode(val)
            t.left=root
            return t
        root.right=self.insertIntoMaxTree(root.right,val)
        return root
       
       
       
  def freeTime(schedule):
    cur = "07:00"
    res = []
    for s, e in sorted((s.zfill(5), e.zfill(5)) for s, e in schedule):
        print("cur: ",cur,"s ",s,"e ",e)
        if cur < s:
            cur = cur[1:] if cur[0] == "0" else cur
            s = s[1:] if s[0] == "0" else s
            res.append([cur, s])
        cur = e
    return res
Find Free Time for Meetup

schedule = [["16:00", "16:30"], ["6:00", "7:30"], ["8:00", "9:20"], ["8:00", "9:00"], ["17:30", "19:20"]]
print(freeTime(schedule))
schedule = [["12:00", "17:30"], ["8:00", "10:00"], ["10:00", "11:30"]]
print(freeTime(schedule))

import bisect 

First occurrence in a two-word string
0
kawrydav's avatar
kawrydav
1
Last Edit: July 18, 2019 6:43 AM

474 VIEWS

Given a string of the form "BLAHBLAHBLAHBLAHBLAH...BLAHAPPLEAPPLEAPPLEAPPLE...APPLE", i.e., consisting of a bunch of concatenated "BLAH"s followed by a bunch of concatenated "APPLE"s, find the starting index of the first occurence of "APPLE". Both "BLAH" and "APPLE" may appear zero or more times.

Examples

"BLAHBLAHBLAH" => return -1 because "APPLE" does not appear at all
"APPLEAPPLEAPPLE" => return 0 becase "APPLE" appears right at the beginning
"BLAHBLAHBLAHAPPLEAPPLEAPPLE" => return 12 because that is the first place where "APPLE" occurs
s = "BLAHBLAHBLAHAPPLEAPPLEAPPLE"
print(bisect.bisect_left(s, "APPLE") - 1)

236. Lowest Common Ancestor of a Binary Tree
Medium

2297

144

Favorite

Share
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

Given the following binary tree:  root = [3,5,1,6,2,0,8,null,null,7,4]

class Solution:
    # @param {TreeNode} root
    # @param {TreeNode} p
    # @param {TreeNode} q
    # @return {TreeNode}
    def lowestCommonAncestor(self, root, p, q):
        # escape condition
        if (not root) or (root == p) or (root == q):
            return root
        # search left and right subtree
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            # both found, root is the LCA
            return root
        return left or right



Given directory change command -

cd a/b/../c/d/e/../../

Output the visit count for each directory such as -
Root - 1
a - 2
b - 1
c - 2
d - 2
e - 1

import collections
def dirFreq(s):
	curr_path_stack = []
	commands = s.split("/")
	freq = collections.defaultdict(int)
	for c in commands:
		if c != "..":
			curr_path_stack.append(c)
		else:
			curr_path_stack.pop()
		freq[curr_path_stack[-1]] += 1
	for k,v in freq.items():
		print(k,v)
dirFreq("a/b/../c/d/e/../..")



811. Subdomain Visit Count
Easy

301

462

Favorite

Share
A website domain like "discuss.leetcode.com" consists of various subdomains. At the top level, we have "com", at the next level, we have "leetcode.com", and at the lowest level,
 "discuss.leetcode.com". When we visit a domain like "discuss.leetcode.com", we will also visit the parent domains "leetcode.com" and "com" implicitly.

Now, call a "count-paired domain" to be a count (representing the number of visits this domain received), followed by a space, followed by the address. An example of a count-paired domain might be "9001 discuss.leetcode.com".

We are given a list cpdomains of count-paired domains. We would like a list of count-paired domains, (in the same format as the input, and in any order), that explicitly counts the number of visits to each subdomain.

Example 1:
Input: 
["9001 discuss.leetcode.com"]
Output: 
["9001 discuss.leetcode.com", "9001 leetcode.com", "9001 com"]
Explanation: 
We only have one website domain: "discuss.leetcode.com". As discussed above, the subdomain "leetcode.com" and "com" will also be visited. So they will all be visited 9001 times.

Example 2:
Input: 
["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
Output: 
["901 mail.com","50 yahoo.com","900 google.mail.com","5 wiki.org","5 org","1 intel.mail.com","951 com"]
Explanation: 
We will visit "google.mail.com" 900 times, "yahoo.com" 50 times, "intel.mail.com" once and "wiki.org" 5 times. For the subdomains, we will visit "mail.com" 900 + 1 = 901 times, "com" 900 + 50 + 1 = 951 times, and "org" 5 times.


from collections import defaultdict

class Solution:
    def subdomainVisits(self, cpdomains):
        counter = defaultdict(int)
        
        for string in cpdomains:
            count, cpdomain = string.split()
            count = int(count)
            
            while True:
                counter[cpdomain] += count
                next_dot = cpdomain.find('.')
                if next_dot < 0:
                    break
                cpdomain = cpdomain[cpdomain.find('.')+1:]
        
        result = []
        for key, value in counter.items():
            result.append("{} {}".format(value, key))
        
        return result
        
 Next challenges:
Sudoku Solver
Maximum Size Subarray Sum Equals k
Subarrays with K Different Integers


Type of array and its maximum element
Given an array, it can be of 4 types
(a) Ascending
(b) Descending
(c) Ascending Rotated
(d) Descending Rotated
Find out which kind of array it is and return the maximum of that array.

Examples :

Input :  arr[] = { 2, 1, 5, 4, 3}
Output : Descending rotated with maximum element 5

Input :  arr[] = { 3, 4, 5, 1, 2}
Output : Ascending rotated with maximum element 5


# Python3 program to find type of array, ascending  
# descending, clockwise rotated or anti-clockwise  
# rotated. 
  
def findType(arr, n) : 
  
    i = 0;  
  
    # Check if the array is in ascending order  
    while (i < n-1 and arr[i] <= arr[i + 1]) : 
        i = i + 1
  
    # If i reaches to last index that means  
    # all elements are in increasing order  
    if (i == n-1): 
        print(("Ascending with maximum element = "
              + str(arr[n-1]))) 
        return None
      
  
    # If first element is greater than next one  
    if (i == 0): 
        while (i < n-1 and arr[i] >= arr[i + 1]): 
            i = i + 1;  
  
        # If i reaches to last index  
        if (i == n - 1): 
            print(("Descending with maximum element = "
                  + str(arr[0]))) 
            return None
      
  
        # If the whole array is not in decreasing order  
        # that means it is first decreasing then  
        # increasing, i.e., descending rotated, so  
        # its maximum element will be the point breaking  
        # the order i.e. i so, max will be i + 1  
        if (arr[0] < arr[i + 1]): 
            print(("Descending rotated with maximum element = " 
                  + str(max(arr[0], arr[i + 1])))) 
            return None
        else: 
          
            print(("Ascending rotated with maximum element = " 
                  + str(max(arr[0], arr[i + 1])))) 
                    
            return None
          
      
  
    # If whole array is not increasing that means at some  
    # point it is decreasing, which makes it ascending rotated  
    # with max element as the decreasing point  
    if (i < n -1 and arr[0] > arr[i + 1]): 
      
        print(("Ascending rotated with maximum element = " 
             + str(max(arr[i], arr[0])))) 
        return None
      
  
    print(("Descending rotated with maximum element "
          + str(max(arr[i], arr[0])))) 
  
# Driver code  
if __name__=='__main__': 
    arr1 = [ 4, 5, 6, 1, 2, 3] # Ascending rotated  
    n = len(arr1) 
    findType(arr1, n);  
  
    arr2 = [ 2, 1, 7, 5, 4, 3] # Descending rotated  
    n = len(arr2)  
    findType(arr2, n);  
  
    arr3 = [ 1, 2, 3, 4, 5, 8] # Ascending  
    n = len(arr3)  
    findType(arr3, n);  
  
    arr4 = [ 9, 5, 4, 3, 2, 1] # Descending  
    n = len(arr4)  
    findType(arr4, n);  
  
# this code is contributed by YatinGupta 


Given a list of names (LastName, FirstName) and an input param ("xyz"), return a list of all the names 
where either the first or last name start with that entry parm.



208. Implement Trie (Prefix Tree)
Medium

1826

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


class Trie(object):

	def __init__(self):
		self.trie = {}


	def insert(self, word):
		t = self.trie
		for c in word:
			if c not in t: t[c] = {}
			t = t[c]
		t["-"] = True


	def search(self, word):
		t = self.trie
		for c in word:
			if c not in t: return False
			t = t[c]
		return "-" in t

	def startsWith(self, prefix):
		t = self.trie
		for c in prefix:
			if c not in t: return False
			t = t[c]
		return True
      
      
 ell me about a time you handled pressure.
Tell me about a time you had a conflict with your manager.
Tell me about a time you had to work with a difficult coworker.
      
 The universe of the Game of Life is an infinite, two-dimensional orthogonal grid of square cells, each of which is in one of two possible states, alive or dead, (or populated and unpopulated, respectively). Every cell interacts with its eight neighbours, which are the cells that are horizontally, vertically, or diagonally adjacent. At each step in time, the following transitions occur:

Any live cell with fewer than two live neighbours dies, as if by underpopulation.
Any live cell with two or three live neighbours lives on to the next generation.
Any live cell with more than three live neighbours dies, as if by overpopulation.
Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
The initial pattern constitutes the seed of the system. The first generation is created by applying the above rules simultaneously to every cell in the seed; births and deaths occur simultaneously, and the discrete moment at which this happens is sometimes called a tick. Each generation is a pure function of the preceding one. 
The rules continue to be applied repeatedly to create further generations.

def gameOfLife(self, board):
    board[:] = [[int(3 in (count, count - live))
                 for j, live in enumerate(row)
                 for count in [sum(sum(row[j-(j>0):j+2])
                                   for row in board[i-(i>0):i+2])]]
                for i, row in enumerate(board)]
                
                
 
 
 
 Print Longest substring without repeating characters
Given a string, print the longest substring without repeating characters. For example, the longest substrings without repeating characters for “ABDEFGABEF” are “BDEFGA” and “DEFGAB”, with length 6. For “BBBB” the longest substring is “B”, with length 1. The desired time complexity is O(n) where n is the length of the string.

Prerequisite: Length of longest substring without repeating characters

Examples:

Input : GEEKSFORGEEKS
Output : EKSFORG

Input : ABDEFGABEF
Output : BDEFGA               
  # Python3 program to find and print longest  
# substring without repeating characters.  
  
# Function to find and print longest  
# substring without repeating characters.  
def findLongestSubstring(string): 
  
    n = len(string)  
  
    # starting point of current substring.  
    st = 0
  
    # maximum length substring without  
    # repeating characters.  
    maxlen = 0
  
    # starting index of maximum  
    # length substring.  
    start = 0
  
    # Hash Map to store last occurrence  
    # of each already visited character.  
    pos = {}  
  
    # Last occurrence of first 
    # character is index 0  
    pos[string[0]] = 0
  
    for i in range(1, n):  
  
        # If this character is not present in hash,  
        # then this is first occurrence of this  
        # character, store this in hash.  
        if string[i] not in pos:  
            pos[string[i]] = i  
  
        else: 
            # If this character is present in hash then  
            # this character has previous occurrence,  
            # check if that occurrence is before or after  
            # starting point of current substring.  
            if pos[string[i]] >= st:  
  
                # find length of current substring and  
                # update maxlen and start accordingly.  
                currlen = i - st  
                if maxlen < currlen:  
                    maxlen = currlen  
                    start = st  
  
                # Next substring will start after the last  
                # occurrence of current character to avoid  
                # its repetition.  
                st = pos[string[i]] + 1
              
            # Update last occurrence of  
            # current character.  
            pos[string[i]] = i  
          
    # Compare length of last substring with maxlen  
    # and update maxlen and start accordingly.  
    if maxlen < i - st:  
        maxlen = i - st  
        start = st  
      
    # The required longest substring without  
    # repeating characters is from string[start]  
    # to string[start+maxlen-1].  
    return string[start : start + maxlen]  
  
# Driver Code 
if __name__ == "__main__":  
  
    string = "GEEKSFORGEEKS"
    print(findLongestSubstring(string))  
  
# This code is contributed by Rituraj Jain 


3. Longest Substring Without Repeating Characters


class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        
        """
        if len(s)==0:
            return 0
        temp=s[0]
        max_len=1
        for c in s[1:]:
            if c in temp:
                i=temp.find(c)
                temp=temp[i+1:]
            temp+=c
            print(temp)
            if len(temp)>max_len:
                max_len=len(temp)
        return max_len
        
        
        https://www.programcreek.com/2013/02/longest-substring-which-contains-2-unique-characters/
    395. Longest Substring with At Least K Repeating Characters
Medium

833

77

Favorite

Share
Find the length of the longest substring T of a given string (consists of lowercase letters only) such that every character in T appears no less than k times.

Example 1:

Input:
s = "aaabb", k = 3

Output:
3

The longest substring is "aaa", as 'a' is repeated 3 times.
Example 2:

Input:
s = "ababbc", k = 2

Output:
5    
        
        from collections import defaultdict
class Solution(object):
    def longestSubstring(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        def __longestSubstring(s, k):
            if len(s) < k:
                return 0
            stat = defaultdict(int)
            milestone = defaultdict(list)

            for ind, char in enumerate(s):
                stat[char]  += 1
                milestone[char].append(ind)

            foot = []

            for key, val in stat.iteritems():
                if val < k:
                    foot.extend(milestone[key])
                
            if len(foot) == 0:
                return len(s)
            else:
                foot.sort()
                foot.insert(0, -1)
                foot.append(len(s))
                res = 0
                for i in range(len(foot)-1):
                    tmp = __longestSubstring(s[foot[i]+1:foot[i+1]], k)
                    if res < tmp:
                        res = tmp
                return res
            
        return __longestSubstring(s, k)
        
        
        Find the longest substring with k unique characters in a given string
Given a string you need to print longest possible substring that has exactly M unique characters. If there are more than one substring of longest possible length, then print any one of them.
Examples:

"aabbcc", k = 1
Max substring can be any one from {"aa" , "bb" , "cc"}.

"aabbcc", k = 2
Max substring can be any one from {"aabb" , "bbcc"}.

"aabbcc", k = 3
There are substrings with exactly 3 unique characters
{"aabbcc" , "abbcc" , "aabbc" , "abbc" }
Max is "aabbcc" with length 6.

"aaabbb", k = 3
There are only two unique characters, thus show error message. 
Source: Google Interview Question.


# Python program to find the longest substring with k unique 
# characters in a given string 
MAX_CHARS = 26
  
# This function calculates number of unique characters 
# using a associative array count[]. Returns true if 
# no. of characters are less than required else returns 
# false. 
def isValid(count, k): 
    val = 0
    for i in xrange(MAX_CHARS): 
        if count[i] > 0: 
            val += 1
  
    # Return true if k is greater than or equal to val 
    return (k >= val) 
  
# Finds the maximum substring with exactly k unique characters 
def kUniques(s, k): 
    u = 0    # number of unique characters 
    n = len(s) 
  
    # Associative array to store the count 
    count = [0] * MAX_CHARS 
  
    # Tranverse the string, fills the associative array 
    # count[] and count number of unique characters 
    for i in xrange(n): 
        if count[ord(s[i])-ord('a')] == 0: 
            u += 1
        count[ord(s[i])-ord('a')] += 1
  
    # If there are not enough unique characters, show 
    # an error message. 
    if u < k: 
        print "Not enough unique characters"
        return
  
    # Otherwise take a window with first element in it. 
    # start and end variables. 
    curr_start = 0
    curr_end = 0
  
    # Also initialize values for result longest window 
    max_window_size = 1
    max_window_start = 0
  
    # Initialize associative array count[] with zero 
    count = [0] * len(count) 
  
    count[ord(s[0])-ord('a')] += 1    # put the first character 
  
    # Start from the second character and add 
    # characters in window according to above 
    # explanation 
    for i in xrange(1,n): 
        # Add the character 's[i]' to current window 
        count[ord(s[i])-ord('a')] += 1
        curr_end+=1
  
        # If there are more than k unique characters in 
        # current window, remove from left side 
        while not isValid(count, k): 
            count[ord(s[curr_start])-ord('a')] -= 1
            curr_start += 1
  
        # Update the max window size if required 
        if curr_end-curr_start+1 > max_window_size: 
            max_window_size = curr_end-curr_start+1
            max_window_start = curr_start 
  
    print "Max substring is : " + s[max_window_start:] \ 
            + " with length " + str(max_window_size) 
  
# Driver function 
s = "aabacbebebe"
k = 3
kUniques(s, k) 
  
# This code is contributed by BHAVYA JAIN 

Output:
Max sustring is : cbebebe with length 7
Time Complexity: Considering function “isValid()” takes constant time, time complexity of above solution is O(n)

Treasure Island

You have a map that marks the location of a treasure island. Some of the map area has jagged rocks and dangerous reefs. Other areas are safe to sail in. There are other explorers trying to find the treasure. So you must figure out a shortest route to the treasure island.

Assume the map area is a two dimensional grid, represented by a matrix of characters. You must start from the top-left corner of the map and can move one block up, down, left or right at a time. The treasure island is marked as X in a block of the matrix. X will not be at the top-left corner. Any block with dangerous rocks or reefs will be marked as D. You must not enter dangerous blocks. You cannot leave the map area. Other areas O are safe to sail in. The top-left corner is always safe. Output the minimum number of steps to get to the treasure.

Example:

Input:
[['O', 'O', 'O', 'O'],
 ['D', 'O', 'D', 'O'],
 ['O', 'O', 'O', 'O'],
 ['X', 'D', 'D', 'O']]

Output: 5
Explanation: Route is (0, 0), (0, 1), (1, 1), (2, 1), (2, 0), (3, 0) The minimum route takes 5 steps.

def find_treasure(t_map, row, col, curr_steps, min_steps):
    if row >= len(t_map) or row < 0 or col >= len(t_map[0]) or col < 0 or t_map[row][col] == 'D' or t_map[row][col] == '#':
        return None, min_steps

    if t_map[row][col] == 'X':
        curr_steps += 1
        if min_steps > curr_steps:
            min_steps = min(curr_steps, min_steps)

        return None, min_steps

    else:
        tmp = t_map[row][col]
        t_map[row][col] = '#'
        curr_steps += 1
        left = find_treasure(t_map, row, col-1, curr_steps, min_steps)
        right = find_treasure(t_map, row, col+1, curr_steps, min_steps)
        up = find_treasure(t_map, row-1, col, curr_steps, min_steps)
        down = find_treasure(t_map, row+1, col, curr_steps, min_steps)

        t_map[row][col] = tmp
        print(t_map,[row][col],left,right,up,down)

        return curr_steps, min(left[1], right[1], up[1], down[1])


if __name__ == '__main__':
 #    treasure_map = [['O', 'O', 'O', 'O'],
 # ['D', 'O', 'D', 'O'],
 # ['O', 'O', 'O', 'O'],
 # ['X', 'D', 'D', 'O']]
    treasure_map = [['O', 'O', 'O', '0'],
 ['D', 'O', 'D', 'O'],
 ['O', 'O', 'O', 'O'],
 ['X', 'D', 'D', 'D']]

    res = find_treasure(treasure_map, 0, 0, -1, float('inf'))
    print("Result: ", res[1])

Subtree with Maximum Average
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        this.val = val
        this.left, this.right = None, None
"""
class Solution:
    # @param {TreeNode} root the root of binary tree
    # @return {TreeNode} the root of the maximum average of subtree
    avg, node = 0, None
    def findSubtree2(self, root):
        # Write your code here
        if root is None:
            return None
        self.getSubtree(root)
        return self.node;
        
    def getSubtree(self, root):
        if root is None:
            return 0, 0
            
        sumLeft, countLeft = self.getSubtree(root.left)
        sumRight, countRight = self.getSubtree(root.right)
        
        sumRoot = sumLeft + sumRight + root.val
        countRoot = countLeft + countRight + 1
        average = sumRoot * 1.0 / countRoot 
        if self.node is None or average > self.avg:
            self.avg = average
            self.node = root
        return sumRoot, countRoot



def maxAppeal(arr,n):

    maxVal = -float('inf')
    currval=0
    a , b = 0 , 0 # result index

    # since left and right could be maxval when they are farthest,
    # merge it to middle
    left = 0
    right = n-1
    while (left <= right):
        curIndex = [left, right]
        if (arr[left] < arr[right]):
            currval = arr[left]+ arr[right] + abs(right-left)
            left += 1
        else:
            # arr[right] is smaller
            currval = arr[right]+ arr[left] + abs(right-left)
            right -= 1
        if currval > maxVal:
            a , b = curIndex[0] , curIndex[1]
        maxVal = max(maxVal, currval)

    return [a, b]


arr = [1, 2, 9, 4, 7, 1, 6] #[8, 1, 9, 4]
n = len(arr)

print(maxAppeal(arr,n), '[2,6]') 

arr2 = [6, 2, 100, 4, 7, 1, 6] #[8, 1, 9, 4]
n = len(arr2)
print(maxAppeal(arr2,n), '[2,2]') 

arr2 =[1, 6, 1, 1, 1, 1, 7] #[16, 2, 17, 4, 7, 1, 16] #[8, 1, 9, 4]
n = len(arr2)
print(maxAppeal(arr2,n), '[0,6]') 



Two Sum
Easy

11768

406

Favorite

Share
Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].

class Solution(): 
    def twoSum(self, nums, target):
        lookup = {nums[i]:i for i in range(len(nums))}
        for i in range(len(nums)):
            complement = target - nums[i]
            j = lookup.get(complement)#hash table to search 
            if j != None and j != i: 
                return [i, j]
        return [] 


def minimumCost(N: int, roads: List[List[int]], repairs: List[List[int]]) -> int:
  cost_map = {}
  for c1, c2, c in repairs:
    cost_map[(c1, c2)] = c
    
  for edge in roads:
    c1, c2 = edge
    if (c1, c2) not in cost_map:
      cost_map[(c1, c2)] = 0
    
  connections = []
  for key in cost_map:
    c1, c2 = key
    connections.append([c1, c2, cost_map[key]])
    
    
  if len(connections) < N - 1:
    return -1
  if N is 1:
    return 0

  connections = sorted(connections, key=lambda x: x[2])
  total_cost = 0

  parent = {}

  def findParent(city):
    parent.setdefault(city, city)
    if parent[city] == city:
      return city
    else:
      return findParent(parent[city])

  def mergeSets(city1, city2):
    parent1, parent2 = findParent(city1), findParent(city2)
    if parent1 != parent2:
      parent[parent1] = parent2
      return cities - 1, total_cost + cost
    return cities, total_cost

  cities = N - 1
  for city1, city2, cost in connections:
    cities, total_cost = mergeSets(city1, city2)
          
  return total_cost if cities == 0 else -1

print(minimumCost(N = 5, roads = [[1, 2], [2, 3], [3, 4], [4, 5], [1, 5]], repairs = [[1, 2, 12], [3, 4, 30], [1, 5, 8]]))
print(minimumCost(6, [[1, 2], [2, 3], [4, 5], [3, 5], [1, 6], [2, 4]], [[1, 6, 410], [2, 4, 800]]))
print(minimumCost(N = 6, roads = [[1, 2], [2, 3], [4, 5], [5, 6], [1, 5], [2, 4], [3, 4]], repairs = [[1, 5, 110], [2, 4, 84], [3, 4, 79]]))

 def minCostForRepair(self, n, edges, edgesToRepair):
        graph=defaultdict(list)
        addedEdges=set()
        for edge in edgesToRepair:
            graph[edge[0]].append((edge[2], edge[1]))
            graph[edge[1]].append((edge[2], edge[0]))
            addedEdges.add((edge[0], edge[1]))
            addedEdges.add((edge[1], edge[0]))
        for edge in edges:
            if tuple(edge) not in addedEdges:
                graph[edge[0]].append((0, edge[1]))
                graph[edge[1]].append((0, edge[0]))

        res=0
        priorityQueue=[(0, 1)]
        heapq.heapify(priorityQueue)
        visited=set()

        while priorityQueue:
            minCost, node=heapq.heappop(priorityQueue)
            if node not in visited:
                visited.add(node)
                res+=minCost
                for cost, connectedNode in graph[node]:
                    if connectedNode not in visited:
                        heapq.heappush(priorityQueue, (cost, connectedNode))

        return res


s=Solution()

print(s.minCostForRepair(5, [[1, 2], [2, 3], [3, 4], [4, 5], [1, 5]], [[1, 2, 12], [3, 4, 30], [1, 5, 8]]))
print(s.minCostForRepair(6, [[1, 2], [2, 3], [4, 5], [3, 5], [1, 6], [2, 4]], [[1, 6, 410], [2, 4, 800]]))
print(s.minCostForRepair(6, [[1, 2], [2, 3], [4, 5], [5, 6], [1, 5], [2, 4], [3, 4]], [[1, 5, 110], [2, 4, 84], [3, 4, 79]]))

https://leetcode.com/discuss/interview-question/357310


71. Simplify Path
Medium

494

1315

Favorite

Share
Given an absolute path for a file (Unix-style), simplify it. Or in other words, convert it to the canonical path.

In a UNIX-style file system, a period . refers to the current directory. Furthermore, a double period .. moves the directory up a level. For more information, see: Absolute path vs relative path in Linux/Unix

Note that the returned canonical path must always begin with a slash /, and there must be only a single slash / between two directory names. The last directory name (if it exists) must not end with a trailing /. Also, the canonical path must be the shortest string representing the absolute path.

 class Solution(object):
    def simplifyPath(self, path):
        ls = path.split("/")
        ln = len(path)
        if ln == 0:
            return ""
        st = []
        for item in ls:
            if item == "" or item == ".":
                continue
            if item == "..":
                if st:
                    st.pop(-1)
            else:
                st.append(item)


        return "/" + "/".join(st)
        
        
   75. Sort Colors
Medium

1928

170

Favorite

Share
Given an array with n objects colored red, white or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

Note: You are not suppose to use the library's sort function for this problem.

Example:

Input: [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]


class Solution(object):
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

Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.

Example 1:
Given tree s:

     3
    / \
   4   5
  / \
 1   2
Given tree t:
   4 
  / \
 1   2
Return true, because t has the same structure and node values with a subtree of s.
Example 2:
Given tree s:

     3
    / \
   4   5
  / \
 1   2
    /
   0
Given tree t:
   4
  / \
 1   2
 
 
 class Solution:
    def isSubtree(self, s,t):
        if not s: 
            return False
        if self.isSameTree(s, t): 
            return True
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

    def isSameTree(self, p,q):
        if p and q:
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return p is q

   Substrings of size K with K distinct chars
0
Sithis's avatar
Sithis
Moderator
3270
Last Edit: August 29, 2019 4:28 PM

620 VIEWS

Given a string s and an int k, return all unique substrings of s of size k with k distinct characters.

Example 1:

Input: s = "abcabc", k = 3
Output: ["abc", "bca", "cab"]
Example 2:

Input: s = "abacab", k = 3
Output: ["bac", "cab"]
Example 3:

Input: s = "awaglknagawunagwkwagl", k = 4
Output: ["wagl", "aglk", "glkn", "lkna", "knag", "gawu", "awun", "wuna", "unag", "nagw", "agwk", "kwag"]
Explanation: 
Substrings in order are: "wagl", "aglk", "glkn", "lkna", "knag", "gawu", "awun", "wuna", "unag", "nagw", "agwk", "kwag", "wagl" 
"wagl" is repeated twice, but is included in the output once.
Constraints:
	
	
def kSubstring(s, k):
  if not s or k == 0:
    return []
  seen = set()
  res = []
  for i in range(len(s) - k+1):
    sub = s[i:i+k]
    if uniWindow(sub):
      if sub not in seen:
        seen.add(sub)
        res.append(sub)
  return res

def uniWindow(sub):
  c = set()
  print("c ",c)
  for i in sub:
    if i in c:
      return False
    else:
      c.add(i)
  return True



s1 = "awaglknagawunagwkwagl"
k1 = 4
print(kSubstring(s1,k1))

s2 = "abcabc"
k2 = 3
print(kSubstring(s2,k2))

s3 = "abacab"
k3 = 3
print(kSubstring(s3,k3))

Check if two given strings are isomorphic to each other
Two strings str1 and str2 are called isomorphic if there is a one to one mapping possible for every character of str1 to every character of str2. And all occurrences of every character in ‘str1’ map to same character in ‘str2’

Examples:

Input:  str1 = "aab", str2 = "xxy"
Output: True
'a' is mapped to 'x' and 'b' is mapped to 'y'.

Input:  str1 = "aab", str2 = "xyz"
Output: False
One occurrence of 'a' in str1 has 'x' in str2 and 
other occurrence of 'a' has 'y'.

1) If lengths of str1 and str2 are not same, return false.
2) Do following for every character in str1 and str2
   a) If this character is seen first time in str1, 
      then current of str2 must have not appeared before.
      (i) If current character of str2 is seen, return false.
          Mark current character of str2 as visited.
      (ii) Store mapping of current characters.
   b) Else check if previous occurrence of str1[i] mapped
      to same character.
     
# Python program to check if two strings are isomorphic 
MAX_CHARS = 256
  
# This function returns true if str1 and str2 are isomorphic 
def areIsomorphic(string1, string2): 
    m = len(string1) 
    n = len(string2) 
  
    # Length of both strings must be same for one to one 
    # corresponance 
    if m != n: 
        return False
  
    # To mark visited characters in str2 
    marked = [False] * MAX_CHARS 
  
    # To store mapping of every character from str1 to 
    # that of str2. Initialize all entries of map as -1 
    map = [-1] * MAX_CHARS 
  
    # Process all characters one by one 
    for i in xrange(n): 
  
        # if current character of str1 is seen first 
        # time in it. 
        if map[ord(string1[i])] == -1: 
  
            # if current character of st2 is already 
            # seen, one to one mapping not possible 
            if marked[ord(string2[i])] == True: 
                return False
  
            # Mark current character of str2 as visited 
            marked[ord(string2[i])] = True
  
            # Store mapping of current characters 
            map[ord(string1[i])] = string2[i] 
  
        # If this is not first appearance of current 
        # character in str1, then check if previous 
        # appearance mapped to same character of str2 
        elif map[ord(string1[i])] != string2[i]: 
            return False
  
    return True
  
# Driver program 
print areIsomorphic("aab","xxy") 
print areIsomorphic("aab","xyz") 
# This code is contributed by Bhavya Jain 


Count Number of Squares in a grid

# Python 3 program to find nth term  
# divisible by a or b  
import sys 
  
# Function to return gcd of a and b  
def gcd(a, b): 
    if a == 0: 
        return b 
    return gcd(b % a, a) 
  
# Function to calculate how many numbers  
# from 1 to num are divisible by a or b  
def divTermCount(a, b, lcm, num): 
  
    # calculate number of terms divisible  
    # by a and by b then, remove the terms  
    # which are divisible by both a and b  
    return num // a + num // b - num // lcm 
  
# Binary search to find the nth term  
# divisible by a or b  
def findNthTerm(a, b, n): 
  
    # set low to 1 and high to max(a, b)*n,  
    # here we have taken high as 10^18  
    low = 1; high = sys.maxsize 
    lcm = (a * b) // gcd(a, b) 
    while low < high: 
        mid = low + (high - low) // 2
  
        # if the current term is less  
        # than n then we need to increase   
        # low to mid + 1  
        if divTermCount(a, b, lcm, mid) < n: 
            low = mid + 1
  
        # if current term is greater  
        # than equal to n then high = mid  
        else: 
            high = mid 
    return low 
  
# Driver code 
a = 2; b = 5; n = 10
print(findNthTerm(a, b, n)) 
  
# This code is contributed by Shrikant13 

Find minimum difference between any two elements
Given an unsorted array, find the minimum difference between any pair in given array.

Examples :

Input  : {1, 5, 3, 19, 18, 25};
Output : 1
Minimum difference is between 18 and 19

Input  : {30, 5, 20, 9};

# Returns minimum difference between any pair 
def findMinDiff(arr, n): 
  
    # Sort array in non-decreasing order 
    arr = sorted(arr) 
  
    # Initialize difference as infinite 
    diff = 10**20
  
    # Find the min diff by comparing adjacent 
    # pairs in sorted array 
    for i in range(n-1): 
        if arr[i+1] - arr[i] < diff: 
            diff = arr[i+1] - arr[i] 
  
    # Return min diff 
    return diff 
  
# Driver code 
arr = [1, 5, 3, 19, 18, 25] 
n = len(arr) 
print("Minimum difference is " + str(findMinDiff(arr, n))) 
 
 
 Array with XOR equal X
1
Sithis's avatar
Sithis
Moderator
3287
Last Edit: a day ago

187 VIEWS

You are given two natural numbers n and x. You are required to create an array of n natural numbers such that the bitwise XOR of the numbers is equal to x. The sum of all the natural numbers that are available in the array is as minimum as possible. If their exists multiple arrays, then output the one that is the smallest among them.

Example 1:

Input: n = 3, x = 2
Output: [1, 1, 2]
Constraints:

1<= n <=10^5
1<= x <=10^18
online assessment 
# This code is contributed by Pratik Chhajer 
def xor( n, x):
        res = []
        if n % 2 == 0:
            return res
    
        i = 1
        cnt = 0
        while cnt < n:
            if cnt == n-1: break
            res.append(i)
            res.append(i)
            cnt += 2
        res.append(x)
        val = res[0]
        for i in range(1, len(res)):
            val = val ^ res[i]
        print('val is ', val)
    
        return res
n = 3
x = 2
print(xor(n, x))

63. Unique Paths II
Medium

988
63. Unique Paths II
Medium

988

165

Favorite

Share
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

Now consider if some obstacles are added to the grids. How many unique paths would there be?


class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        m=len(obstacleGrid)    
        n=len(obstacleGrid[0])
        dp= [[0]*n]*m
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j]==1: dp[i][j]=0
                elif i==0 and j==0: dp[i][j]=1 
                elif i==0: dp[i][j]= dp[i][j-1]
                elif j==0: dp[i][j]= dp[i-1][j]
                else : dp[i][j]= dp[i-1][j]+ dp[i][j-1]
        return dp[m-1][n-1]
