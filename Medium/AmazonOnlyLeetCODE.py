763. Partition Labels
Medium

1126

59

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
Note:

S will have length in range [1, 500].
S will consist of lowercase letters ('a' to 'z') only.

class Solution(object):
    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        
        counter = collections.Counter(S)
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
        
138. Copy List with Random Pointer
Medium

1750

464

Favorite

Share
A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.

Return a deep copy of the list.

 

Example 1:



Input:
{"$id":"1","next":{"$id":"2","next":null,"random":{"$ref":"2"},"val":2},"random":{"$ref":"2"},"val":1}

Explanation:
Node 1's value is 1, both of its next and random pointer points to Node 2.
Node 2's value is 2, its next pointer points to null and its random pointer points to itself.

# Definition for a Node.
class Node(object):
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random


class Solution:
    def copyRandomList(self, head):
        prev = None
        temp = head
        while temp:
            temp.copy = ListNode(temp.val)
            if prev:
                prev.next = temp.copy
            prev = temp.copy
            temp = temp.next
        
        temp = head
        while temp:
            temp.copy.random = temp.random and temp.random.copy
            temp = temp.next

        return head and head.copy
        
        
  236. Lowest Common Ancestor of a Binary Tree
Medium

2189

141

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
        
5. Longest Palindromic Substring
Medium

4072

376

Favorite

Share
Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

Example 1:

Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.
Example 2:

Input: "cbbd"
Output: "bb"
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
        
 238. Product of Array Except Self
Medium

2537

219

Favorite

Share
Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

Example:

Input:  [1,2,3,4]
Output: [24,12,8,6]
Note: Please solve it without division and in O(n).

Follow up:
Could you solve it with constant space complexity? (The output array does not count as extra space for the purpose of space complexity analysis.)


class Solution:
    # @param {integer[]} nums
    # @return {integer[]}
    def productExceptSelf(self, nums):
        p = 1
        n = len(nums)
        output = []
        for i in range(0,n):
            output.append(p)
            p = p * nums[i]
        p = 1
        for i in range(n-1,-1,-1):
            output[i] = output[i] * p
            p = p * nums[i]
        return output
 
 
 Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.


The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped. Thanks Marcos for contributing this image!

Example:

Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

      
 class Solution(object):
    def trap(self, bars):
        if not bars or len(bars) < 3:
            return 0
        volume = 0
        left, right = 0, len(bars) - 1
        l_max, r_max = bars[left], bars[right]
        while left < right:
            l_max, r_max = max(bars[left], l_max), max(bars[right], r_max)
            if l_max <= r_max:
                volume += l_max - bars[left]
                left += 1
            else:
                volume += r_max - bars[right]
                right -= 1
        return volume



121. Best Time to Buy and Sell Stock
Easy

2930

137

Favorite

Share
Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.

Example 1:

Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
Example 2:

Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.


646. Maximum Length of Pair Chain
Medium

601

56

Favorite

Share
You are given n pairs of numbers. In every pair, the first number is always smaller than the second number.

Now, we define a pair (c, d) can follow another pair (a, b) if and only if b < c. Chain of pairs can be formed in this fashion.

Given a set of pairs, find the length longest chain which can be formed. You needn't use up all the given pairs. You can select pairs in any order.

Example 1:
Input: [[1,2], [2,3], [3,4]]
Output: 2
Explanation: The longest chain is [1,2] -> [3,4]


class Solution(object):
    def findLongestChain(self, pairs):
        pairs = sorted(pairs, key=lambda x: (x[1], x[0]))

        c, res = float('-inf'), 0
        for pair in pairs:
            print (pair)
            if c < pair[0]:
                c, res = pair[1], res+1

        return res
        
  or 
  
  class Solution(object):
    def findLongestChain(self, pairs):
        pairs.sort()
        mx,N=1,len(pairs)
        dp=[1]*N
        for i in range(N):
            for j in range(i):
                if pairs[i][0]>pairs[j][1]:
                    dp[i] = max(dp[i],dp[j]+1)
        
        for i in range(N):
            mx = max(mx,dp[i])
        
        return mx


127. Word Ladder
Medium

1648

863

Favorite

Share
Given two words (beginWord and endWord), and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord, such that:

Only one letter can be changed at a time.
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
Note:

Return 0 if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.
Example 1:

Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output: 5

Explanation: As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
return its length 5.
Example 2:

Input:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

Output: 0

Explanation: The endWord "cog" is not in wordList, therefore no possible transformation.

class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        wordList = set(wordList)
        if endWord not in wordList:
            return 0
        # BFS visit
        curr_level = {beginWord}
        dist = 1
        while curr_level:
            wordList -= curr_level
            next_level = set()
            for word in curr_level:
                for i in range(len(word)):
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        new_word = word[:i] + c + word[i+1:]
                        if new_word == endWord:
                            return 1 + dist
                        if new_word in wordList:
                            next_level.add(new_word)
            curr_level = next_level
            dist += 1
        return 0
387. First Unique Character in a String
Easy

1126

84

Favorite

Share
Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.

Examples:

s = "leetcode"
return 0.

s = "loveleetcode",
return 2.

from collections import Counter

class Solution:
    def firstUniqChar(self, s):
    	for i,j in Counter(s).items(): 
    		if j == 1: return(s.index(i))
    	return -1
    	
 234. Palindrome Linked List
Easy

1842

268

Favorite

Share
Given a singly linked list, determine if it is a palindrome.

Example 1:

Input: 1->2
Output: false
Example 2:

Input: 1->2->2->1
Output: true
Follow up:
Could you do it in O(n) time and O(1) space?

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def isPalindrome(self, head):

        slow = fast = head
        prev_node = None
        while fast and fast.next:       
            fast = fast.next.next
            head = head.next
        
           # reverse first half
            next_node = slow.next
            slow.next = prev_node
            prev_node = slow
            slow = next_node
            
        if fast:
            head = head.next
        
        # compare the (first reversed half) and the (second half)    
        while prev_node and head:
            if prev_node.val!=head.val:
                return False
            
            prev_node = prev_node.next
            head = head.next
        
        return True
        
        
 819. Most Common Word
Easy

343

825

Favorite

Share
Given a paragraph and a list of banned words, return the most frequent word that is not in the list of banned words.  It is guaranteed there is at least one word that isn't banned, and that the answer is unique.

Words in the list of banned words are given in lowercase, and free of punctuation.  Words in the paragraph are not case sensitive.  The answer is in lowercase.

 

Example:

Input: 
paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
banned = ["hit"]
Output: "ball"
Explanation: 
"hit" occurs 3 times, but it is a banned word.
"ball" occurs twice (and no other word does), so it is the most frequent non-banned word in the paragraph. 
Note that words in the paragraph are not case sensitive,
that punctuation is ignored (even if adjacent to words, such as "ball,"), 
and that "hit" isn't the answer even though it occurs more because it is banned.
 

Note:

1 <= paragraph.length <= 1000.
0 <= banned.length <= 100.
1 <= banned[i].length <= 10.
The answer is unique, and written in lowercase (even if its occurrences in paragraph may have uppercase symbols, and even if it is a proper noun.)
paragraph only consists of letters, spaces, or the punctuation symbols !?',;.
There are no hyphens or hyphenated words.
Words only consist of letters, never apostrophes or other punctuation symbols.

class Solution(object):

    def mostCommonWord(self, paragraph, banned):
        words = re.findall("\w+", paragraph.lower())
        print(words)
        map = {}
        for word in words:
            if word not in banned:
                if word in map:
                    map[word] += 1
                else:
                    map[word] = 1
        max_val = 0
        wanted_str = ""
        for key,value in map.iteritems():
            if value > max_val:
                max_val = value
                wanted_str = key
        return wanted_str

791. Custom Sort String
Medium

434

134

Favorite

Share
S and T are strings composed of lowercase letters. In S, no letter occurs more than once.

S was sorted in some custom order previously. We want to permute the characters of T so that they match the order that S was sorted. More specifically, if x occurs before y in S, then x should occur before y in the returned string.

Return any permutation of T (as a string) that satisfies this property.

Example :
Input: 
S = "cba"
T = "abcd"
Output: "cbad"
Explanation: 
"a", "b", "c" appear in S, so the order of "a", "b", "c" should be "c", "b", and "a". 
Since "d" does not appear in S, it can be at any position in T. "dcba", "cdba", "cbda" are also valid outputs.


class Solution:
    def customSortString(self, S, T):
        d = {}
        for t in T:
            if t in d: d[t] +=1
            else: d[t] = 1
        
        res = []
        for s in S:
            if s in d:
                #print(d[s],s)
                res += s * d[s]
                
        T = list(T)
        print(T)
        T.sort()
        print("SORTED ",T)
        
        rest = []
        for t in T:
            if t not in res:
                rest.append(t)
                
        return ''.join(res + rest)
        
 686. Repeated String Match
Easy

529

522

Favorite

Share
Given two strings A and B, find the minimum number of times A has to be repeated such that B is a substring of it. If no such solution, return -1.

For example, with A = "abcd" and B = "cdabcdab".

Return 3, because by repeating A three times (“abcdabcdabcd”), B is a substring of it; and B is not a substring of A repeated two times ("abcdabcd").

Note:
The length of A and B will be between 1 and 10000.

# Python3 program to conStruct a  
# binary tree from the given String  
  
# Helper class that allocates a new node 
class newNode: 
    def __init__(self, data): 
        self.data = data  
        self.left = self.right = None
  
# This funtcion is here just to test  
def preOrder(node): 
    if (node == None):  
        return
    print(node.data, end = " ")  
    preOrder(node.left)  
    preOrder(node.right) 
  
# function to return the index of  
# close parenthesis  
def findIndex(Str, si, ei): 
    if (si > ei):  
        return -1
  
    # Inbuilt stack  
    s = [] 
    for i in range(si, ei + 1): 
  
        # if open parenthesis, push it  
        if (Str[i] == '('):  
            s.append(Str[i])  
  
        # if close parenthesis  
        elif (Str[i] == ')'):  
            if (s[-1] == '('): 
                s.pop(-1)  
  
                # if stack is empty, this is  
                # the required index  
                if len(s) == 0:  
                    return i 
    # if not found return -1  
    return -1
  
# function to conStruct tree from String  
def treeFromString(Str, si, ei): 
      
    # Base case  
    if (si > ei):  
        return None
  
    # new root  
    root = newNode(ord(Str[si]) - ord('0')) 
    index = -1
  
    # if next char is '(' find the  
    # index of its complement ')'  
    if (si + 1 <= ei and Str[si + 1] == '('):  
        index = findIndex(Str, si + 1, ei)  
  
    # if index found  
    if (index != -1): 
  
        # call for left subtree  
        root.left = treeFromString(Str, si + 2,  
                                     index - 1)  
  
        # call for right subtree  
        root.right = treeFromString(Str, index + 2,  
                                            ei - 1) 
    return root 
  
# Driver Code  
if __name__ == '__main__': 
    Str = "4(2(3)(1))(6(5))"
    root = treeFromString(Str, 0, len(Str) - 1)  
    preOrder(root) 
  
# This code is contributed by pranchalK 


49. Group Anagrams
Medium

1892

123

Favorite

Share
Given an array of strings, group anagrams together.

Example:

Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
Output:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
Note:

All inputs will be in lowercase.
The order of your output does not matter.

class Solution(object):
    
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        res ={}
        for i in strs:
            s = ''.join(sorted(i))
            res[s] = res.get(s,[])+[i]
        return res.values()
        
 139. Word Break
Medium

2522

135

Favorite

Share
Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words.

Note:

The same word in the dictionary may be reused multiple times in the segmentation.
You may assume the dictionary does not contain duplicate words.
Example 1:

Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".

class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        #dp = [0 for _ in range(len(s))]
        dp=[0]*len(s)
        print(dp)

        for i in range(len(s)):
            for j in range(i):
                if dp[j] == True:
                    if s[j+1:i+1] in wordDict:
                        dp[i] = True
            if not dp[i]:
                dp[i] = s[:i+1] in wordDict
        return dp[-1]
 692. Top K Frequent Words
Medium

839

79

Favorite

Share
Given a non-empty list of words, return the k most frequent elements.

Your answer should be sorted by frequency from highest to lowest. If two words have the same frequency, then the word with the lower alphabetical order comes first.

Example 1:
Input: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
Output: ["i", "love"]
Explanation: "i" and "love" are the two most frequent words.
    Note that "i" comes before "love" due to a lower alphabetical order.
Example 2:
	
	
	# Function to reverse words of string 
  
def reverseWords(input): 
      
    # split words of string separated by space 
    inputWords = input.split(" ") 
  
    # reverse list of words 
    # suppose we have list of elements list = [1,2,3,4],  
    # list[0]=1, list[1]=2 and index -1 represents 
    # the last element list[-1]=4 ( equivalent to list[3]=4 ) 
    # So, inputWords[-1::-1] here we have three arguments 
    # first is -1 that means start from last element 
    # second argument is empty that means move to end of list 
    # third arguments is difference of steps 
    inputWords=inputWords[-1::-1] 
  
    # now join words with space 
    output = ' '.join(inputWords) 
      
    return output 
  
if __name__ == "__main__": 
    input = 'geeks quiz practice code'
    print reverseWords(input) 
    
297. Serialize and Deserialize Binary Tree
Hard

1745

87

Favorite

Share
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Example: 

You may serialize the following tree:

    1
   / \
  2   3
     / \
    4   5

as "[1,2,3,null,null,4,5]"
Clarification: The above format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

Note: Do not use class member/global/static variables to store states. Your serialize and deserialize algorithms should be stateless.

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        
    
        if (root is None):
            return "" 

        q = [root]
        ans = ""
        while q:
            curr = q.pop(0)
            if (curr is None):
                ans += " null"
            else:
                ans += " "+str(curr.val)
                if (curr.left is not None):
                    q.append(curr.left)
                else:
                    q.append(None)
                if (curr.right is not None):
                    q.append(curr.right)
                else:
                    q.append(None)    
        return str(ans)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if (len(data) == 0):
            return None

        vals = data.split(' ')[1:]

        root = TreeNode(vals[0])
        q = [root]
        for i in range(1,len(vals),2):
            parent = q.pop(0)
            if (vals[i] != "null"):
                a = TreeNode(vals[i])
                parent.left = a
                q.append(a)
            else:
                parent.left = None

            if (vals[i+1] != "null"): 
                b = TreeNode(vals[i+1])
                parent.right = b
                q.append(b)
            else:
                parent.right = None
        return root
        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
206. Reverse Linked List
Easy

2641

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
        rev = None
        while head: 
            head.next, rev, head = rev, head, head.next
        return rev


class Solution(object):
    def reverseList(self, head):
        rev = None
        while head: 
            head.next, rev, head = rev, head, head.next
        return rev
        
   or iterative
   
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


682. Baseball Game
Easy

334

824

Favorite

Share
You're now a baseball game point recorder.

Given a list of strings, each string can be one of the 4 following types:

Integer (one round's score): Directly represents the number of points you get in this round.
"+" (one round's score): Represents that the points you get in this round are the sum of the last two valid round's points.
"D" (one round's score): Represents that the points you get in this round are the doubled data of the last valid round's points.
"C" (an operation, which isn't a round's score): Represents the last valid round's points you get were invalid and should be removed.
Each round's operation is permanent and could have an impact on the round before and the round after.

You need to return the sum of the points you could get in all the rounds.

Example 1:
Input: ["5","2","C","D","+"]
Output: 30
Explanation: 
Round 1: You could get 5 points. The sum is: 5.
Round 2: You could get 2 points. The sum is: 7.
Operation 1: The round 2's data was invalid. The sum is: 5.  
Round 3: You could get 10 points (the round 2's data has been removed). The sum is: 15.
Round 4: You could get 5 + 10 = 15 points. The sum is: 30.



Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
getMin() -- Retrieve the minimum element in the stack.
 
 
class Solution(object):
    def calPoints(self, ops):
        """
        :type ops: List[str]
        :rtype: int
        """
        score = []
        
        for element in ops:
            if str(element[-1]).isdigit():
                score.append(int(element))
            if element == "+" and len(score) > 1:
                score.append(score[-1] + score[-2])
            if element == "D" and score:
                score.append(score[-1] * 2)
            if element == "C" and score:
                score.pop(-1)
            
        return sum(score)
        
        
      class MinStack(object):


    def __init__(self):
        """
        initialize your data structure here.
        """
        self.minNums = []
        self.nums = []

    def push(self, x):
        self.nums.append(x)
        if not self.minNums or self.minNums[-1] >= x:
            self.minNums.append(x)

    def pop(self):
        num = self.nums.pop()
        if self.minNums[-1] == num:
            self.minNums.pop()

    def top(self) :
        return self.nums[-1]

    def getMin(self):
        return self.minNums[-1]
        


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()


775. Global and Local Inversions
Medium

220

123

Favorite

Share
We have some permutation A of [0, 1, ..., N - 1], where N is the length of A.

The number of (global) inversions is the number of i < j with 0 <= i < j < N and A[i] > A[j].

The number of local inversions is the number of i with 0 <= i < N and A[i] > A[i+1].

Return true if and only if the number of global inversions is equal to the number of local inversions.

Example 1:

Input: A = [1,0,2]
Output: true
Explanation: There is 1 global inversion, and 1 local inversion.
Example 2:

Input: A = [1,2,0]
Output: false
Explanation: There are 2 global inversions, and 1 local inversion.
Note:

A will be a permutation of [0, 1, ..., A.length - 1].
A will have length in range [1, 5000].
The time limit for this problem has been reduced.
Accepted
13,972
Submissions
35,192
class Solution:
    def isIdealPermutation(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        for i, n in enumerate(A):
            if i - n >= 2 or (i - n == 1 and A[i-1] < A[i]):
                return False
        return True


771. Jewels and Stones
Easy

1547

289

Favorite

Share
You're given strings J representing the types of stones that are jewels, and S representing the stones you have.  Each character in S is a type of stone you have.  You want to know how many of the stones you have are also jewels.

The letters in J are guaranteed distinct, and all characters in J and S are letters. Letters are case sensitive, so "a" is considered a different type of stone from "A".

Example 1:

Input: J = "aA", S = "aAAbbbb"
Output: 3
Example 2:

Input: J = "z", S = "ZZ"
Output: 0

class Solution(object):
    def numJewelsInStones(self, J, S):
        """
        :type J: str
        :type S: str
        :rtype: int
        """
        Jewel = dict()
        for j in J:
            Jewel[j]= 1
        count = 0
        for s in S:
            try:
                Jewel[s]
                count += 1
            except:
                continue
        return count
  762. Prime Number of Set Bits in Binary Representation
Easy

164

250

Favorite

Share
Given two integers L and R, find the count of numbers in the range [L, R] (inclusive) having a prime number of set bits in their binary representation.

(Recall that the number of set bits an integer has is the number of 1s present when written in binary. For example, 21 written in binary is 10101 which has 3 set bits. Also, 1 is not a prime.)

Example 1:

Input: L = 6, R = 10
Output: 4
Explanation:
6 -> 110 (2 set bits, 2 is prime)
7 -> 111 (3 set bits, 3 is prime)
9 -> 1001 (2 set bits , 2 is prime)
10->1010 (2 set bits , 2 is prime)


class Solution:
    def countPrimeSetBits(self, L, R):
        prime_set = {2, 3, 5, 7, 11, 13, 17, 19}
        
        ret = 0
        for i in range(L, R+1):
            one_count = bin(i).count('1')
            if one_count in prime_set:
                ret +=1
        
        return ret
        
 746. Min Cost Climbing Stairs
Easy

1179

271

Favorite

Share
On a staircase, the i-th step has some non-negative cost cost[i] assigned (0 indexed).

Once you pay the cost, you can either climb one or two steps. You need to find minimum cost to reach the top of the floor, and you can either start from the step with index 0, or the step with index 1.

Example 1:
Input: cost = [10, 15, 20]
Output: 15
Explanation: Cheapest is start on cost[1], pay that cost and go to the top.
Example 2:
Input: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
Output: 6
Explanation: Cheapest is start on cost[0], and only step on 1s, skipping cost[3].

class Solution(object):
    def minCostClimbingStairs(self, cost):
        if(len(cost)<=2):
            return min(cost)
        else:
            l=[0,0]
            for i in range(2,len(cost)+1):
                l.append(min(l[i-1]+cost[i-1],l[i-2]+cost[i-2]))
            x=l[len(cost)]
            return x
            
 738. Monotone Increasing Digits
Medium

262

45

Favorite

Share
Given a non-negative integer N, find the largest number that is less than or equal to N with monotone increasing digits.

(Recall that an integer has monotone increasing digits if and only if each pair of adjacent digits x and y satisfy x <= y.)

Example 1:
Input: N = 10
Output: 9
Example 2:
Input: N = 1234
Output: 1234
          
 class Solution(object):
    def monotoneIncreasingDigits(self, N):
        nums = map(int, str(N))
        n, L = len(nums), 0
        for R in xrange(n-1):
            if nums[L] != nums[R]:
                L = R
            if nums[R] > nums[R+1]:
                nums[L] -= 1
                print(nums)
                for i in xrange(L+1, n): nums[i] = 9
                return reduce(lambda x, y: x*10+y, nums)
        return N
        
   or 
   class Solution(object):
    def monotoneIncreasingDigits(self, N):
        """
        :type N: int
        :rtype: int
        """
        if N < 10:
            return N
        
        nums = map(int,list(str(N)))
        for i in xrange(len(nums) - 1):
            if nums[i] <= nums[i + 1]:
                continue
            else:
                j = i
                while j >= 0 and nums[j] == nums[i]:
                    j -= 1
                idx = j + 1
                nums[idx] -= 1
                for k in xrange(idx + 1,len(nums)):
                    nums[k] = 9
        
        return int(''.join(map(str,nums)))
        
662. Maximum Width of Binary Tree
Medium

703

140

Favorite

Share
Given a binary tree, write a function to get the maximum width of the given tree. The width of a tree is the maximum width among all levels. The binary tree has the same structure as a full binary tree, but some nodes are null.

The width of one level is defined as the length between the end-nodes (the leftmost and right most non-null nodes in the level, where the null nodes between the end-nodes are also counted into the length calculation.

Example 1:

Input: 

           1
         /   \
        3     2
       / \     \  
      5   3     9 

Output: 4
Explanation: The maximum width existing in the third level with the length 4 (5,3,null,9). 
 
 class Solution(object):
    def widthOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        q = [(root, 1)]
        res = 1
        while q:
            nextlevel = []
            
            for node in q:
                #print(node[1])
                if node[0].left:
                    nextlevel.append((node[0].left, 2 * node[1] - 1))
                    print(nextlevel)
                if node[0].right:
                    nextlevel.append((node[0].right, 2 * node[1]))
                    print(nextlevel)
            if len(nextlevel) >= 2:
                res = max(res, nextlevel[-1][1] - nextlevel[0][1] + 1)
                print(res)
            q = nextlevel    
        return res
        
661. Image Smoother
Easy

189

852

Favorite

Share
Given a 2D integer matrix M representing the gray scale of an image, you need to design a smoother to make the gray scale of each cell becomes the average gray scale (rounding down) of all the 8 surrounding cells and itself. If a cell has less than 8 surrounding cells, then use as many as you can.

Example 1:
Input:
[[1,1,1],
 [1,0,1],
 [1,1,1]]
Output:
[[0, 0, 0],
 [0, 0, 0],
 [0, 0, 0]]
Explanation:
For the point (0,0), (0,2), (2,0), (2,2): floor(3/4) = floor(0.75) = 0
For the point (0,1), (1,0), (1,2), (2,1): floor(5/6) = floor(0.83333333) = 0
For the point (1,1): floor(8/9) = floor(0.88888889) = 0
        
 from itertools import product

class Solution:
    def imageSmoother(self, M):
    	m, n = len(M), len(M[0])
    	A = [[0]*n for i in range(m)]
    	for i,j in product(range(m), range(n)):
    		s = []
    		for I,J in product(range(max(0,i-1),min(i+2,m)),range(max(0,j-1),min(j+2,n))): s.append(M[I][J])
    		A[i][j] = int(sum(s)/len(s))
    	return A

645. Set Mismatch
Easy

445

244

Favorite

Share
The set S originally contains numbers from 1 to n. But unfortunately, due to the data error, one of the numbers in the set got duplicated to another number in the set, which results in repetition of one number and loss of another number.

Given an array nums representing the data status of this set after the error. Your task is to firstly find the number occurs twice and then find the number that is missing. Return them in the form of an array.

Example 1:
Input: nums = [1,2,2,4]
Output: [2,3]
Note:
The given array size will in the range [2, 10000].
The given array's numbers won't have any order.

class Solution:
    def findErrorNums(self, nums):
        
        sum_of_list = sum(nums)
        
        sum_of_nos_till_n = int((len(nums) + 1) * len(nums)/2)
        
        sum_of_list_without_repeat = sum(set(nums))
        
        # print(sum_of_list)
        # print(sum_of_nos_till_n)
        
        final_array = []
        
        final_array.append(sum_of_list - sum_of_list_without_repeat)
        
        final_array.append(sum_of_nos_till_n - sum_of_list_without_repeat)
        
        return final_array
        
        
 640. Solve the Equation
Medium

175

413

Favorite

Share
Solve a given equation and return the value of x in the form of string "x=#value". The equation contains only '+', '-' operation, the variable x and its coefficient.

If there is no solution for the equation, return "No solution".

If there are infinite solutions for the equation, return "Infinite solutions".

If there is exactly one solution for the equation, we ensure that the value of x is an integer.

Example 1:
Input: "x+5-3+x=6+x-2"
Output: "x=2"
Example 2:
Input: "x=x"
Output: "Infinite solutions"
Example 3:
Input: "2x=x"
Output: "x=0"
Example 4:
Input: "2x+3x-6x=x+2"
Output: "x=-1"
Example 5:
Input: "x=x+2"
Output: "No solution"
	
class Solution:
    def calc(self, s):
        coe, rem = 0, 0
        start = 0
        sign = 1
        for i, c in enumerate(s+'+'):
            
            if c in '+-':
                if i > 0:
                    if s[i-1] == 'x':
                        coe += sign * int(s[start:i-1]) if s[start:i-1] else sign
                    else:
                        rem += sign * int(s[start:i])
                if c == '-':
                    sign = -1
                else:
                    sign = 1
                start = i + 1
        return coe, rem
            
    def solveEquation(self, equation):
        """
        :type equation: str
        :rtype: str
        """
        if '=' not in equation:
            return 'No solution'
        left, right = equation.split('=')
        coel, reml = self.calc(left)
        coer, remr = self.calc(right)
        if coel == coer:
            return 'Infinite solutions' if reml == remr else 'No solution'
        return 'x='+str((remr-reml)//(coel-coer))
553. Optimal Division
Medium

131

978

Favorite

Share
Given a list of positive integers, the adjacent integers will perform the float division. For example, [2,3,4] -> 2 / 3 / 4.

However, you can add any number of parenthesis at any position to change the priority of operations. You should find out how to add parenthesis to get the maximum result, and return the corresponding expression in string format. Your expression should NOT contain redundant parenthesis.

Example:
Input: [1000,100,10,2]
Output: "1000/(100/10/2)"
Explanation:
1000/(100/10/2) = 1000/((100/10)/2) = 200
However, the bold parenthesis in "1000/((100/10)/2)" are redundant, 
since they don't influence the operation priority. So you should return "1000/(100/10/2)". 

Other cases:
1000/(100/10)/2 = 50
1000/(100/(10/2)) = 50
1000/100/10/2 = 0.5
1000/100/(10/2) = 2

class Solution(object):
    def optimalDivision(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        n = len(nums)
        if n == 1: return str(nums[0])
        if n == 2: return str(nums[0])+'/'+str(nums[1])
        return str(nums[0])+'/('+'/'.join(map(str,nums[1:]))+')'
537. Complex Number Multiplication
Medium

172

591

Favorite

Share
Given two strings representing two complex numbers.

You need to return a string representing their multiplication. Note i2 = -1 according to the definition.

Example 1:
Input: "1+1i", "1+1i"
Output: "0+2i"
Explanation: (1 + i) * (1 + i) = 1 + i2 + 2 * i = 2i, and you need convert it to the form of 0+2i.
Example 2:
Input: "1+-1i", "1+-1i"
Output: "0+-2i"
Explanation: (1 - i) * (1 - i) = 1 + i2 - 2 * i = -2i, and you need convert it to the form of 0+-2i.


class Solution:
    def get_coefficients(self, num):
        coefficients = num.split('+')
        a = coefficients[0]
        b = coefficients[1]
        return int(a), int(b[:-1])
    
    def get_string_from_coefficients(self, a, b):
        return '{}+{}i'.format(str(a), str(b))
    
    def complexNumberMultiply(self, a,b):
        
        x1, y1 = self.get_coefficients(a)
        x2, y2 = self.get_coefficients(b)
        
        a = x1*x2 - 1*y1*y2
        b = y1*x2 + y2*x1
        
        return self.get_string_from_coefficients(a, b)
        
662. Maximum Width of Binary Tree
Medium

703

140

Favorite

Share
Given a binary tree, write a function to get the maximum width of the given tree. The width of a tree is the maximum width among all levels. The binary tree has the same structure as a full binary tree, but some nodes are null.

The width of one level is defined as the length between the end-nodes (the leftmost and right most non-null nodes in the level, where the null nodes between the end-nodes are also counted into the length calculation.

Example 1:

Input: 

           1
         /   \
        3     2
       / \     \  
      5   3     9 

Output: 4
Explanation: The maximum width existing in the third level with the length 4 (5,3,null,9).

class Solution(object):
    def widthOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        q = [(root, 1)]
        res = 1
        while q:
            nextlevel = []
            
            for node in q:
                print(node[1])
                if node[0].left:
                    nextlevel.append((node[0].left, 2 * node[1] - 1))
                if node[0].right:
                    nextlevel.append((node[0].right, 2 * node[1]))
            if len(nextlevel) >= 2:
                res = max(res, nextlevel[-1][1] - nextlevel[0][1] + 1)
            q = nextlevel    
        return res
 
 
 
 801. Minimum Swaps To Make Sequences Increasing
Medium

550

32

Favorite

Share
We have two integer sequences A and B of the same non-zero length.

We are allowed to swap elements A[i] and B[i].  Note that both elements are in the same index position in their respective sequences.

At the end of some number of swaps, A and B are both strictly increasing.  (A sequence is strictly increasing if and only if A[0] < A[1] < A[2] < ... < A[A.length - 1].)

Given A and B, return the minimum number of swaps to make both sequences strictly increasing.  It is guaranteed that the given input always makes it possible.

Example:
Input: A = [1,3,5,4], B = [1,2,3,7]
Output: 1
Explanation: 
Swap A[3] and B[3].  Then the sequences are:
A = [1, 3, 5, 7] and B = [1, 2, 3, 4]
which are both strictly increasing.

class Solution(object):
    def minSwap(self, A, B):
        N = len(A)
        not_swap, swap = [N] * N, [N] * N
        not_swap[0], swap[0] = 0, 1
        for i in range(1, N):
            if A[i - 1] < A[i] and B[i - 1] < B[i]:
                not_swap[i] = not_swap[i - 1]
                swap[i] = swap[i - 1] + 1
            if A[i - 1] < B[i] and B[i - 1] < A[i]:
                not_swap[i] = min(not_swap[i], swap[i - 1])
                swap[i] = min(swap[i], not_swap[i - 1] + 1)
        return min(swap[-1], not_swap[-1])       
 Next challenges:
Cheapest Flights Within K Stops
Binary Tree Cameras
Shortest Way to Form String


532. K-diff Pairs in an Array
Easy

383

862

Favorite

Share
Given an array of integers and an integer k, you need to find the number of unique k-diff pairs in the array. Here a k-diff pair is defined as an integer pair (i, j), where i and j are both numbers in the array and their absolute difference is k.

Example 1:
Input: [3, 1, 4, 1, 5], k = 2
Output: 2
Explanation: There are two 2-diff pairs in the array, (1, 3) and (3, 5).
Although we have two 1s in the input, we should only return the number of unique pairs.
Example 2:
Input:[1, 2, 3, 4, 5], k = 1
Output: 4
Explanation: There are four 1-diff pairs in the array, (1, 2), (2, 3), (3, 4) and (4, 5).

class Solution(object):
    def findPairs(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        nums.sort()
        count = []
        dict = {}
        for i in xrange(len(nums)):
            if nums[i] in dict:
                
                count.append((dict[nums[i]],nums[i]))
            dict[nums[i]+k] = nums[i]
            #print((dict[nums[i]+k]))
        return len(set(count))
        


617. Merge Two Binary Trees
Easy

1976

136

Favorite

Share
Given two binary trees and imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not.

You need to merge them into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of new tree.

Example 1:

Input: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
Output: 
Merged tree:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7
	 
	 
	 class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if not t1: return t2
        if not t2: return t1
        root = TreeNode(t1.val + t2.val)
        root.val = t1.val + t2.val
        root.left = self.mergeTrees(t1.left, t2.left)
        root.right = self.mergeTrees(t1.right,t2.right)
        return root
        
	 Next challenges:
Construct Binary Tree from Inorder and Postorder Traversal
Count Complete Tree Nodes
Inorder Successor in BST II


204. Count Primes
Easy

1201

432

Favorite

Share
Count the number of prime numbers less than a non-negative number, n.

Example:

Input: 10
Output: 4
Explanation: There are 4 prime numbers less than 10, they are 2, 3, 5, 7.



204. Count Primes
Easy

1201

432

Favorite

Share
Count the number of prime numbers less than a non-negative number, n.

Example:

Input: 10
Output: 4
Explanation: There are 4 prime numbers less than 10, they are 2, 3, 5, 7.
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2: return 0
            
        primes = [True] * n
        primes[0] = primes[1] = False
        for num in range(2, int(math.sqrt(n)) + 1):
            if primes[num]:
                primes[num**2:n:num] = [False] * len(primes[num**2:n:num])
        
        return sum(primes)
        
414. Third Maximum Number
Easy

412

764

Favorite

Share
Given a non-empty array of integers, return the third maximum number in this array. If it does not exist, return the maximum number. The time complexity must be in O(n).

Example 1:
Input: [3, 2, 1]

Output: 1

Explanation: The third maximum is 1.


class Solution(object):
    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = list(set(nums))
        nums.sort()
        return nums[-3] if len(nums) >= 3 else nums[-1]
        
        
Next challenges:
Kth Largest Element in an Array


189. Rotate Array
Easy

1529

634

Favorite

Share
Given an array, rotate the array to the right by k steps, where k is non-negative.

Example 1:

Input: [1,2,3,4,5,6,7] and k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]

189. Rotate Array
Easy

1529

634

Favorite

Share
Given an array, rotate the array to the right by k steps, where k is non-negative.

Example 1:

Input: [1,2,3,4,5,6,7] and k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]


class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        a = [0] * len(nums)
        for i in range(len(nums)):
            a[(i+k)%len(nums)] = nums[i] #recycle

        for i in range(len(nums)):
            nums[i] = a[i]



451. Sort Characters By Frequency
Medium

793

72

Favorite

Share
Given a string, sort it in decreasing order based on the frequency of characters.

Example 1:

Input:
"tree"

Output:
"eert"

Explanation:
'e' appears twice while 'r' and 't' both appear once.
So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.

class Solution(object):
    def frequencySort(self, s):
        d, res = {}, ""
        for c in s:
            d[c] = d.get(c,0) + 1
        dl = sorted(d, key=d.get, reverse=True)
        print (dl)
        for v in dl:
            res += (v * d[v]) 
        return res
355. Design Twitter
Medium

525

142

Favorite

Share
Design a simplified version of Twitter where users can post tweets, follow/unfollow another user and is able to see the 10 most recent tweets in the user's news feed. Your design should support the following methods:

postTweet(userId, tweetId): Compose a new tweet.
getNewsFeed(userId): Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
follow(followerId, followeeId): Follower follows a followee.
unfollow(followerId, followeeId): Follower unfollows a followee.
Example:

Twitter twitter = new Twitter();

// User 1 posts a new tweet (id = 5).
twitter.postTweet(1, 5);

// User 1's news feed should return a list with 1 tweet id -> [5].
twitter.getNewsFeed(1);

// User 1 follows user 2.
twitter.follow(1, 2);

// User 2 posts a new tweet (id = 6).
twitter.postTweet(2, 6);

// User 1's news feed should return a list with 2 tweet ids -> [6, 5].
// Tweet id 6 should precede tweet id 5 because it is posted after tweet id 5.
twitter.getNewsFeed(1);

// User 1 unfollows user 2.
twitter.unfollow(1, 2);

// User 1's news feed should return a list with 1 tweet id -> [5],
// since user 1 is no longer following user 2.
twitter.getNewsFeed(1);


from Queue import PriorityQueue as PQ
class Twitter(object):
    #since we don't have auto-increment id from data base, we need to maintain a global count (post_count) and map tweetid to postcount for ordering (id_map) 
    #poster_map to map tweet to its poster
    #follow_map user to its follower
    #to get news feed, we use a PQ, everytime:
    #    we put the latest tweet from each followee into PQ
    #    get one from the heap
    #    find it's poster
    #    and put another from the same poster onto heap and repeat until exausted or number 10 reached
    def __init__(self):
        self.post_count = 2**31-1 #auto dec counter so new post has smaller count (python only has min heap ... don't want to do the negation trick 
        self.count_map = {}   #map tweet -> post_count
        self.owner_map = {}   #list (ordered) of tweets -> same owner 
        self.poster_map = {}  #map tweet -> poster
        self.follow_map = {}  #map followee -> follower in a set (also need to add self as followee as self-posted need to be fetched for news feed)        
        
    def postTweet(self, userId, tweetId):
        """
        Compose a new tweet.
        :type userId: int        :type tweetId: int        :rtype: void
        """
        self.count_map[tweetId] = self.post_count
        self.owner_map[userId] = self.owner_map.get(userId, []) + [tweetId]
        self.poster_map[tweetId] = userId
        if userId not in self.follow_map: self.follow_map[userId] = {userId} #make sure I can see the posts by myself 
        self.post_count -= 1 #finally increment global count
        

    def getNewsFeed(self, userId, n = 10):
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        :type userId: int        :rtype: List[int]      
        """
        if userId not in self.follow_map: return [] #no follower
        ans = []
        pq = PQ() #for quick sorting of tweets
        idx_map = {k: len(self.owner_map.get(k,[]))-1 for k in self.follow_map[userId]} #to track the last tweet retrieved from each followee
         
        #prime the PQ with the latest feed from each followee
        for f in self.follow_map[userId]:
            if idx_map[f] < 0: continue #this guy has no tweet
            one_feed = self.owner_map[f][idx_map[f]]
            one_feed_count = self.count_map[one_feed]
            pq.put((one_feed_count, one_feed,))
            
        while n > 0:
            if pq.empty(): break                
            new_feed = pq.get()[1] #disregard [0] which has the count
            new_feed_poster = self.poster_map[new_feed]
            ans.append(new_feed)
            #find the next new feed from the same poster
            idx_map[new_feed_poster] -= 1
            if idx_map[new_feed_poster] >= 0:
                one_feed = self.owner_map[new_feed_poster][idx_map[new_feed_poster]]
                one_feed_count = self.count_map[one_feed]                
                pq.put((one_feed_count, one_feed,))            
            n -= 1
        return ans 


    def follow(self, followerId, followeeId):
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int       :type followeeId: int        :rtype: void
        """
        #currently we cannot verify if followeeId is valid or not (??? non-existing followee)
        #automaticall follow self
        self.follow_map[followerId] = self.follow_map.get(followerId, {followerId}).union({followeeId})  
        

    def unfollow(self, followerId, followeeId):
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int        :type followeeId: int        :rtype: void
        """
        if followerId == followeeId: return #noop not allowed to unfollow your self        
        try: self.follow_map[followerId].remove(followeeId)
        except: pass
        
 538. Convert BST to Greater Tree
Easy

1405

92

Favorite

Share
Given a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus sum of all keys greater than the original key in BST.

Example:

Input: The root of a Binary Search Tree like this:
              5
            /   \
           2     13

Output: The root of a Greater Tree like this:
             18
            /   \
          20     13

class Solution:
    sum_values = 0
    
    def convertBST(self, root):
        if not root:
            return None
        self.convertBST(root.right)
        self.sum_values += root.val
        root.val = self.sum_values
        self.convertBST(root.left)
        return root
        
8. String to Integer (atoi)
Medium

1043

6534

Favorite

Share
Implement atoi which converts a string to an integer.

The function first discards as many whitespace characters as necessary until the first non-whitespace character is found. Then, starting from this character, takes an optional initial plus or minus sign followed by as many numerical digits as possible, and interprets them as a numerical value.

The string can contain additional characters after those that form the integral number, which are ignored and have no effect on the behavior of this function.

If the first sequence of non-whitespace characters in str is not a valid integral number, or if no such sequence exists because either str is empty or it contains only whitespace characters, no conversion is performed.

If no valid conversion could be performed, a zero value is returned.

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
        
 508. Most Frequent Subtree Sum
Medium

415

82

Favorite

Share
Given the root of a tree, you are asked to find the most frequent subtree sum. The subtree sum of a node is defined as the sum of all the node values formed by the subtree rooted at that node (including the node itself). So what is the most frequent subtree sum value? If there is a tie, return all the values with the highest frequency in any order.

Examples 1
Input:

  5
 /  \
2   -3
return [2, -3, 4], since all the values happen only once, return all of them in any order.
Examples 2
Input:

  5
 /  \
2   -5

ans = []
        if not root:
            return ans
        queue = [root]
        while queue:
            ans.append(queue[-1].val)
            temp = []
            for node in queue:
                if node.left:
                    temp.append(node.left)
                if node.right:
                    temp.append(node.right)
            queue = temp
        return ans
        
 73. Set Matrix Zeroes
Medium

1204

210

Favorite

Share
Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place.

Example 1:

Input: 
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
Output: 
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]

class Solution(object):
    def setZeroes(self, matrix):
        firstRowHasZero = not all(matrix[0])
        for i in range(1,len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    matrix[0][j] = 0
                    matrix[i][0] = 0

        for i in range(1,len(matrix)):
            for j in range(len(matrix[0])-1,-1,-1):
                if matrix[0][j] == 0 or matrix[i][0] == 0:
                    matrix[i][j] = 0

        if firstRowHasZero:
            matrix[0] = [0]*len(matrix[0])
            
 516. Longest Palindromic Subsequence
Medium

1015

135

Favorite

Share
Given a string s, find the longest palindromic subsequence's length in s. You may assume that the maximum length of s is 1000.

Example 1:
Input:

"bbbab"
Output:
4
One possible longest palindromic subsequence is "bbbb".
Example 2:
Input:

"cbbd"
 
 class Solution(object):
    def longestPalindromeSubseq(self, s):
        d = {}
        def f(s):
            if s not in d:
                maxL = 0    
                for c in set(s):
                    i, j = s.find(c), s.rfind(c)
                    #print(c,"i:",i,"j",j)
                    maxL = max(maxL, 1 if i==j else 2+f(s[i+1:j]))
                    #print("MaxL",maxL)
                d[s] = maxL
            return d[s]
        return f(s)
 Next challenges:
Palindromic Substrings
Count Different Palindromic Subsequences
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

def twoSum(self, numbers, target):
		memory = {}     # dictionary where missing numbers will be stored with pairing indexes
		for i in range(len(numbers)):       # iterate through the whole list
			if numbers[i] not in memory:        # it looks for dic keys, so missing numbers
				memory[target - numbers[i]] = i     # if the number isn't mising, add its target together with its index
			else:
                #print(memory[numbers[i]] + 1)
				return [memory[numbers[i]] + 1, i + 1]  
				
653. Two Sum IV - Input is a BST
Easy

938

113

Favorite

Share
Given a Binary Search Tree and a target number, return true if there exist two elements in the BST such that their sum is equal to the given target.

Example 1:

Input: 
    5
   / \
  3   6
 / \   \
2   4   7

Target = 9

from collections import deque
class Solution:
    def findTarget(self, root,k):
        if not root:
            return False
        queue, seen = deque([root]),set()
        while queue:
            cur = queue.popleft()
            if cur.val in seen:
                return True
            else:
                seen.add(k-cur.val)
                if cur.left: queue.append(cur.left)
                if cur.right: queue.append(cur.right)
        return False
        
102. Binary Tree Level Order Traversal
Medium

1687

44

Favorite

Share
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]
class Solution(object):
    def levelOrder(self, root):
        ret = []

        level = [root]

        while root and level:
            currentNodes = []
            nextLevel = []
            for node in level:
                currentNodes.append(node.val)
                if node.left:
                    nextLevel.append(node.left)
                if node.right:
                    nextLevel.append(node.right)
            ret.append(currentNodes)
            level = nextLevel


        return ret
            
 Binary Tree Zigzag Level Order Traversal
Binary Tree Level Order Traversal II
Minimum Depth of Binary Tree
Binary Tree Vertical Order Traversal
Average of Levels in Binary Tree
Cousins in Binary Tree


517. Super Washing Machines
Hard

212

117

Favorite

Share
You have n super washing machines on a line. Initially, each washing machine has some dresses or is empty.

For each move, you could choose any m (1 ≤ m ≤ n) washing machines, and pass one dress of each washing machine to one of its adjacent washing machines at the same time .

Given an integer array representing the number of dresses in each washing machine from left to right on the line, you should find the minimum number of moves to make all the washing machines have the same number of dresses. If it is not possible to do it, return -1.

Example1

Input: [1,0,5]

Output: 3

Explanation: 
1st move:    1     0 <-- 5    =>    1     1     4
2nd move:    1 <-- 1 <-- 4    =>    2     1     3    
3rd move:    2     1 <-- 3    =>    2     2     2 

class Solution(object):
    def findMinMoves(self, machines):
        total, n = sum(machines), len(machines)
        if total % n: return -1
        target, res, toRight = total / n, 0, 0
        for m in machines:
            toRight = m + toRight - target
            res = max(res, abs(toRight), m - target)
        return res
 Next challenges:
Self Crossing
Count Numbers with Unique Digits
Minimum Factorization


126. Word Ladder II
Hard

1093

197

Favorite

Share
Given two words (beginWord and endWord), and a dictionary's word list, find all shortest transformation sequence(s) from beginWord to endWord, such that:

Only one letter can be changed at a time
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
Note:

Return an empty list if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.
Example 1:

Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output:
[
  ["hit","hot","dot","dog","cog"],
  ["hit","hot","lot","log","cog"]
]
Example 2:

Input:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

Output: []

Explanation: The endWord "cog" is not in wordList, therefore no possible transformation.
 
126. Word Ladder II
Hard

1093

197

Favorite

Share
Given two words (beginWord and endWord), and a dictionary's word list, find all shortest transformation sequence(s) from beginWord to endWord, such that:

Only one letter can be changed at a time
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
Note:

Return an empty list if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.
Example 1:

Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]
https://leetcode.com/problems/word-ladder-ii/discuss/269012/Python-BFS%2BBacktrack-Greatly-Improved-by-directional-BFS
            
def findLadders(beginWord, endWord, wordList):
	tree, words, n = collections.defaultdict(set), set(wordList), len(beginWord)
	if endWord not in wordList: return []
	found, bq, eq, nq, rev = False, {beginWord}, {endWord}, set(), False
	while bq and not found:
		words -= set(bq)
		for x in bq:
			for y in [x[:i]+c+x[i+1:] for i in range(n) for c in 'qwertyuiopasdfghjklzxcvbnm']:
				if y in words:
					if y in eq: found = True
					else: nq.add(y)
					tree[y].add(x) if rev else tree[x].add(y)
		bq, nq = nq, set()
		if len(bq) > len(eq): bq, eq, rev = eq, bq, not rev
	def bt(x): 
		return [[x]] if x == endWord else [[x] + rest for y in tree[x] for rest in bt(y)]
	return bt(beginWord)
            

160. Intersection of Two Linked Lists
Easy

2325

221

Favorite

Share
Write a program to find the node at which the intersection of two singly linked lists begins.

For example, the following two linked lists:


begin to intersect at node c1.

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        hash_table_A = {}
        while headA != None:
            hash_table_A[headA] = headA.next
            headA = headA.next
        while headB != None:
            if headB in hash_table_A:
                return headB
            headB = headB.next
        return None
        
 215. Kth Largest Element in an Array
Medium

2283

188

Favorite

Share
Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

Example 1:

Input: [3,2,1,5,6,4] and k = 2
Output: 5
Example 2:

Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4

Next challenges:
Wiggle Sort II
Top K Frequent Elements
Kth Largest Element in a Stream
K Closest Points to Origin
class Solution(object):
    def findKthLargest(self, nums, k):
  
        nums.sort()
        return nums[-k]
 380. Insert Delete GetRandom O(1)
Medium

1263

95

Favorite

Share
Design a data structure that supports all following operations in average O(1) time.

insert(val): Inserts an item val to the set if not already present.
remove(val): Removes an item val from the set if present.
getRandom: Returns a random element from current set of elements. Each element must have the same probability of being returned.
Example:

// Init an empty set.
RandomizedSet randomSet = new RandomizedSet();

// Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomSet.insert(1);

// Returns false as 2 does not exist in the set.
randomSet.remove(2);

// Inserts 2 to the set, returns true. Set now contains [1,2].
randomSet.insert(2);

// getRandom should return either 1 or 2 randomly.
randomSet.getRandom();

// Removes 1 from the set, returns true. Set now contains [2].
randomSet.remove(1);

// 2 was already in the set, so return false.
randomSet.insert(2);

// Since 2 is the only number in the set, getRandom always return 2.
randomSet.getRandom();

743 VIEWS

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
 240. Search a 2D Matrix II
Medium

1695

48

Favorite

Share
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.
Example:

Consider the following matrix:

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
Given target = 5, return true.

Given target = 20, return false.

class Solution(object):
    def searchMatrix(self, matrix, target):
        m = len(matrix)
        if m==0:return False
        n = len(matrix[0])
        if n==0:return False
        print(m,n)
        index = n-1
        for i in range(m):
            while index>=0 and matrix[i][index]>target:
                index-=1
            if index == -1:break
            if matrix[i][index]==target:return True
        return False
Next challenges:
Search a 2D Matrix

78. Subsets
Medium

2211

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

import itertools
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = [[]]
        for l in range(len(nums)):
            for item in itertools.combinations(nums, l+1):
                res.append(sorted(item))
        return res
Next challenges:
Subsets II
Generalized Abbreviation
Letter Case Permutation
784. Letter Case Permutation
Easy

718

87

Favorite

Share
Given a string S, we can transform every letter individually to be lowercase or uppercase to create another string.  Return a list of all possible strings we could create.

Examples:
Input: S = "a1b2"
Output: ["a1b2", "a1B2", "A1b2", "A1B2"]

Input: S = "3z4"
Output: ["3z4", "3Z4"]

Input: S = "12345"
Output: ["12345"]
class Solution(object):
    def letterCasePermutation(self, S):
        """
        :type S: str
        :rtype: List[str]
        """
        digits = {str(x) for x in range(10)}
        A = ['']
        for c in S:
            B = []
            if c in digits:
                for a in A:
                    B.append(a+c)
            else:
                for a in A:
                    B.append(a+c.lower())
                    B.append(a+c.upper())
            A=B
        return A
        
 535. Encode and Decode TinyURL
Medium

497

1063

Favorite

Share
Note: This is a companion problem to the System Design problem: Design TinyURL.
TinyURL is a URL shortening service where you enter a URL such as https://leetcode.com/problems/design-tinyurl and it returns a short URL such as http://tinyurl.com/4e9iAk.

Design the encode and decode methods for the TinyURL service. There is no restriction on how your encode/decode algorithm should work. You just need to ensure that a URL can be encoded to a tiny URL and the tiny URL can be decoded to the original URL.
import string
import random

class Codec:
    
    def __init__(self):
		# dict to store all longUrl to shortUrl mappings
        self.long_to_short = dict()

    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.
        
        :type longUrl: str
        :rtype: str
        """
		# Test if mapping already exists
        if longUrl in self.long_to_short.keys():
            return self.long_to_short[longUrl]
        
		# Generate random 6 character string for tinyUrl
        def generate_shortUrl():
            all_chars = string.ascii_letters + string.digits
            shortUrl = "".join(random.choice(all_chars) for x in range(6))
            return shortUrl
        
		# Keep generating new shortUrl till it finds one that doesn't exist in our dict
        shortUrl = generate_shortUrl()
        while shortUrl in self.long_to_short.values():
            shortUrl = generate_shortUrl()
        
		# map
        self.long_to_short[longUrl] = shortUrl
        
        return shortUrl
                

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.
        
        :type shortUrl: str
        :rtype: str
        """
		# Simple mapping
        for k, v in self.long_to_short.items():
            if v == shortUrl:
                return k
        return None
        
        Next challenges:
Generate Random Point in a Circle
Number of Squareful Arrays
Numbers With Repeated Digits
