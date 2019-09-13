# ~ https://leetcode.com/problems/longest-common-subsequence/

# ~ 1143. Longest Common Subsequence
# ~ Medium

# ~ 18

# ~ 2

# ~ Favorite

# ~ Share
# ~ Given two strings text1 and text2, return the length of their longest common subsequence.

# ~ A subsequence of a string is a new string generated from the original string with some characters(can be none) deleted without changing the relative order of the remaining characters. (eg, "ace" is a subsequence of "abcde" while "aec" is not). A common subsequence of two strings is a subsequence that is common to both strings.

# ~ If there is no common subsequence, return 0.

 

# ~ Example 1:

# ~ Input: text1 = "abcde", text2 = "ace" 
# ~ Output: 3  
# ~ Explanation: The longest common subsequence is "ace" and its length is 3.
# ~ Example 2:

# ~ Input: text1 = "abc", text2 = "abc"
# ~ Output: 3
# ~ Explanation: The longest common subsequence is "abc" and its length is 3.
# ~ Example 3:

# ~ Input: text1 = "abc", text2 = "def"
# ~ Output: 0
# ~ Explanation: There is no such common subsequence, so the result is 0.
 

# ~ Constraints:

# ~ 1 <= text1.length <= 1000
# ~ 1 <= text2.length <= 1000
# ~ The input strings consist of lowercase English characters only.
import numpy as np
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
	def addTwoNumbers(self, l1,l2):
		h=ListNode(0)
		p,carry=h,0
		while l1 or l2:
			sum1=carry
			if l1:
				sum1,l1=sum1+l1.val,l1.next
			if l2:
				sum1, l2=sum1+l2.val,l2.next
			p.next,carry=ListNode(sum1%10),sum1/10
			p=p.next
		p.next=ListNode(carry) if carry else None
		return h.next
			




		
b1 = Solution()
b1.val=2
in1=[2 , 4 , 3]
in2=[5 , 6 , 4]
res=b1.addTwoNumbers(in1,in2)
print(res)

