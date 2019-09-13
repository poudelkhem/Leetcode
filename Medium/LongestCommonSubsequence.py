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
class Solution(object):
	def longestCommonSubsequence(self, x, y):
		
		m,n=len(x),len(y)
		c=[[0] *(n+1) for i in range (m+1)]
		#print(np.matrix(c))
		for i in range(1, m+1):
			for j in range(1,n+1):
				if x[i-1]==y[j-1]:
					c[i][j]=c[i-1][j-1] +1
					print(i,j,"c[i][j] :" , c[i][j])
				else:
					
					c[i][j]=max(c[i-1][j], c[i][j-1])
					print(i,j," ELSE c[i][j] :" , c[i][j])
					
		return c[m][n]
		
b1 = Solution()
# ~ text1 = "abcde"
# ~ text2 = "ace" 
text1 = "abc"
text2 = "def"
res=b1.longestCommonSubsequence(text1,text2)
print(res)

