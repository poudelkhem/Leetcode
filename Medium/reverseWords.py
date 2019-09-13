
# ~ Given an input string, reverse the string word by word.

 

# ~ Example 1:

# ~ Input: "the sky is blue"
# ~ Output: "blue is sky the"
# ~ Example 2:

# ~ Input: "  hello world!  "
# ~ Output: "world! hello"
# ~ Explanation: Your reversed string should not contain leading or trailing spaces.
# ~ Example 3:

# ~ Input: "a good   example"
# ~ Output: "example good a"
# ~ Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.

class Solution:
	def reverseWords(self, s):
		result=s.strip().split()
		
		i,j=0,len(result)-1
		while i<j:
			result[i],result[j]=result[j],result[i]
			i+=1
			j-=1
		return ' '.join(result)
	
			




		
b1 = Solution()
in1="the sky is blue"

res=b1.reverseWords(in1)
print(res)

