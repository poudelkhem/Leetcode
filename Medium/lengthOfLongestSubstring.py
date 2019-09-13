
# ~ https://leetcode.com/problems/longest-substring-without-repeating-characters/

# ~ 3. Longest Substring Without Repeating Characters
# ~ Medium

# ~ 5972

# ~ 342

# ~ Favorite

# ~ Share
# ~ Given a string, find the length of the longest substring without repeating characters.

# ~ Example 1:

# ~ Input: "abcabcbb"
# ~ Output: 3 
# ~ Explanation: The answer is "abc", with the length of 3. 
# ~ Example 2:

# ~ Input: "bbbbb"
# ~ Output: 1
# ~ Explanation: The answer is "b", with the length of 1.
# ~ Example 3:

# ~ Input: "pwwkew"
# ~ Output: 3
# ~ Explanation: The answer is "wke", with the length of 3. 
             # ~ Note that the answer must be a substring, "pwke" is a subsequence and not a substring.

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
            if len(temp)>max_len:
                max_len=len(temp)
        return max_len
