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

dict = ["apple", "boy", "cat", "dog", "element", "zack", "zill"]
range("a", "z");
r1=RangeDictionary(dict)
