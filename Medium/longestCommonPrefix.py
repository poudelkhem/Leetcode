strs=["flower","flow","flight"]
lcp = ""
for s in zip(*strs):
	print("S ",s)
	print((s[0],)* len(s))
	
	if (s[0],) * len(s) == s:
		lcp += s[0]
	else:
		break
print (lcp)
