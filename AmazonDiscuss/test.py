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
