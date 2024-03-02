while True:
    a = int(input('a: '))
    b = int(input('b: '))

    if (b > a) or (-1 < a < 1e8) or (0 < b < 1e8 + 1):
        break

def isOdd(num):
    if num % 2 == 0:
        return False
    else:
        return True


result = 0
for i in range(a + 1, b):
    if isOdd(i):

        result += i

result %= 10000007

print(result)


# python Question_1.py
