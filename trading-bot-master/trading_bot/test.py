import copy

class cp:
    def __init__(self):
        self.a = 0

        self.buy = 5
        self.sell = 3

    def add(self, mode):
        if (mode == 'buy'):
            self.value = copy.copy(self.buy)
        elif (mode == 'sell'):
            self.value = copy.copy(self.sell)

        self.value += 1
        print('value : ', self.value)
        print('buy : ', self.buy)
        print('sell: ', self.sell)

tmp_cp = cp()

tmp_cp.add('buy')
tmp_cp.add('sell')
tmp_cp.add('sell')
tmp_cp.add('sell')
tmp_cp.add('sell')

print(tmp_cp.buy)
print(tmp_cp.sell)

global A
B = 6
A = 5

def add2(mode):
    if (mode == 'buy'):
        num = A
    elif (mode == 'sell'):
        num = B

    num += 1
    print(num)

add2('buy')
print(A)
print(B)