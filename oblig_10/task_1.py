land = (
    lambda x, y: x and y
)  # x and y are logicals (i.e. not lists of logicals; we want you to treat this in the code further below.)


# more to IT
lxor = lambda x, y: (x or y) and not (
    x and y
)  # fill in (such that the output is an exclusive or)


# DO MORE
lnot = lambda x: not x  # fill in (such that the output is "logical not")

# PART 1


class uint8:
    def __init__(self, bits):
        self.bits = bits
        # the bits should be a list of 8 boolean variables
        # eg [True, False, False, True, True, False, False, False]
        # which represents the binary number 10011000  (= 152 in decimal)

    def __repr__(self):
        # notice the use of map() from the functional programming intro!
        # you can leave this unchanged, or modify it if you want to...
        # ...display your numbers in another manner
        return " ".join(map(str, self.bits))

    def __add__(self, other):  # 7 points
        # you should add numbers by combining XOR (add without carry)
        # and AND + leftshift (add with carry)
        # the implementation should be recursive (i.e. call itself)

        # carry = 0
        # Hint: the line under does add without carry
        x = uint8(list(map(lxor, self.bits, other.bits)))
        # left shift, mangler carry
        # Hvordan forflytte etter dette, fordi nå gjør jeg sånn at jeg bare ignorerer siste
        y = list(map(land, self.bits, other.bits))

        y = y[1:] + [False]

        y = uint8(y)

        return x + y

    def __mul__(self, other):  # 13 points

        return  # fill in
        # follow the algorithm in the lectures, i.e. multiply powers of 2
        # by shifting the requisite amount
        # do such a shift for each '1' (i.e. 'True') in the number you multiply by
        # add all the resulting shifts using __add__

    def leftshift(self, n):  # 5 points
        if n <= 0:
            return uint8(self.bits[:])  # original

        # Note: In a leftshift operation, all bits move in memory "to the left". Here,
        # the rightmost bit gets "filled up" by a "False"
        # i.e.  a leftshift on the list of bits
        #  [True, False, False, True, False, True, True, False] will be
        #        [False, False, True, False, True, True, False, False]
        # The variable "n" indicates how often this operation gets used,
        # i.e. a shit with n=3 will use leftshift three times.
        result_bits = self.bits[:]
        for _ in range(n):
            result_bits = result_bits[1:] + [False]
        return uint8(result_bits)

    @classmethod
    def integer(cls, n):  # 10 points
        # make an 8-bit integer from the python integer n
        # i.e. this should convert decimal to binary
        # https://www.geeksforgeeks.org/program-decimal-binary-conversion/'

        # rahter interger to bit instead
        bits = []

        for _ in range(8):
            bits.insert(0, bool(n & 1))
            n >>= 1
        return cls(bits)


# example test
x1 = uint8([False, False, False, False, True, False, False, False])


y1 = uint8.integer(4)

print(x1 + y1)
print(x1 * y1)


# PART 2
class int8(uint8):
    def __neg__(self):  # 9 points
        # use two's complement
        # this finds the negative of the integer
        inverted_bits = [lnot(bit) for bit in self.bits]
        negated = int8(inverted_bits)
        return negated

    def __sub__(self, other):  # 9 points
        # this should combine __add__ and __neg__ to do subtraction
        # depending on how they are programmed, you may need small overrides of
        # the inherited function
        return self + other.__neg__()

    def __add__(self, other):
        # you might be able to inherit from uint8
        result = super().__add__(other)
        return int8(result.bits)

    def __mul__(self, other):
        # you might be able to inherit from uint8
        # implement multiplication with the same algorithm as for unsigned integers
        # this works properly if both numbers you multiply are 4 bits, i.e. -16 to 15
        # see the discussion below
        result = super().__mul__(other)
        return int8(result.bits)

    @classmethod
    def integer(cls, n):  # 7 points
        # fill in

        # handle neg, set to 1, invert
        pass


# example test
x = int8.integer([True, False, False, False, True, False, True, False])

y = int8.integer(4)

z = int8.integer(-2)

print(x - y)
print(y * z)
