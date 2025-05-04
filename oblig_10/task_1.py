land = lambda x, y: x and y
lxor = lambda x, y: x ^ y
lnot = lambda x: not x


class uint8:
    def __init__(self, bits: list[bool]) -> None:
        self.bits = bits

    def __repr__(self) -> str:
        # Changed to bit format as python uses
        return "0b" + ("".join(map(lambda x: str(int(x)), self.bits)))

    def __add__(self, other: "uint8") -> "uint8":  # 7 points
        x = uint8(list(map(lxor, self.bits, other.bits)))
        carry = uint8(list(map(land, self.bits, other.bits))).leftshift(1)

        if not any(carry.bits):
            return x

        return x + carry

    def __mul__(self, other: "uint8") -> "uint8":  # 13 points
        return sum(
            [self.leftshift(i) for i in range(8) if other.bits[7 - i]], uint8.integer(0)
        )

    def leftshift(self, n: int) -> "uint8":  # 5 points
        n = max(0, min(n, 8))
        return uint8(self.bits[n:] + [False] * n)

    @classmethod
    def integer(cls, n: int) -> "uint8":  # 10 points
        return cls([bool(((2**x) & n)) for x in range(7, -1, -1)])


class int8(uint8):
    def __neg__(self):  # 9 points
        return int8(list(map(lnot, self.bits))) + int8.integer(1)

    def __sub__(self, other: "int8") -> "int8":  # 9 points
        return self + -other

    def __add__(self, other: "int8") -> "int8":
        return int8(super().__add__(other).bits)

    def __mul__(self, other: "int8") -> "int8":
        return int8(super().__mul__(other).bits)

    @classmethod
    def integer(cls, n: int | list) -> "int8":  # 7 points

        # Convert for list
        if isinstance(n, list):
            return cls(n)

        if n < 0:
            # Convert to positive, then negate
            return -cls(uint8.integer(-n).bits)
        else:
            # For positive numbers
            return cls(uint8.integer(n).bits)


if __name__ == "__main__":
    x1 = uint8([False, False, False, False, True, False, False, False])
    y1 = uint8.integer(4)
    print("x1 + y1:", x1 + y1)
    print("x1 * y1:", x1 * y1)

    # Test int8
    x2 = int8.integer([True, False, False, False, True, False, True, False])
    y2 = int8.integer(4)
    z2 = int8.integer(-2)

    print("x2 - y2:", x2 - y2)
    print("y2 * z2:", y2 * z2)
