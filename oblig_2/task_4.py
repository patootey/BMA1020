class Complex:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return "({0},{1})".format(self.x, self.y)

    def conj(self):
        return Complex(self.x, -self.y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Complex(x, y)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Complex(x, y)

    def __mul__(self, other):
        x = (self.x * other.x) - (self.y * other.y)  # Real part: ac - bd
        y = (self.y * other.x) + (self.x * other.y)  # Imaginary part: ad + bc
        return Complex(x, y)

    def __truediv__(self, other):
        kvadratisk_norm_other = (other.x) ** 2 + (other.y) ** 2
        x = (self.x * other.x + self.y * other.y) / kvadratisk_norm_other
        y = (self.y * other.x - self.x * other.y) / kvadratisk_norm_other
        return Complex(x, y)
