from task_1 import uint8, int8


def int_bin8(v: int) -> str:
    v = v % 256
    return "0b" + ("".join([str(int(bool(v & (2**b)))) for b in range(7, -1, -1)]))


def uint8_int(v: uint8) -> int:
    return sum(
        map(lambda x, y: x * y, map(lambda x: 2**x, range(7, -1, -1)), map(int, v.bits))
    )


if __name__ == "__main__":
    for i in range(256):
        a = uint8.integer(i)
        assert repr(a) == int_bin8(i), f"uint8: {repr(a)}=={int_bin8(i)}"

    for i in range(256):
        for j in range(8):
            a = uint8.integer(i)
            s1 = a.leftshift(j)
            s2 = (i << j) % 256
            assert uint8_int(s1) == s2, f"uint8: {i}<<{j}: {uint8_int(s1)}!={s2}"

    for i in range(256):
        for j in range(256):
            a = uint8.integer(i)
            b = uint8.integer(j)
            s1 = a + b
            s2 = (i + j) % 256
            assert uint8_int(s1) == s2, f"uint8: {i}+{j}: {uint8_int(s1)}!={s2}"

    for i in range(256):
        for j in range(256):
            a = uint8.integer(i)
            b = uint8.integer(j)
            s1 = a * b
            s2 = (i * j) % 256
            assert uint8_int(s1) == s2, f"uint8: {i}*{j}: {uint8_int(s1)}!={s2}"

    for i in range(-128, 128):
        a = int8.integer(i)
        assert repr(a) == int_bin8(i), f"int8: {repr(a)}=={int_bin8(i)}"

    for i in range(-128, 128):
        for j in range(8):
            a = int8.integer(i)
            s1 = a.leftshift(j)
            s2 = (i << j) % 256
            assert uint8_int(s1) == s2, f"int8: {i}<<{j}: {uint8_int(s1)}!={s2}"

    for i in range(-128, 128):
        for j in range(256):
            a = int8.integer(i)
            b = int8.integer(j)
            s1 = a + b
            s2 = (i + j) % 256
            assert uint8_int(s1) == s2, f"int8: {i}+{j}: {uint8_int(s1)}!={s2}"

    for i in range(-128, 128):
        for j in range(256):
            a = int8.integer(i)
            b = int8.integer(j)
            s1 = a - b
            s2 = (i - j) % 256
            assert uint8_int(s1) == s2, f"int8: {i}-{j}: {uint8_int(s1)}!={s2}"

    for i in range(-128, 128):
        for j in range(-128, 128):
            a = int8.integer(i)
            b = int8.integer(j)
            s1 = a * b
            s2 = (i * j) % 256
            assert uint8_int(s1) == s2, f"int8: {i}*{j}: {uint8_int(s1)}!={s2}"
