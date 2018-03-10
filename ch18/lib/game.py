"""
4-in-a-row game-related functions.

Field is 6*7 with pieces falling from the top to the bottom. There are two kinds of pieces: black and white,
which are encoded by 1 (black) and 0 (white).

There are two representation of the game:
1. List of 7 lists with elements ordered from the bottom. For example, this field
 
0     1
0     1
10    1
10  0 1
10  1 1
101 111

Will be encoded as [
  [1, 1, 1, 1, 0, 0],
  [0, 0, 0, 0],
  [1],
  [],
  [1, 1, 0],
  [1],
  [1, 1, 1, 1, 1, 1]
]
  
2. integer number consists from:
    a. 7*6 bits (column-wise) encoding the field. Unoccupied bits are zero
    b. 7*3 bits, each 3-bit number encodes amount of free entries on the top.
In this representation, the field above will be equal to those bits:
[
    111100,
    000000,
    100000,
    000000,
    110000,
    100000,
    111111,
    000,
    010,
    101,
    110,
    011,
    101,
    000
]

All the code is generic, so, in theory you can try to adjust the field size. 
But tests could become broken.
"""
GAME_ROWS = 6
GAME_COLS = 7
BITS_IN_LEN = 3


def bits_to_int(bits):
    res = 0
    for b in bits:
        res *= 2
        res += b
    return res


def int_to_bits(num, bits):
    res = []
    for _ in range(bits):
        res.append(num % 2)
        num //= 2
    return res[::-1]


def encode_lists(field_lists):
    """
    Encode lists representation into the binary numbers
    :param field_lists: list of GAME_COLS lists with 0s and 1s
    :return: integer number with encoded game state
    """
    assert isinstance(field_lists, list)
    assert len(field_lists) == GAME_COLS

    bits = []
    len_bits = []
    for col in field_lists:
        bits.extend(col)
        free_len = GAME_ROWS-len(col)
        bits.extend([0] * free_len)
        len_bits.extend(int_to_bits(free_len, bits=BITS_IN_LEN))
    bits.extend(len_bits)
    return bits_to_int(bits)


def decode_binary(num):
    """
    Decode binary representation into the list view
    :param num: integer representing the field 
    :return: list of GAME_COLS lists 
    """
    bits = int_to_bits(num, bits=GAME_COLS*GAME_ROWS + GAME_COLS*BITS_IN_LEN)
    res = []
    len_bits = bits[GAME_COLS*GAME_ROWS:]
    for col in range(GAME_COLS):
        vals = bits[col*GAME_ROWS:(col+1)*GAME_ROWS]
        lens = bits_to_int(len_bits[col*BITS_IN_LEN:(col+1)*BITS_IN_LEN])
        if lens > 0:
            vals = vals[:-lens]
        res.append(vals)
    return res
