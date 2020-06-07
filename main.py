import numpy as np 
from rnn_utils import *
import pprint

def main():
    data = open('dino.txt', 'r').read()
    data= data.lower()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)

    chars = sorted(chars)
    print(chars)
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(ix_to_char)


if __name__ == "__main__":
    main()