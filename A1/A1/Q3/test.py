import sys
import os
import numpy as np


# Add the 'oracle' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'oracle'))
import oracle as oracle 

def main():
    res = oracle.q3_hyper(23607)
    print(res)


if __name__ == '__main__':
    main()