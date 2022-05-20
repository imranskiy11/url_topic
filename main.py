import time
from termcolor import colored
import torch

def main():
    print(colored(f'Starting ///', 'green'))
    start_time = time.time()
    print(f'is cuda available :: {torch.cuda.is_available()}')\

    #do something

    print(colored(f'Finish ///', 'green'), colored(f'spended time : {round(time.time() - start_time, 2)}', 'red'))


if __name__ == '__main__':
    main()
