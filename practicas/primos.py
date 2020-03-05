'''
Desarrollar un programa que encuentre todos los números primos del 1 a n.
'''

import sys

def is_prime(n, primes):
    for p in primes:
        if n % p == 0:
            return False
    return True

def get_primes(n):
    primes = [2]
    for i in range(3, n + 1, 2):
        if is_prime(i, primes[1:]):
            primes.append(i)
    return primes

if __name__ == "__main__":
    n = int(sys.argv[1])
    primes = get_primes(n)
    print(primes)
