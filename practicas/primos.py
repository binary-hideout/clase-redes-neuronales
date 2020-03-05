'''
Desarrollar un programa que encuentre todos los números primos del 1 a n.
'''

import sys

def is_prime(n, primes):
    for p in primes[1:]:
        if n % p == 0:
            return False
    return True

n = int(sys.argv[1])

primes = [2]
for i in range(3, n + 1, 2):
    if is_prime(i, primes):
        primes.append(i)
