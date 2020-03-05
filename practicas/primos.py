'''
Desarrollar un programa que encuentre todos los números primos del 1 a n.
'''

import sys

def is_prime_loop(x):
    for i in range(2, x):
        if x % i == 0:
            return False
    return True

def get_primes_filter(n):
    return filter(is_prime_loop, range(2, n))

def get_primes_gen_loop(n):
    for i in range(2, n):
        if is_prime_loop(i):
            yield i

def is_prime(n, primes):
    for p in primes:
        if n % p == 0:
            return False
    return True

def get_primes_list(n):
    primes = []
    for i in range(2, n + 1):
        if is_prime(i, primes):
            primes.append(i)
    return primes

def get_primes_generator(n):
    primes = []
    for i in range(2, n + 1):
        if is_prime(i, primes):
            primes.append(i)
            yield i

if __name__ == "__main__":
    n = int(sys.argv[1])
    option = sys.argv[2]
    if option == 'list':
        primes = get_primes_list(n)
        for x in primes[:10]:
            print('%d, ' % x, end='')
    elif option == 'gen':
        primes = get_primes_generator(n)
        for i, x in enumerate(primes):
            if i == 10:
                break
            print('%d, ' % x, end='')
    elif option == 'genloop':
        primes = get_primes_gen_loop(n)
        for i, x in enumerate(primes):
            if i == 10:
                break
            print('%d, ' % x, end='')
