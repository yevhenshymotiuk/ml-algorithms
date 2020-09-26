from functools import reduce


def add(A, B):
    return [A[i] + B[i] for i in range(len(A))]


def dot(A, B):
    return sum(A[i] * B[i] for i in range(len(A)))


def multiply_by_number(A, n):
    return [a * n for a in A]


def subtract(A, B):
    return [A[i] - B[i] for i in range(len(A))]


def vsum(vectors):
    return reduce(lambda v, result: add(result, v), vectors)
