import treex as tx

mlp = tx.MLP([784, 256, 128, 10]).init(42)

print(mlp)
print(mlp.tabulate())

print()
print()
print()
print()
