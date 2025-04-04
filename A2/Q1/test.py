import Q1.oracle as oracle

res = oracle.q1_get_cifar100_train_test(23607)

with open("res.txt", "w") as f:
    f.write(str(res))

