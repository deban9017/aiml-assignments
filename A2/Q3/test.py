import oracle
import pickle

data = oracle.q3_linear_1(23607)

# The output is a 4-tuple: (X_train,y_train,X_test,y_test) ; save it to a file
with open('data1.pkl', 'wb') as f:
    pickle.dump(data, f)


data = oracle.q3_linear_2(23607)

# The output is a 4-tuple: (X_train,y_train,X_test,y_test) ; save it to a file
with open('data2.pkl', 'wb') as f:
    pickle.dump(data, f)
