import numpy as np
import neurolab as nl

# Щ М Ю
target = [[1, 1, 1, 1, 0,
           1, 1, 1, 1, 0,
           1, 1, 1, 1, 0,
           1, 1, 1, 1, 0,
           1, 1, 1, 1, 1],
          
          [1, 1, 0, 1, 1,
           1, 0, 1, 0, 1,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1],
          
          [1, 0, 0, 1, 0,
           1, 0, 1, 0, 1,
           1, 1, 1, 0, 1,
           1, 0, 1, 0, 1,
           1, 0, 0, 1, 0]]

chars = ['Щ', 'М', 'Ю']
target = np.asfarray(target)
target[target == 0] = -1

# Create and train network
net = nl.net.newhop(target)

output = net.sim(target)
print("Test on train samples:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())


print("\nTest on defaced 'Ю':")
test_you = np.asfarray([1, 0, 1, 1, 1,
                        1, 0, 1, 0, 1,
                        1, 0, 1, 0, 1,
                        1, 0, 1, 0, 1,
                        1, 0, 1, 1, 1])
test_you[test_you == 0] = -1
out_you = net.sim([test_you])
print((out_you[0] == target[0]).all(), 'Sim. steps', len(net.layers[0].outs))


print("\nTest on defaced 'М':")
test_m = np.asfarray([1, 1, 1, 1, 1,
                      1, 0, 1, 0, 1,
                      1, 0, 0, 0, 1,
                      1, 0, 0, 0, 1,
                      1, 0, 0, 0, 1])
test_m[test_m == 0] = -1
out_m = net.sim([test_m])
print((out_m[0] == target[0]).all(), 'Sim. steps', len(net.layers[0].outs))
