import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update( {'font.size': 10} )

file = 'bo_std_log'

i = 1
f = open(file)
lines=[]
svpg_params = []
for line in f.readlines():
    if i % 2 == 0 :
        lines.append( line )
        parse = line.split()
        svpg_params.append(float(parse[1]))
    i += 1

print(svpg_params)

plt.plot(svpg_params, marker='o')

plt.ylabel('std of mean')
plt.xlabel('Called BO 10 times, each time BO has 15 iters')
#plt.title('trend of SVPG')
plt.show()