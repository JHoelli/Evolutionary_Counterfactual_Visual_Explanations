import Additional_Help_Functions

import matplotlib.pyplot as plt
import numpy as np
import random
#TODO + and -

x_train, y_train,x_test, y_test=Additional_Help_Functions.load_mnist()

'''High L2, Einmal Low L1, Einmal Heigh L1 '''
i = 0
plt.figure()
plt.imshow(np.array(x_test[0] ).reshape(28, 28, 1),cmap='gray')
plt.savefig(f'Orig.png')
plt.close()

l2 =400
# Low L1
x=x_test[0].reshape(-1)
#new_id=np.where(x>100)
#print(new_id)
new_im = x_test[0].reshape(-1)
for a in range(0,4):
    pixel=random.choice(new_im)
    new_im[pixel]=np.abs(x_test[0].reshape(-1)[pixel]-100)

plt.figure()
plt.imshow(np.array(new_im ).reshape(28, 28, 1), cmap='gray')
plt.savefig(f'HigL2LowL1.png')
plt.close()



x=x_test[0].reshape(-1)
new_id=np.where(x>40)
print(new_id)
new_im = x_test[0].reshape(-1)
for a in range(0,10):
    pixel=random.choice(new_im)
    new_im[pixel]=np.abs(x_test[0].reshape(-1)[pixel]-40)

plt.figure()
plt.imshow(np.array(new_im ).reshape(28, 28, 1), cmap='gray')
plt.savefig(f'HigL2HighL1.png')
plt.close()