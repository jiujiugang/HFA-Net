import numpy as np
image = np.asarray([np.random.rand (320,320,4)for _ in range])
image =image [:,:,:,:3]
image =image .transpose(0,3,1,2)
print(image.shape)