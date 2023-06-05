import cv2
import numpy as np




def data_generator(x1, x2,xf,y, batch_size):
    # batch_size=batch_size
    size=x1.shape[0]
    print('size:',size)
    while 1:
        for i in range(int(size / batch_size)):
            in1 = x1[i*batch_size: (i+1)*batch_size]
            in2 = x2[i*batch_size: (i+1)*batch_size]
            inf = xf[i*batch_size: (i+1)*batch_size]
            inf=inf.reshape(9,)
            out = y[i*batch_size: (i+1)*batch_size]
            in1 = in1.reshape(in1.shape[0],in1.shape[1],in1.shape[2],1)
            in2 = in2.reshape(in2.shape[0],in2.shape[1],in2.shape[2],1)
            # out2 = y[i*batch_size: (i+1)*batch_size]
            yield [in1,in2,inf],[out]
# print(data_generator(x1, x2, 2))

# def data_generator_simple(x1, batch_size):
#     # batch_size=batch_size
#     size=x1.shape[0]
#     print('size:',size)
#     while 1:
#         for i in range(int(size / batch_size)):
#             in1 = x1[i*batch_size: (i+1)*batch_size]
#             in1 = in1.reshape(batch_size,73,73,256)
#             yield in1