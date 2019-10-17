import ctypes

testlib = ctypes.CDLL('knn_lib.so')

def gpu_knn(ref, query, dim, k):

    ref_arr = (ctypes.c_float * len(ref))(*ref)
    query_arr = (ctypes.c_float * len(query))(*query)
    ref_len = ctypes.c_int(int(len(ref) / dim))
    query_len = ctypes.c_int(int(len(query) / dim))
    c_dim = ctypes.c_int(dim)
    res = (ctypes.c_int * (k * int(len(query) / dim)))()
    k = ctypes.c_int(k)

    testlib.knn_gpu.argtypes = [ctypes.POINTER(ctypes.c_float), 
                                ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                ctypes.POINTER(ctypes.c_int)]

    testlib.knn_gpu(ref_arr, ref_len, query_arr, query_len, c_dim, k, res)
    
    return res

