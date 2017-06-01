def repr_blobs_shape(blobs):
    res = []
    for b in blobs:
        if b is not None: 
            res.append('x'.join(map(str, b.shape)))
        else:
            res.append('null')
    return ','.join(res)
