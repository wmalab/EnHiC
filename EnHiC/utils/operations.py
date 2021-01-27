import numpy as np
import gzip
import os


def scn_normalization(X, max_iter=1000, eps=1e-6, copy=True):
    m, n = X.shape
    if m != n:
        raise ValueError
    if copy:
        X = X.copy()
    X = np.asarray(X)
    X = X.astype(float)
    D = np.ones((X.shape[0],))
    for it in np.arange(max_iter):
        # sqrt(sqrt(sum(X.^2, 1))).^(-1)
        square = np.multiply(X, X)
        # sss_row and sss_col should be equal because of sysmmetry
        sss_row = np.sqrt(np.sqrt(square.sum(axis=-1)))
        sss_row[sss_row == 0] = 1
        sss_row = sss_row**(-1)

        # sss_col = np.sqrt(np.sqrt(square.sum(axis=-2)))
        # sss_col[sss_col == 0] = 1
        # sss_col = sss_col**(-1)

        sss_col = sss_row
        # D*X*D
        # next_X = np.diag(sss_row)@X@np.diag(sss_col)
        next_X = (sss_row*(X*sss_col).T).T
        D = sss_row * D

        if np.abs(X - next_X).sum() < eps:
            print("break at iteration %d" % (it,))
            break
        X = next_X
    return X, D


def scn_recover(normX, D):
    # recover matrix from scn_normalization
    # normX, D = scn_normalization(X, max_iter=1000, eps=1e-10, copy=True)
    # X = scn_recover(normX, D)
    return ((D**-1)*(normX*(D**-1)).T).T


def check_scn(X, normX, D):
    recover = scn_recover(normX, D)
    print("sum of matrix: ", X.sum())
    print("sum of recover: ", recover.sum())
    print(
        "diff abs ratio (recover - X)/x: {:6.4f}%".format((np.abs(recover-X)).sum()/X.sum()*100))
    print("sum of axis0: ", (normX**2).sum(axis=0))
    print("sum of axis1: ", (normX**2).sum(axis=1))


def redircwd_back_projroot(project_name='EnHiC'):
    root = os.getcwd().split('/')
    for i, f in enumerate(root):
        if f == project_name:
            root = root[:i+1]
            break
    root = '/'.join(root)
    os.chdir(root)
    print('current working directory: ', os.getcwd())
    return root


def sampling_hic(hic_matrix, sampling_ratio, fix_seed=False):
    """sampling dense hic matrix"""
    m = np.matrix(hic_matrix)
    all_sum = m.sum(dtype='float')
    idx_prob = np.divide(m, all_sum, out=np.zeros_like(m), where=all_sum != 0)
    idx_prob = np.asarray(idx_prob.reshape(
        (idx_prob.shape[0]*idx_prob.shape[1],)))
    idx_prob = np.squeeze(idx_prob)
    sample_number_counts = int(all_sum/(2*sampling_ratio))
    id_range = np.arange(m.shape[0]*m.shape[1])
    if fix_seed:
        np.random.seed(0)
    id_x = np.random.choice(
        id_range, size=sample_number_counts, replace=True, p=idx_prob)
    sample_m = np.zeros_like(m)
    for i in np.arange(sample_number_counts):
        x = int(id_x[i]/m.shape[0])
        y = int(id_x[i] % m.shape[0])
        sample_m[x, y] += 1.0
    sample_m = np.transpose(sample_m) + sample_m
    return np.asarray(sample_m)


def divide_pieces_hic(hic_matrix, block_size=128, max_distance=None, save_file=False, pathfile=None):
    # max_boundary
    if max_distance is not None:
        max_distance = int(max_distance/(block_size/2.0))
    M = hic_matrix

    IMG_HEIGHT, IMG_WIDTH = int(block_size), int(block_size)
    print('Height: ', IMG_HEIGHT, 'Weight: ', IMG_WIDTH)
    M_h, M_w = M.shape
    block_height = int(IMG_HEIGHT/2)
    block_width = int(IMG_WIDTH/2)
    M_d0 = np.split(M, np.arange(block_height, M_h, block_height), axis=0)
    M_d1 = list(map(lambda x: np.split(x, np.arange(block_width, M_w, block_width), axis=1), M_d0))
    hic_half_h = np.array(M_d1)
    if M_h % block_height != 0 or M_w % block_width != 0:
        hic_half_h = hic_half_h[0:-1, 0:-1]
    print('shape of blocks: ', hic_half_h.shape)

    hic_m = list()
    hic_index = dict()
    hic_index_rev = dict()
    count = 0
    for dis in np.arange(1, hic_half_h.shape[0]):
        if (max_distance is not None) and (dis > max_distance):
            break
        for i in np.arange(0, hic_half_h.shape[1]-dis):
            hic_m.append(np.block([[hic_half_h[i, i], hic_half_h[i, i+dis]],
                                   [hic_half_h[i+dis, i], hic_half_h[i+dis, i+dis]]]))
            hic_index[count] = (i, i+dis)
            hic_index_rev[(i, i+dis)] = count
            count = count + 1
    print('# of hic pieces: ', len(hic_m))

    if save_file:
        from numpy import savez_compressed, savez
        if pathfile is None:
            pathfile = './datasets_hic'
        savez_compressed(pathfile+'.npz', hic=hic_m,
                         index_1D_2D=hic_index, index_2D_1D=hic_index_rev)

    return hic_m, hic_index, hic_index_rev


def merge_hic(hic_lists, index_1D_2D, max_distance=None):
    # hic_lists: pieces of hic matrix
    # index_1D_2D: index of matrix from list to 2D(x,y)
    # max_distance: if hic_lists is not for whole matrix, the max_distance is required,
    #   max_distance = max_genomic_distance/resolution
    hic_m = np.asarray(hic_lists)
    lensize, Height, Width = hic_m.shape
    lenindex = len(index_1D_2D)
    print('lenindex: ', lenindex)
    if lenindex != lensize:
        raise 'ERROR dimension must equal. length of hic list: ' + lensize + \
            'is not equal to length of index_1D_2D: ' + len(index_1D_2D)

    Height_hf = int(Height/2)
    Width_hf = int(Width/2)

    if max_distance is None:
        if 2*lenindex != int(np.sqrt(2*lenindex))*(int(np.sqrt(2*lenindex))+1):
            raise 'ERROR: not square'
        n = int(np.sqrt(2*lenindex)+1)
    else:
        k = np.ceil(max_distance/Height_hf)
        n = int((lensize+(1+k)*k/2)/k)

    matrix = np.zeros(shape=(n*Height_hf, n*Width_hf))
    dig = np.zeros(shape=(n,))
    for i in np.arange(lenindex):
        h, w = index_1D_2D[i]
        '''if (max_distance is not None) and (np.abs(h-w) > np.ceil(max_distance/Height_hf)):
            continue'''
        x = h*Height_hf
        y = w*Width_hf
        matrix[x:x+Height_hf, y:y+Width_hf] += hic_m[i, 0:Height_hf, 0+Width_hf:Width]
        if abs(h-w)==1:
            dig[h] += 1
            dig[w] += 1
            matrix[x:x+Height_hf, x:x+Height_hf] += hic_m[i, 0:Height_hf, 0:Width_hf]
            matrix[y:y+Width_hf, y:y+Width_hf] += hic_m[i, 0+Height_hf:Height, 0+Width_hf:Width]
    
    matrix = matrix + np.transpose(matrix)

    for i in np.arange(0,n):
        matrix[i*Height_hf:(i+1)*Height_hf, i*Height_hf:(i+1)*Height_hf] /= (2.0*dig[i])

    return matrix


def filter_diag_boundary(hic, diag_k=1, boundary_k=None):
    if boundary_k is None:
        boundary_k = hic.shape[0]-1
    filter_m = np.tri(N=hic.shape[0], k=boundary_k)
    filter_m = np.triu(filter_m, k=diag_k)
    filter_m = filter_m + np.transpose(filter_m)
    return np.multiply(hic, filter_m)


'''def dense2tag(matrix):
    """converting a square matrix (dense) to coo-based tag matrix"""
    Height, Width = matrix.shape
    tag_mat = list()
    for i in np.arange(Height):
        for j in np.arange(i+1, Width):
            if float(matrix[i, j]) > 1.0e-20:
                tag_mat.append([int(i), int(j), float(matrix[i, j])])
    tag_mat = np.asarray(tag_mat, dtype=np.float)
    return tag_mat


def tag2dense(tag_mat, mat_length):
    """converting a tag matrix to square matrix (dense)"""
    Height, Width = int(mat_length), int(mat_length)
    matrix = np.zeros(shape=(Height, Width))

    for i in np.arange(len(tag_mat)):
        x, y, c = tag_mat[i]
        matrix[int(x), int(y)] = float(c)

    return tag_mat'''

# def format_contact(matrix, coordinate=(0, 1), resolution=10000, chrm='1', save_file=True, filename=None):
def format_contact(matrix, resolution=10000, chrm='1', save_file=True, filename=None):
    """chr1 bin1 chr2 bin2 value"""
    n = len(matrix)
    #nhf = np.floor(n/2)
    contact = list()
    for i in np.arange(n):
        for j in np.arange(i+1, n):
            value = float(matrix[i, j])
            if value <= 1.0e-10:
                continue
            chr1 = chrm
            chr2 = chrm
            # print('i: {}, j: {}, nhf: {}, int(i/nhf): {}, int(j/nhf): {}'.format(i, j, nhf, int(i/nhf), int(j/nhf)))
            # bin1 = (i - int(i/nhf)*nhf + coordinate[int(i/nhf)]*nhf)*resolution
            # bin2 = (j - int(j/nhf)*nhf + coordinate[int(j/nhf)]*nhf)*resolution
            bin1 = i*resolution
            bin2 = j*resolution
            entry = [chr1, str(bin1), chr2, str(bin2), str(value)]
            contact.append('\t'.join(entry))
    contact_txt = "\n".join(contact)
    #contact_txt = format(contact_txt, 'b')
    if save_file:
        if filename is None:
            filename = './demo_contact.gz'
        output = gzip.open(filename, 'w+')
        try:
            output.write(contact_txt.encode())
        finally:
            output.close()
    return contact

# def format_bin(matrix, coordinate=(0, 1), resolution=10000, chrm='1', save_file=True, filename=None):
def format_bin(matrix, resolution=10000, chrm='1', save_file=True, filename=None):
    """chr start end name"""
    n = len(matrix)
    # nhf = int(len(matrix)/2)
    bins = list()

    for i in np.arange(n):
        chr1 = 'chr' + chrm
        start = int(i*resolution)
        # start = int((i - int(i/nhf)*nhf + coordinate[int(i/nhf)]*nhf)*resolution)
        end = int(start + resolution)
        entry = [chr1, str(start), str(end), str(start)]
        bins.append('\t'.join(entry))
    if save_file:
        if filename is None:
            filename = './demo.bed.gz'
        file = gzip.open(filename, "w+")
        for l in bins:
            line = l + '\n'
            file.write(line.encode())
        file.close()
    return bins


def remove_zeros(matrix):
    idxy = ~np.all(np.isnan(matrix), axis=0)
    M = matrix[idxy, :]
    M = M[:, idxy]
    M = np.asarray(M)
    idxy = np.asarray(idxy)
    return M, idxy


if __name__ == '__main__':
    """m = np.ones(shape=(8,8))*2+ 400*np.diag(np.ones(shape=(8,)))
    print(m.sum())
    #print(m)
    sample = sampling_hic(m, sampling_ratio=2, fix_seed=True)
    print(sample.shape)
    print(sample.sum())
    print(sample)"""
    X = np.random.rand(8, 8)
    X = np.abs(X + X.T)
    np.set_printoptions(precision=4)
    print(X)
    normX, D = scn_normalization(X, max_iter=2000, eps=1e-6, copy=True)
    np.set_printoptions(precision=4)
    print(normX)
    print(scn_recover(normX, D))
    check_scn(X, normX, D)
