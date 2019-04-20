
def setup_path(filepath):
    import os
    import pickle
    splits = filepath.split("/")
    currentDir = ""
    for i in range(len(splits) - 1):
        currentDir = currentDir + splits[i] + "/"
        if not os.path.exists(currentDir):
            os.makedirs(currentDir)
            print("create dir: " + currentDir)
        else:
            print("dir (" + currentDir + ") already exists")

def pca(mat, k = 2):
    """
    @input:
    - mat: 2D matrix of size (#record, #features)
    - k: top k component to extract
    @output:
    - 
    """
    import torch
    mean = torch.mean(mat, 0)
    X = mat - mean.expand_as(mat)
    U,S,V = torch.svd(X)
    print("U: " + str(U.shape))
    print("S: " + str(S.shape))
    print("V: " + str(V.shape))
    return U,S,V