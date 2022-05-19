import numpy as np

def NMF(V, W, fixW=True, thresh = 0.001, L = 1000):
    K, R = W.shape
    K, N = V.shape
    H = np.random.rand(R, N)

    eps_machine = np.finfo(np.float32).eps

    W_diff = float('inf')
    for _ in range(L):
        H_prev = H
        H = H * (W.transpose().dot(V) / (W.transpose().dot(W).dot(H) + eps_machine))
        H_diff = np.linalg.norm(H - H_prev, ord=2)
        if not fixW:
            W_prev = W
            W = W * (V.dot(H.transpose()) / (W.dot(H).dot(H.transpose()) + eps_machine))
            W_diff = np.linalg.norm(W - W_prev, ord=2)
        if H_diff < thresh and W_diff < thresh:
            break

    V_approx = W.dot(H)
    V_approx_err = np.linalg.norm(V - V_approx, ord=2)
    #print(f"NMF finished with V_approx_err={V_approx_err}\n")
    return V_approx, W, H