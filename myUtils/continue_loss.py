import torch
# 计算连续性方程  delta u/delta x + delta v/delta y = 0 的loss
def continue_loss(u_matrix, v_matrix, n=2):
    size = u_matrix.size()
    vy_result = (v_matrix[:,0:size[1]-n, :] - v_matrix[:,n:size[1], :]) / n
    ux_result = (u_matrix[:,:, 0:size[2]-n] - u_matrix[:,:, n:size[2]]) / n
    loss = torch.mean((vy_result[: ,:, 1:-1] + ux_result[: ,1:-1, :])**2)
    return loss