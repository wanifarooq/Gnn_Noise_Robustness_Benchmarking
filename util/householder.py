import torch

def torch_householder_orgqr(A: torch.Tensor) -> torch.Tensor:
    batch_shape = A.shape[:-2]
    d, r = A.shape[-2], A.shape[-1]
    
    # Normalizza le colonne come fa householder.py
    eps = 1e-12
    param = A / torch.linalg.norm(A, dim=-2, keepdim=True).clamp(min=eps)
    
    # Eye è (d, r) come nel C++
    eye = torch.zeros(*batch_shape, d, r, dtype=A.dtype, device=A.device)
    eye[..., :r, :r] = torch.eye(r, dtype=A.dtype, device=A.device)
    
    m = eye.clone()
    
    # Loop inverso come nel C++
    for j in range(r-1, -1, -1):
        u = param[..., :, j:j+1]        # (batch, d, 1)
        uT = u.transpose(-1, -2)         # (batch, 1, d)
        cx = torch.matmul(uT, m)         # (batch, 1, r)
        m = m + torch.matmul(-2 * u, cx) # m - 2*u*(u^T*m)
    
    return m