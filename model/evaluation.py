import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
from scipy.linalg import svd
from scipy.sparse.linalg import eigsh

class OversmoothingMetrics:
    
    def __init__(self, device='cpu'):
        self.device = device
        
    def compute_all_metrics(self, X, edge_index, edge_weight=None, batch_size=None, graphs_in_class=None):

        metrics = {}
        
        X_np = X.detach().cpu().numpy()

        try:
            metrics['EDir'] = self._compute_edir_average(graphs_in_class)
        except Exception as e:
            print(f"Warning: Could not compute EDir: {e}")
            metrics['EDir'] = 0.0

        try:
            metrics['NumRank'] = self._compute_numerical_rank(X_np)
        except Exception as e:
            print(f"Warning: Could not compute NumRank: {e}")
            metrics['NumRank'] = float(min(X_np.shape))

        try:
            metrics['Erank'] = self._compute_effective_rank(X_np)
        except Exception as e:
            print(f"Warning: Could not compute Erank: {e}")
            metrics['Erank'] = float(min(X_np.shape))

        try:
            metrics['EDir_traditional'] = self._compute_dirichlet_energy_traditional(X, edge_index, edge_weight)
        except Exception as e:
            print(f"Warning: Could not compute EDir_traditional: {e}")
            metrics['EDir_traditional'] = 0.0

        try:
            metrics['EProj'] = self._compute_projection_energy(X, edge_index, edge_weight)
        except Exception as e:
            print(f"Warning: Could not compute EProj: {e}")
            metrics['EProj'] = 0.0

        try:
            metrics['MAD'] = self._compute_mad(X, edge_index)
        except Exception as e:
            print(f"Warning: Could not compute MAD: {e}")
            metrics['MAD'] = 0.0
        
        return metrics
    
    def _compute_edir_average(self, graphs_in_class):
        if not graphs_in_class or len(graphs_in_class) == 0:
            return 0.0
        
        total_energy = 0.0
        num_graphs = len(graphs_in_class)
        
        for graph_data in graphs_in_class:
            X = graph_data['X']
            edge_index = graph_data['edge_index']
            edge_weight = graph_data.get('edge_weight', None)
            
            if edge_index.size(1) == 0:
                continue
            
            graph_energy = 0.0
            num_edges = edge_index.size(1)
            
            for i in range(num_edges):
                u, v = edge_index[0, i], edge_index[1, i]
                
                grad = X[u] - X[v]
                
                edge_energy = torch.norm(grad, p=2)**2

                if edge_weight is not None:
                    edge_energy *= edge_weight[i]
                
                graph_energy += edge_energy.item()
            
            total_energy += graph_energy

        return total_energy / (2 * num_graphs)
    
    def _compute_numerical_rank(self, X):

        frobenius_norm_sq = np.sum(X**2)
        
        try:

            s = svd(X, full_matrices=False, compute_uv=False)
            spectral_norm_sq = s[0]**2 if len(s) > 0 else 1e-8
        except:
            spectral_norm_sq = np.linalg.norm(X, ord=2)**2
            
        if spectral_norm_sq < 1e-12:
            return 1.0
            
        return frobenius_norm_sq / spectral_norm_sq
    
    def _compute_effective_rank(self, X):

        try:

            s = svd(X, full_matrices=False, compute_uv=False)
            
            s = s[s > 1e-12]
            
            if len(s) == 0:
                return 1.0
                
            p = s / np.sum(s)

            p = p[p > 1e-12]
            entropy = -np.sum(p * np.log(p))
            
            return np.exp(entropy)
            
        except:
            return float(min(X.shape))
    
    def _compute_message_passing_matrix_eigenvector(self, edge_index, num_nodes, edge_weight=None):
        try:
            adj = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]
            adj_np = adj.cpu().numpy()
            
            adj_sym = (adj_np + adj_np.T) / 2
            
            try:
                eigenvalues, eigenvectors = eigsh(adj_sym, k=1, which='LA', maxiter=1000)
                dominant_eigenvector = np.abs(eigenvectors[:, 0])

                dominant_eigenvector = np.maximum(dominant_eigenvector, 1e-4)
                
                return dominant_eigenvector
            except:
                degrees = np.array(adj_sym.sum(axis=1)).flatten()

                return np.maximum(degrees, 1e-4)
                
        except Exception as e:
            return np.ones(num_nodes)
    
    def _compute_dirichlet_energy_traditional(self, X, edge_index, edge_weight=None):

        num_nodes = X.size(0)

        u = self._compute_message_passing_matrix_eigenvector(edge_index, num_nodes, edge_weight)
        
        # Normalization
        u = u / np.max(u)
        u = np.maximum(u, 1e-4)
        
        total_energy = 0.0
        num_edges = edge_index.size(1)
        
        for i in range(num_edges):
            node_i, node_j = edge_index[0, i].item(), edge_index[1, i].item()
            
            X_i_norm = X[node_i] / u[node_i]
            X_j_norm = X[node_j] / u[node_j]
            
            diff = X_i_norm - X_j_norm
            energy = torch.norm(diff, p=2)**2
            
            if edge_weight is not None:
                energy *= edge_weight[i]
                
            total_energy += energy.item()
            
        return total_energy
    
    def _compute_projection_energy(self, X, edge_index, edge_weight=None):

        num_nodes = X.size(0)
        
        u = self._compute_message_passing_matrix_eigenvector(edge_index, num_nodes, edge_weight)
        u = torch.tensor(u, device=X.device, dtype=X.dtype).unsqueeze(1)
        
        # Normalization
        u = u / torch.norm(u)

        P = torch.mm(u, u.t())
        
        PX = torch.mm(P, X)

        diff = X - PX
        energy = torch.norm(diff, p='fro')**2
        
        return energy.item()
    
    def _compute_mad(self, X, edge_index):

        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            return 0.0
            
        total_distance = 0.0
        
        for i in range(num_edges):
            node_i, node_j = edge_index[0, i], edge_index[1, i]
            
            X_i, X_j = X[node_i], X[node_j]
            
            norm_i = torch.norm(X_i, p=2)
            norm_j = torch.norm(X_j, p=2)
            
            if norm_i > 1e-8 and norm_j > 1e-8:
                cosine_sim = torch.dot(X_i, X_j) / (norm_i * norm_j)

                cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
                distance = 1.0 - cosine_sim
            else:
                distance = 1.0
            if isinstance(distance, float):
                total_distance += distance
            else:
                total_distance += distance.item()
            
        return total_distance / num_edges
    
    def evaluate_model_oversmoothing(self, model, data, device='cpu'):

        model.eval()
        
        with torch.no_grad():
            data = data.to(device)
            
            if hasattr(model, 'get_embeddings'):
                embeddings = model.get_embeddings(data.x, data.edge_index, data.edge_attr)
            else:
                embeddings = model(data.x, data.edge_index, data.edge_attr)
                
            metrics = self.compute_all_metrics(
                X=embeddings,
                edge_index=data.edge_index,
                edge_weight=getattr(data, 'edge_attr', None)
            )
            
        return metrics
    
    def print_metrics(self, metrics):

        print("Oversmoothing metrics:")
        print(f"NumRank: {metrics['NumRank']:.6f}")
        print(f"Erank: {metrics['Erank']:.6f}")
        print(f"Dirichlet energy: {metrics['EDir']:.6f}")
        print(f"Traditional Dirichlet energy: {metrics['EDir_traditional']:.6f}")
        print(f"EProj: {metrics['EProj']:.6f}")
        print(f"MAD: {metrics['MAD']:.6f}")
        print("-"*50)