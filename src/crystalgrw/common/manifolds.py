import torch
import torch.nn.functional as F
from math import pi as PI
# import geomstats.backend as gs
# from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
# from geomstats.geometry.product_manifold import ProductManifold, ProductRiemannianMetric


class BaseManifold:
    def __init__(self):
        self.m_dim = None
        self.eps = None

    def get_tangent(self, x):
        pass

    def get_randn_tangent(self, x):
        pass

    def exp(self, tan_vec, x):
        pass

    def log(self, x_0, x_t):
        pass


class BaseSphere(BaseManifold):
    def __init__(self, manifold,
                 input_coord_type="intrinsic", 
                 data_range=(0,1)):
        self.input_coord_type = input_coord_type
        dim = [int(d[1]) for d in manifold.split("x")]
        if self.input_coord_type == "intrinsic":
            self.m_dim = sum(dim)
        elif self.input_coord_type == "extrinsic":
            self.m_dim = sum([d+1 for d in dim]) // len(dim)
        else:
            raise NotImplementedError

        self.mid = (data_range[1] + data_range[0]) / 2
        self.margin = (data_range[1] - data_range[0])
        self.algo = "score"
    
    def _scale(self, r):
        r"""
        return: r \in [-pi,pi]
        """
        return 2*PI * (r - self.mid) / self.margin
    
    def _rescale(self, r):
        r"""
        return: r \in [min,max]
        """
        #return self.max * (r - self.min) - PI
        return (r / (2*PI)) * self.margin + self.mid

    def _assert_manifold_dim(self, r):
        assert r.shape[-1] == self.m_dim, \
        f"{self.input_coord_type} coord requires {self.m_dim} for last dim."
        
    def _to_extrinsic(self, r):
        self._assert_manifold_dim(r)
        if self.input_coord_type == "intrinsic":
            r = self._scale(r)
            x = torch.cos(r)
            y = torch.sin(r)
            r = torch.stack([x, y], dim=-1)
        # elif self.input_coord_type == "extrinsic":
        #     return r
        return r

    def _to_intrinsic(self, r):
        self._assert_manifold_dim(r)
        r = self._scale(r)
        # if self.input_coord_type == "intrinsic":
        #     return r
        if self.input_coord_type == "extrinsic":
            r = torch.atan2(r[...,1], r[...,0])
        return r

    def _return_coord(self, r):
        if self.input_coord_type == "intrinsic":
            r = torch.atan2(r[...,1], r[...,0])
        # elif self.input_coord_type == "extrinsic":
        #     return r
        return self._rescale(r)
        

class Torus3d(BaseSphere):
    def __init__(self, input_coord_type="intrinsic", scale=(0,1)):
        super().__init__("s1xs1xs1", input_coord_type)
        self.input_coord_type = input_coord_type

    def get_tangent(self, r):
        """
        r: [num_nodes x 3 x coord_type_dim]
        return tan_vec: [num_nodes x 3 x 2]
        """
        r = self._to_extrinsic(r)
        tan_vec = torch.stack((-r[...,1], r[...,0]), dim=-1)
        tan_vec = F.normalize(tan_vec, p=2, dim=-1)
        return tan_vec  #.to(r.device)

    def exp(self, tan_vec, r):
        """
        tan_vec: [num_nodes x 3 x 2]
        r: [num_nodes x 3 x coord_type_dim]
        return r: [num_nodes x 3 x 2]
        """
        norm = torch.norm(tan_vec, p=2, dim=-1, keepdim=True)
        r = self._to_extrinsic(r)
        r = torch.cos(norm) * r + torch.sin(norm) * tan_vec/(norm + 1e-10)
        return self._return_coord(r)

    def log(self, r_0, r_t):
        r_0 = self._to_intrinsic(r_0)
        r_t = self._to_intrinsic(r_t)
        r_inv = r_0 - r_t
        upper = torch.where(r_inv > PI)
        lower = torch.where(r_inv < -PI)
        r_inv[upper] = r_inv[upper] - 2*PI 
        r_inv[lower] = r_inv[lower] + 2*PI
        # r_inv = r_inv / (2*PI)
        return r_inv


class TorusNd(BaseSphere):
    def __init__(self, input_coord_type="intrinsic", scale=(0,1), torus_dim=3):
        super().__init__("x".join(["s1" for _ in range(torus_dim)]), input_coord_type)
        self.input_coord_type = input_coord_type

    def get_tangent(self, r):
        """
        r: [num_nodes x m_dim x coord_type_dim]
        return tan_vec: [num_nodes x m_dim x 2]
        """
        r = self._to_extrinsic(r)
        tan_vec = torch.stack((-r[...,1], r[...,0]), dim=-1)
        tan_vec = F.normalize(tan_vec, p=2, dim=-1)
        return tan_vec  #.to(r.device)

    def exp(self, tan_vec, r):
        """
        tan_vec: [num_nodes x m_dim x 2]
        r: [num_nodes x m_dim x coord_type_dim]
        return r: [num_nodes x m_dim x 2]
        """
        norm = torch.norm(tan_vec, p=2, dim=-1, keepdim=True)
        r = self._to_extrinsic(r)
        r = torch.cos(norm) * r + torch.sin(norm) * tan_vec/(norm + 1e-10)
        return self._return_coord(r)

    def log(self, r_0, r_t):
        r_0 = self._to_intrinsic(r_0)
        r_t = self._to_intrinsic(r_t)
        r_inv = r_0 - r_t
        upper = torch.where(r_inv > PI)
        lower = torch.where(r_inv < -PI)
        r_inv[upper] = r_inv[upper] - 2*PI
        r_inv[lower] = r_inv[lower] + 2*PI
        # r_inv = r_inv / (2*PI)
        return r_inv


class Simplex1d(BaseManifold):
    def __init__(self, scale=(0,1), bc="reflection"):
        self.bc = bc
        self.min = scale[0]
        self.max = scale[1]
        self.m_dim = 1

    def _scale(self, r):
        return (r - self.min) / self.max

    def _rescale(self, r):
        return r * self.max + self.min

    @staticmethod
    def get_tangent(r):
        return torch.ones_like(r)

    def exp(self, tan_vec, r): 
        r = self._scale(r)
        r = r + tan_vec
        if self.bc == "reflection":
            r = r % 2
            r[r>1] = 2 - r[r>1]
        elif self.bc == "absorption":
            r[torch.where(r<=0)[0]] = 0
            r[torch.where(r>=1.)[0]] = 1
        else:
            raise NotImplementedError
        return self._rescale(r)
    
    def log(self, r_0, r_t):
        r_0 = self._scale(r_0)
        r_t = self._scale(r_t)
        assert r_0.shape == r_t.shape
        r_inv = r_0 - r_t
        return r_inv


class Euclid3d(BaseManifold):
    def __init__(self, algo="ve"):
        self.algo = algo
        self.m_dim = 3

    @staticmethod
    def get_tangent(r):
        return torch.ones_like(r)

    def get_randn_tangent(self, r):
        self.eps = torch.randn_like(r)
        return self.eps
    
    @staticmethod
    def exp(tan_vec, r):
        return r + tan_vec
    
    def log(self, r_0, r_t):
        if self.algo == "ve":
            return r_0 - r_t
        elif self.algo == "vp":
            return self.eps


class EuclidNd(BaseManifold):
    def __init__(self, algo="ve", euclid_dim=3):
        self.algo = algo
        self.m_dim = euclid_dim

    @staticmethod
    def get_tangent(r):
        return torch.ones_like(r)

    def get_randn_tangent(self, r):
        self.eps = torch.randn_like(r)
        return self.eps

    @staticmethod
    def exp(tan_vec, r):
        return r + tan_vec

    def log(self, r_0, r_t):
        if self.algo == "ve":
            return r_0 - r_t
        elif self.algo == "vp":
            return self.eps


class Hypercube(BaseManifold):
    def __init__(self, hpc_dim):
        self.m_dim = hpc_dim

    def get_tangent(self, x):
        assert x.size(-1) == self.m_dim
        return torch.ones_like(x)

    def get_randn_tangent(self, x):
        assert x.size(-1) == self.m_dim
        self.eps = torch.randn_like(x)
        return self.eps

    def exp(self, tan_vec, x):
        assert x.size(-1) == self.m_dim
        x += tan_vec
        x = x % 2
        x[x > 1.] = 2 - x[x > 1.]
        return x

    def log(self, x_0, x_t):
        assert x_0.size(-1) == self.m_dim
        assert x_t.size(-1) == self.m_dim
        return x_0 - x_t

    def simp_from_hpc(self, hpc, return_indices=False):
        assert hpc.shape[-1] == self.m_dim

        sorted_indices = torch.argsort(hpc, dim=-1)
        z = torch.sort(hpc, dim=-1).values
        n = z.shape[-1]
        x = torch.zeros(z.shape[0], n + 1, device=hpc.device)
        x[:, 0] = z[:, 0]

        for i in range(1, n):
            x[:, i] = z[:, i] - z[:, i - 1]
        x[:, n] = 1 - z[:, n - 1]

        if return_indices:
            return x, sorted_indices
        else:
            return x

    def simp_to_hpc(self, x, sorted_indices=None):
        assert x.shape[-1] == self.m_dim + 1
        assert torch.allclose(x.sum(-1), torch.ones(x.size(0), device=x.device)), \
            f"Sum of simplex coordinates should close to 1, but it is {x.sum(-1)}."

        if sorted_indices is None:
            sorted_indices = torch.arange(x.shape[-1] - 1).unsqueeze(0).repeat(x.size(0), 1)

        num_nodes = x.shape[0]
        z = torch.zeros(num_nodes, self.m_dim).to(x.device)
        z[:, 0] = x[:, 0]

        for i in range(1, self.m_dim):
            z[:, i] = z[:, i-1] + x[:, i]
        hpc = torch.zeros(num_nodes, self.m_dim).to(x.device)

        for i in range(num_nodes):
            hpc[i, sorted_indices[i]] = z[i]
        return hpc


class CombinedManifold(BaseManifold):
    def __init__(self, manifolds, kwarg_list):
        import itertools
        self.manifolds = [m(**kwargs) for m, kwargs in zip(manifolds, kwarg_list)]
        self.m_dims = list(itertools.accumulate([0] + [m.m_dim for m in self.manifolds]))
        self.m_dims = [(di, df) for di, df in zip(self.m_dims[:-1], self.m_dims[1:])]
        self.m_dim = sum([m.m_dim for m in self.manifolds])
        if sum([isinstance(m, BaseSphere) for m in self.manifolds]) > 0:
            self.sphere_exists = True
        else:
            self.sphere_exists = False

    @staticmethod
    def _expand_dim(tensors):
        size, shape = max([(len(t.shape), t.shape) for t in tensors])
        for i, t in enumerate(tensors):
            for _ in range(size-len(t.shape)):
                t = t.unsqueeze(-1)
            tensors[i] = t.expand(*shape[:-2],-1,shape[-1])
        return tensors

    def get_tangent(self, x):
        tangent = []
        for m, (di, df) in zip(self.manifolds, self.m_dims):
            tangent.append(m.get_tangent(x[..., di:df]))
        return torch.cat(self._expand_dim(tangent), dim=-2)

    def get_randn_tangent(self, x):
        self.eps = torch.randn_like(x)
        return self.eps

    def exp(self, tan_vec, x):
        exp = []
        for m, (di, df) in zip(self.manifolds, self.m_dims):
            if self.sphere_exists:
                if isinstance(m, BaseSphere):
                    d = slice(0,2)
                else:
                    d = 0
                exp.append(m.exp(tan_vec[:, di:df, d], x[:, di:df]))
            else:
                exp.append(m.exp(tan_vec[..., di:df], x[..., di:df]))
        return torch.cat(exp, dim=-1)

    def log(self, x_0, x_t):
        log = []
        for m, (di, df) in zip(self.manifolds, self.m_dims):
            log.append(m.log(x_0[..., di:df], x_t[..., di:df]))
        return torch.cat(log, dim=-1)


class T2xD1(CombinedManifold):
    def __init__(self, torus_input_coord_type="intrinsic", torus_scale=(0,1),
                 simplex_scale=(0,1), simplex_bc="reflection"):
        manifolds = [TorusNd, Simplex1d]
        kwarg_list = [{"input_coord_type": torus_input_coord_type,
                       "scale": torus_scale,
                       "torus_dim": 2
                       },
                      {"scale": simplex_scale,
                       "bc": simplex_bc
                       },
                      ]
        super().__init__(manifolds, kwarg_list)
