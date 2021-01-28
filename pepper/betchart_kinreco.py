import numpy as np
import math
from numba import njit
from numba.experimental import jitclass
from numba import float32, float64
from functools import partial
from scipy.optimize import leastsq
from coffea.nanoevents.methods import vector
import awkward as ak


@njit(cache=True)
def unit_circle():
    '''Unit circle in extended representation.'''
    diag = np.array([1., 1., -1.])
    return np.diag(diag)


@njit(cache=True)
def cofactor(mat_a, ij):
    '''Cofactor[i,j] of 3x3 matrix a.'''
    i, j = ij
    iidx, jidx = [0, 1, 2], [0, 1, 2]
    iidx.pop(i)
    jidx.pop(j)
    mat_b = mat_a[np.array(iidx), :]
    mat_a = mat_b[:, np.array(jidx)]
    return (-1)**(i+j) * (mat_a[0, 0] * mat_a[1, 1]
                          - mat_a[1, 0] * mat_a[0, 1])


@njit(cache=True)
def rot_matrix(axis, angle):
    '''Rotation matrix about x,y, or z axis (0,1, or 2, respectively.)'''
    c, s = math.cos(angle), math.sin(angle)
    mat_r = c * np.eye(3)
    for i in [-1, 0, 1]:
        mat_r[(axis - i) % 3, (axis + i) % 3] = i * s + (1 - i * i)
    return mat_r


@njit(cache=True)
def multisqrt(y):
    '''Valid real solutions to y=x*x.'''
    if y < 0:
        return [float(x) for x in range(0)]
    elif y == 0:
        return [0.]
    else:
        sqrty = np.sqrt(y)
        return [-sqrty, sqrty]


@njit(cache=True)
def _maybe_swap(mat_g, swap_xy):
    if swap_xy:
        mat_q = mat_g[np.array([1, 0, 2]), :][:, np.array([1, 0, 2])]
        return mat_q
    else:
        return mat_g


@njit(cache=True)
def factor_degenerate(mat_g, zero=0):
    '''Linear factors of a degenerate quadratic polynomial.'''
    if mat_g[0, 0] == 0 == mat_g[1, 1]:
        return [[mat_g[0, 1], 0,       mat_g[1, 2]],
                [0,           mat_g[0, 1], mat_g[0, 2] - mat_g[1, 2]]]

    swap_xy = abs(mat_g[0, 0]) > abs(mat_g[1, 1])
    mat_q = _maybe_swap(mat_g, swap_xy)
    mat_q /= mat_q[1, 1]
    q22 = cofactor(mat_q.real, (2, 2))

    if -q22 <= zero:
        lines = [[mat_q[0, 1], mat_q[1, 1], mat_q[1, 2] + s]
                 for s in multisqrt(-cofactor(mat_q.real, (0, 0)))]
    else:
        x0, y0 = [cofactor(mat_q.real, (i, 2)) / q22 for i in [0, 1]]
        lines = [[m, mat_q[1, 1], -mat_q[1, 1]*y0 - m*x0]
                 for m in [mat_q[0, 1] + s for s in multisqrt(-q22)]]

    if swap_xy:
        lines = [[L[1],  L[0], L[2]] for L in lines]
    return lines


@njit(cache=True)
def intersections_ellipse_line(ellipse=None, line=None, zero=1e-6):
    '''Points of intersection between ellipse and line.'''
    _, V = np.linalg.eig(np.cross(line, ellipse).T)
    return [v.real / v[2].real for v in V.T if
            abs(np.dot(np.array(line).real, v.real)) < 10 ** -12 and
            abs(np.dot(v.real, ellipse.real).dot(v.real)) < 10 ** -12]


@njit(cache=True)
def intersections_ellipses(mat_a, mat_b, zero=1e-6):
    '''Points of intersection between two ellipses.'''
    mat_a = mat_a.astype(np.complex128)
    mat_b = mat_b.astype(np.complex128)
    if abs(np.linalg.det(mat_b)) > abs(np.linalg.det(mat_a)):
        mat_a, mat_b = mat_b, mat_a
    eigval = [e.real for e in
              np.linalg.eigvals(np.linalg.inv(mat_a).dot(mat_b))
              if np.abs(e.imag) <= zero][0]
    lines = factor_degenerate(mat_b - eigval * mat_a)
    points = [point for L in lines
              for point in intersections_ellipse_line(mat_a, L)]
    return points


nss_spec = {"D2": float32,
            "b": float64[:],
            "mu": float64[:],
            "mb2": float32,
            "pb": float32,
            "pmu": float32,
            "c": float32,
            "s": float32,
            "m": float32,
            "m_alt": float32,
            "x0": float32,
            "y0": float32,
            "x1": float32,
            "y1": float32,
            "Z": float32,
            "mw2": float32}


@jitclass(nss_spec)
class NuSolutionSet(object):
    '''Collection of definitions for neutrino analytic solution
       in t->b, mu, nu.'''

    def __init__(self, b, mu,  # arrays of pt, eta, phi, E components
                 mw2=80.385**2, mt2=172.5**2  # GeV**2
                 ):
        self.mu = mu
        self.b = b
        pb = np.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)
        pmu = np.sqrt(mu[0] ** 2 + mu[1] ** 2 + mu[2] ** 2)
        self.mb2 = b[3] ** 2 - pb ** 2
        self.D2 = 0.5 * (mt2 - mw2 - self.mb2)
        self.pb = pb
        self.pmu = pmu
        self.c = (b[0] * mu[0] + b[1] * mu[1] + b[2] * mu[2]) / (pb * pmu)
        if self.c > 1:
            print("cos exceeds allowed values, set to 1 - calculated val:",
                  self.c)
            self.c = 1
        self.s = math.sqrt(1 - self.c ** 2)
        self.x0 = -0.5 * mw2 / pmu
        self.y0 = - (self.x0 * self.c + self.D2 / pb) / self.s

        self.m = (np.abs(self.b[3]) / pb - self.c) / self.s
        self.m_alt = (- np.abs(self.b[3]) / pb - self.c) / self.s
        self.y1 = - self.x0 / self.m
        self.x1 = self.x0 + (self.y1 - self.y0) / self.m

        self.Z = math.sqrt(max(0, (self.y1 * (self.y1 - 2 * self.y0)
                                   - self.x0 ** 2 - mw2)))
        self.mw2 = mw2

    @property
    def mat_k(self):
        '''Extended rotation from F' to F coordinates.'''
        return np.array([[self.c, -self.s, 0., 0.],
                         [self.s,  self.c, 0., 0.],
                         [0.,      0.,     1., 0.],
                         [0.,      0.,     0., 1.]])

    @property
    def mat_a_mu(self):
        '''F coordinates constraint on W momentum: paraboloid.'''
        return np.array([[0.,      0., 0.,                 self.x0],
                         [0.,      1., 0.,                      0.],
                         [0.,      0., 1.,                      0.],
                         [self.x0, 0., 0., self.mw2 - self.x0 ** 2]])

    @property
    def mat_a_b(self):
        '''F coordinates constraint on W momentum: ellipsoid.'''
        mat_k, D2, mw2, e2 = self.mat_k, self.D2, self.mw2, self.b[3]**2
        return mat_k.dot(np.array(
            [[self.mb2,      0., 0.,     - D2 * self.pb],
             [0.,            e2, 0.,                 0.],
             [0.,            0., e2,                 0.],
             [-D2 * self.pb, 0., 0., e2 * mw2 - D2 ** 2]])).dot(mat_k.T)

    @property
    def rot_t(self):
        '''Rotation from F coordinates to laboratory coordinates.'''
        mu = self.mu
        mutheta = np.arctan2(np.sqrt(mu[0] ** 2 + mu[1] ** 2), mu[2])
        muphi = np.arctan2(mu[1], mu[0])
        R_z = rot_matrix(2, - muphi)
        R_y = rot_matrix(1,  0.5 * math.pi - mutheta)
        R_x = [rot_matrix(0, - math.atan2(z, y))
               for x, y, z in (R_y.dot(R_z.dot(self.b[0:3])), )][0]
        return R_z.T.dot(R_y.T.dot(R_x.T))

    @property
    def mat_e_tilde(self):
        '''Transformation of t=[c,s,1] to nu momentum: F coordinates.'''
        Z, m, x1, y1, p = self.Z, self.m, self.x1, self.y1, self.pmu
        return np.array([[Z/m, 0.,   x1 - p],
                         [Z,   0.,       y1],
                         [0.,   Z,       0.]])

    @property
    def mat_e(self):
        '''Transformation of t=[c,s,1] to nu momentum: lab coordinates.'''
        return self.rot_t.dot(self.mat_e_tilde)

    @property
    def mat_e_perp(self):
        '''Transformation of t=[c,s,1] to nu transverse momentum:
           lab coordinates.'''
        return np.vstack((self.mat_e[:2], np.array([[0., 0., 1.]])))

    @property
    def nu(self):
        '''Solution ellipse of nu transverse momentum: lab coordinates.'''
        E = self.mat_e_perp
        return np.linalg.inv(E.T).dot(unit_circle()).dot(np.linalg.inv(E))


@njit(cache=True)
def nus(ts, es):
    ret_nus = []
    for e, t in zip(es, ts):
        ret_nus.append(e.dot(np.array([math.cos(t), math.sin(t), 1.])))
    return ret_nus


@njit(cache=True)
def test_func(es, met, params):
    ret_xy = - met[:2]
    for nu in nus(params, es):
        ret_xy += nu[:2]
    return ret_xy


def _run_kinreco(lep, alep, b, bbar, met_x, met_y,
                 mw2=80.385**2, mt2=172.5**2):
    nevents = len(lep)
    nu = np.empty((3, nevents, 4), dtype=np.float64)
    nubar = np.empty((3, nevents, 4), dtype=np.float64)
    fails = 0
    sols = np.zeros((nevents, 4), dtype=bool)
    for eventi in range(nevents):
        bs = (b[eventi], bbar[eventi])
        mus = (alep[eventi], lep[eventi])
        metX, metY = met_x[eventi], met_y[eventi]
        solutionSets = [NuSolutionSet(b, mu, mw2, mt2)
                        for b, mu in zip(bs, mus)]

        V0 = np.outer([metX, metY, 0.], [0., 0., 1.])
        S = V0 - unit_circle()

        hasZs = solutionSets[0].Z != 0 and solutionSets[1].Z != 0
        if hasZs:
            N, N_ = [ss.nu for ss in solutionSets]
            n_ = S.T.dot(N_).dot(S)

            v = intersections_ellipses(N, n_)
            v_ = [S.dot(sol) for sol in v]

            if not v and leastsq:
                es = [ss.mat_e_perp for ss in solutionSets]
                met = np.array([metX, metY, 1])
                ts, _ = leastsq(partial(test_func, es, met), [0., 0.],
                                ftol=5e-5, epsfcn=0.01)
                v, v_ = [[i] for i in nus(ts, es)]

            K, K_ = [ss.mat_e.dot(np.linalg.inv(ss.mat_e_perp))
                     for ss in solutionSets]
            for sol_i, s in enumerate(v):
                for i in [0, 1, 2]:
                    nu[i, eventi, sol_i] = K.dot(s)[i]
                    sols[eventi, sol_i] = True
            for sol_i, s_ in enumerate(v_):
                for i in [0, 1, 2]:
                    nubar[i, eventi, sol_i] = K_.dot(s_)[i]
        else:
            fails += 1
    return nu, nubar, sols


def _make_2d_array(lv):
    return np.stack((ak.to_numpy(lv.x), ak.to_numpy(lv.y),
                     ak.to_numpy(lv.z), ak.to_numpy(lv.t)), axis=-1)


def _lorvecfromnumpy(x, y, z, t, behavior, name="LorentzVector"):
    x = ak.Array(x)
    y = ak.Array(y)
    z = ak.Array(z)
    t = ak.Array(t)
    # Make axes to outer layer
    counts = []
    for i in range(x.ndim - 1):
        counts.append(ak.num(x))
        x = ak.flatten(x)
        y = ak.flatten(y)
        z = ak.flatten(z)
        t = ak.flatten(t)
    v = ak.Array({"x": x, "y": y, "z": z, "t": t}, with_name=name,
                 behavior=behavior)
    for count in reversed(counts):
        v = ak.unflatten(v, count)
    return v


def betchart(lep, antilep, b, antib, met, mw=80.385, mt=172.5):
    METx = met.pt * np.cos(met.phi)
    METy = met.pt * np.sin(met.phi)

    v, vb, sols = _run_kinreco(_make_2d_array(lep),
                               _make_2d_array(antilep),
                               _make_2d_array(b),
                               _make_2d_array(antib),
                               METx, METy)

    vE = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    v = _lorvecfromnumpy(v[0], v[1], v[2], vE, vector.behavior)
    v = v[ak.from_regular(sols, axis=1)]
    vbE = np.sqrt(vb[0] ** 2 + vb[1] ** 2 + vb[2] ** 2)
    vb = _lorvecfromnumpy(vb[0], vb[1], vb[2], vbE, vector.behavior)
    vb = vb[ak.from_regular(sols, axis=1)]

    wp = v + antilep
    wm = vb + lep
    t = wp + b
    at = wm + antib
    min_mtt = ak.argmin((t + at).mass, axis=1, keepdims=True)
    t = t[min_mtt]
    at = at[min_mtt]
    return t, at


if __name__ == '__main__':
    lep = _lorvecfromnumpy(
        [26.923591, 86.3662048], [16.170616, -4.14978429],
        [-162.3227, -125.30777765], [165.33320, 152.24451503], vector.behavior)
    antilep = _lorvecfromnumpy(
        [-34.58441, -44.03565424], [-13.27824, 57.64460697],
        [-32.51431, -38.22008281], [49.290821, 81.99277149], vector.behavior)
    b = _lorvecfromnumpy(
        [99.415420, 82.62186155], [-78.89404, 65.20115408],
        [-161.6102, 21.07726162], [205.54469, 108.13159743], vector.behavior)
    antib = _lorvecfromnumpy(
        [-49.87086, -24.91287065], [91.930526, -1.88888067],
        [-347.3868, -8.14035951], [362.82086, 26.73045868], vector.behavior)
    nu = _lorvecfromnumpy(
        [34.521587], [-51.23474], [-6.555319], [70.848953], vector.behavior)
    antinu = _lorvecfromnumpy(
        [11.179965], [-3.844941], [7.0419898], [13.760989], vector.behavior)
    MET = nu + antinu
    MET = _lorvecfromnumpy(
        [MET[0].x, -40.12400817871094], [MET[0].y, -106.56241607666016],
        [0., 0.], [0., 0.], vector.behavior)
    t, t_ = betchart(lep, antilep, b, antib, MET)
    print(t, t_)

    import time
    n = 10000
    lep = ak.Array({
        "t": np.full(n, 165.33320), "x": np.full(n, 26.923591),
        "y": np.full(n, 16.170616), "z": np.full(n, -162.3227)},
        with_name="LorentzVector", behavior=vector.behavior)
    antilep = ak.Array({
        "t": np.full(n, 49.290821), "x": np.full(n, -34.58441),
        "y": np.full(n, -13.27824), "z": np.full(n, -32.51431)},
        with_name="LorentzVector", behavior=vector.behavior)
    b = ak.Array({
        "t": np.full(n, 205.54469), "x": np.full(n, 99.415420),
        "y": np.full(n, -78.89404), "z": np.full(n, -161.6102)},
        with_name="LorentzVector", behavior=vector.behavior)
    antib = ak.Array({
        "t": np.full(n, 362.82086), "x": np.full(n, -49.87086),
        "y": np.full(n, 91.930526), "z": np.full(n, -347.3868)},
        with_name="LorentzVector", behavior=vector.behavior)
    nu = ak.Array({
        "t": np.full(n, 70.848953), "x": np.full(n, 34.521587),
        "y": np.full(n, -51.23474), "z": np.full(n, -6.555319)},
        with_name="LorentzVector", behavior=vector.behavior)
    antinu = ak.Array({
        "t": np.full(n, 13.760989), "x": np.full(n, 11.179965),
        "y": np.full(n, -3.844941), "z": np.full(n, 7.0419898)},
        with_name="LorentzVector", behavior=vector.behavior)
    sump4 = nu + antinu
    met = ak.Array({
        "pt": sump4.pt, "eta": np.zeros(n), "phi": sump4.phi,
        "mass": np.zeros(n)}, with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior)

    t = time.time()
    betchart(lep, antilep, b, antib, met)
    print("Took {} s for {} events".format(time.time() - t, n))
