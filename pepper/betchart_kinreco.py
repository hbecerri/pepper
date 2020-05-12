import numpy as np
from uproot_methods.classes.TLorentzVector import TLorentzVectorArray as vlv
from coffea.analysis_objects import JaggedCandidateArray as Jca
import uproot_methods
import math
from awkward import JaggedArray
import awkward
from numba import jit, njit
from numba.experimental import jitclass
from numba import float32, float64, int64
from functools import partial
from scipy.optimize import leastsq


@njit(cache=True)
def UnitCircle():
    '''Unit circle in extended representation.'''
    diag = np.array([1., 1., -1.])
    return np.diag(diag)


@njit(cache=True)
def cofactor(A, ij):
    '''Cofactor[i,j] of 3x3 matrix A.'''
    i, j = ij
    iidx, jidx = [0, 1, 2], [0, 1, 2]
    iidx.pop(i)
    jidx.pop(j)
    B = A[np.array(iidx), :]
    a = B[:, np.array(jidx)]
    return (-1)**(i+j) * (a[0, 0]*a[1, 1] - a[1, 0]*a[0, 1])


@njit(cache=True)
def R(axis, angle):
    '''Rotation matrix about x,y, or z axis (0,1, or 2, respectively.)'''
    c, s = math.cos(angle), math.sin(angle)
    R = c * np.eye(3)
    for i in [-1, 0, 1]:
        R[(axis - i) % 3, (axis + i) % 3] = i * s + (1 - i * i)
    return R


@njit(cache=True)
def Derivative():
    '''Matrix to differentiate [cos(t),sin(t),1] and R(2,_).'''
    return R(2, math.pi / 2).dot(np.diag([1, 1, 0]))


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
def _maybe_swap(G, swapXY):
    if swapXY:
        Q = G[np.array([1, 0, 2]), :][:, np.array([1, 0, 2])]
        return Q
    else:
        return G


@njit(cache=True)
def factor_degenerate(G, zero=0):
    '''Linear factors of a degenerate quadratic polynomial.'''
    if G[0, 0] == 0 == G[1, 1]:
        return [[G[0, 1], 0, G[1, 2]],
                [0, G[0, 1], G[0, 2] - G[1, 2]]]

    swapXY = abs(G[0, 0]) > abs(G[1, 1])
    Q = _maybe_swap(G, swapXY)
    Q /= Q[1, 1]
    q22 = cofactor(Q.real, (2, 2))

    if -q22 <= zero:
        lines = [[Q[0, 1], Q[1, 1], Q[1, 2] + s]
                 for s in multisqrt(-cofactor(Q.real, (0, 0)))]
    else:
        x0, y0 = [cofactor(Q.real, (i, 2)) / q22 for i in [0, 1]]
        lines = [[m, Q[1, 1], -Q[1, 1]*y0 - m*x0]
                 for m in [Q[0, 1] + s for s in multisqrt(-q22)]]

    if swapXY:
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
def intersections_ellipses(A, B, zero=1e-6):
    '''Points of intersection between two ellipses.'''
    A = A.astype(np.complex128)
    B = B.astype(np.complex128)
    if abs(np.linalg.det(B)) > abs(np.linalg.det(A)):
        A, B = B, A
    eigval = [e.real for e in np.linalg.eigvals(np.linalg.inv(A).dot(B))
              if np.abs(e.imag) <= zero][0]
    lines = factor_degenerate(B - eigval * A)
    points = [point for L in lines
              for point in intersections_ellipse_line(A, L)]
    return points


nSS_spec = {"D2": float32,
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
            "Wm2": float64}


@jitclass(nSS_spec)
class nuSolutionSet(object):
    '''Collection of definitions for neutrino analytic solution
       in t->b, mu, nu.'''

    def __init__(self, b, mu,  # arrays of pt, eta, phi, E components
                 Wm2=80.385**2, Tm2=172.5**2  # GeV**2
                 ):
        self.mu = mu
        self.b = b
        pb = np.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)
        pmu = np.sqrt(mu[0] ** 2 + mu[1] ** 2 + mu[2] ** 2)
        self.mb2 = b[3] ** 2 - pb ** 2
        self.D2 = 0.5 * (Tm2 - Wm2 - self.mb2)
        self.pb = pb
        self.pmu = pmu
        self.c = (b[0] * mu[0] + b[1] * mu[1] + b[2] * mu[2]) / (pb * pmu)
        if self.c > 1:
            print("cos exceeds allowed values, set to 1 - calculated val:",
                  self.c)
            self.c = 1
        self.s = math.sqrt(1 - self.c ** 2)
        self.x0 = -0.5 * Wm2 / pmu
        self.y0 = - (self.x0 * self.c + self.D2 / pb) / self.s

        self.m = (np.abs(self.b[3]) / pb - self.c) / self.s
        self.m_alt = (- np.abs(self.b[3]) / pb - self.c) / self.s
        self.y1 = - self.x0 / self.m
        self.x1 = self.x0 + (self.y1 - self.y0) / self.m

        self.Z = math.sqrt(max(0, (self.y1 * (self.y1 - 2 * self.y0)
                                   - self.x0 ** 2 - Wm2)))
        self.Wm2 = Wm2

    @property
    def K(self):
        '''Extended rotation from F' to F coordinates.'''
        return np.array([[self.c, -self.s, 0., 0.],
                         [self.s,  self.c, 0., 0.],
                         [0.,      0.,     1., 0.],
                         [0.,      0.,     0., 1.]])

    @property
    def A_mu(self):
        '''F coordinates constraint on W momentum: paraboloid.'''
        return np.array([[0.,      0., 0.,                 self.x0],
                         [0.,      1., 0.,                      0.],
                         [0.,      0., 1.,                      0.],
                         [self.x0, 0., 0., self.Wm2 - self.x0 ** 2]])

    @property
    def A_b(self):
        '''F coordinates constraint on W momentum: ellipsoid.'''
        K, D2, Wm2, e2 = self.K, self.D2, self.Wm2, self.b[3]**2
        return K.dot(np.array(
            [[self.mb2,      0., 0.,     - D2 * self.pb],
             [0.,            e2, 0.,                 0.],
             [0.,            0., e2,                 0.],
             [-D2 * self.pb, 0., 0., e2 * Wm2 - D2 ** 2]])).dot(K.T)

    @property
    def R_T(self):
        '''Rotation from F coordinates to laboratory coordinates.'''
        mu = self.mu
        mutheta = np.arctan2(np.sqrt(mu[0] ** 2 + mu[1] ** 2), mu[2])
        muphi = np.arctan2(mu[1], mu[0])
        R_z = R(2, - muphi)
        R_y = R(1,  0.5 * math.pi - mutheta)
        R_x = [R(0, - math.atan2(z, y))
               for x, y, z in (R_y.dot(R_z.dot(self.b[0:3])), )][0]
        return R_z.T.dot(R_y.T.dot(R_x.T))

    @property
    def E_tilde(self):
        '''Transformation of t=[c,s,1] to nu momentum: F coordinates.'''
        Z, m, x1, y1, p = self.Z, self.m, self.x1, self.y1, self.pmu
        return np.array([[Z/m, 0.,   x1 - p],
                         [Z,   0.,       y1],
                         [0.,   Z,       0.]])

    @property
    def E(self):
        '''Transformation of t=[c,s,1] to nu momentum: lab coordinates.'''
        return self.R_T.dot(self.E_tilde)

    @property
    def E_perp(self):
        '''Transformation of t=[c,s,1] to nu transverse momentum:
           lab coordinates.'''
        return np.vstack((self.E[:2], np.array([[0., 0., 1.]])))

    @property
    def N(self):
        '''Solution ellipse of nu transverse momentum: lab coordinates.'''
        E = self.E_perp
        return np.linalg.inv(E.T).dot(UnitCircle()).dot(np.linalg.inv(E))


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


def _run_kinreco(lep, alep, b, bbar, METx, METy, Wm2=80.385**2, Tm2=172.5**2):
    nevents = len(lep)
    nu = np.empty((3, 4 * nevents), dtype=np.float64)
    nubar = np.empty((3, 4 * nevents), dtype=np.float64)
    fails = 0
    nsols = np.zeros(nevents, dtype=np.int64)
    for eventi in range(nevents):
        bs = (b[eventi], bbar[eventi])
        mus = (alep[eventi], lep[eventi])
        metX, metY = METx[eventi], METy[eventi]
        solutionSets = [nuSolutionSet(B, M, Wm2, Tm2) for B, M in zip(bs, mus)]

        V0 = np.outer([metX, metY, 0.], [0., 0., 1.])
        S = V0 - UnitCircle()

        hasZs = solutionSets[0].Z != 0 and solutionSets[1].Z != 0
        if hasZs:
            N, N_ = [ss.N for ss in solutionSets]
            n_ = S.T.dot(N_).dot(S)

            v = intersections_ellipses(N, n_)
            v_ = [S.dot(sol) for sol in v]

            if not v and leastsq:
                es = [ss.E_perp for ss in solutionSets]
                met = np.array([metX, metY, 1])
                ts, _ = leastsq(partial(test_func, es, met), [0., 0.],
                                ftol=5e-5, epsfcn=0.01)
                v, v_ = [[i] for i in nus(ts, es)]

            K, K_ = [ss.E.dot(np.linalg.inv(ss.E_perp)) for ss in solutionSets]
            for sol_i, s in enumerate(v):
                for i in [0, 1, 2]:
                    nu[i, 4 * eventi + sol_i] = K.dot(s)[i]
            for i, s_ in enumerate(v_):
                for i in [0, 1, 2]:
                    nubar[i, 4 * eventi + sol_i] = K_.dot(s_)[i]
            nsols[eventi] = len(v)
        else:
            fails += 1
    print(fails, " / ", len(nsols))
    return nu, nubar, nsols


def _makeJCA(counts, lv):
    return Jca.candidatesfromcounts(counts, px=lv.x, py=lv.y,
                                    pz=lv.z, mass=lv.mass)


def _make_2D_array(lv):
    return np.stack((lv.x, lv.y, lv.z, lv.E), axis=-1)


def betchart(lep, antilep, b, antib, MET, MW=80.385, Mt=172.5):
    METx = MET.pt * np.cos(MET.phi)
    METy = MET.pt * np.sin(MET.phi)
    # Currently, awkward doesn't seem to be surported in numba,
    # so create a helper function with no awkward components
    # which can be compiled. Awkward 1 should be fully numba
    # compatible, so it may be possible to recombine these
    # functions in future
    v, vb, nsols = _run_kinreco(_make_2D_array(lep),
                                _make_2D_array(antilep),
                                _make_2D_array(b),
                                _make_2D_array(antib),
                                METx,
                                METy)
    starts = np.arange(0, 4*len(lep), 4)
    stops = starts + nsols
    vpx = JaggedArray(starts, stops, v[0]).flatten()
    vpy = JaggedArray(starts, stops, v[1]).flatten()
    vpz = JaggedArray(starts, stops, v[2]).flatten()
    v = Jca.candidatesfromcounts(nsols, px=vpx, py=vpy, pz=vpz, mass=0)
    print(v.p4)
    vbpx = JaggedArray(starts, stops, vb[0]).flatten()
    vbpy = JaggedArray(starts, stops, vb[1]).flatten()
    vbpz = JaggedArray(starts, stops, vb[2]).flatten()
    av = Jca.candidatesfromcounts(nsols, px=vbpx, py=vbpy, pz=vbpz, mass=0)
    has_sol = nsols > 0
    lep = _makeJCA(np.ones(len(nsols)), lep)
    alep = _makeJCA(np.ones(len(nsols)), antilep)
    b = _makeJCA(np.ones(len(nsols)), b)
    bbar = _makeJCA(np.ones(len(nsols)), antib)
    wp = v.cross(alep)
    wm = av.cross(lep)
    t = wp.cross(b)
    at = wm.cross(bbar)
    min_mtt = (t + at).p4.mass.argmin()
    t = t[min_mtt]
    at = at[min_mtt]
    return t, at


if __name__ == '__main__':
    lep = vlv([26.923591, 86.3662048], [16.170616, -4.14978429],
              [-162.3227, -125.30777765], [165.33320, 152.24451503])
    antilep = vlv([-34.58441, -44.03565424], [-13.27824, 57.64460697],
                  [-32.51431, -38.22008281], [49.290821, 81.99277149])
    b = vlv([99.415420, 82.62186155], [-78.89404, 65.20115408],
            [-161.6102, 21.07726162], [205.54469, 108.13159743])
    antib = vlv([-49.87086, -24.91287065], [91.930526, -1.88888067],
                [-347.3868, -8.14035951], [362.82086, 26.73045868])
    nu = vlv([34.521587], [-51.23474], [-6.555319], [70.848953])
    antinu = vlv([11.179965], [-3.844941], [7.0419898], [13.760989])
    MET = nu + antinu
    MET = vlv([MET[0].x, -40.12400817871094], [MET[0].y, -106.56241607666016],
              [0., 0.], [0., 0.])
    t, t_ = betchart(lep, antilep, b, antib, MET)
    print(t.p4, t_.p4)

    import time
    n = 10000
    lep = vlv(**{q: np.repeat(getattr(lep, q)[1], n)
                 for q in ["x", "y", "z", "t"]})
    antilep = vlv(**{q: np.repeat(getattr(antilep, q)[1], n)
                     for q in ["x", "y", "z", "t"]})
    b = vlv(**{q: np.repeat(getattr(b, q)[1], n)
               for q in ["x", "y", "z", "t"]})
    antib = vlv(**{q: np.repeat(getattr(antib, q)[1], n)
                   for q in ["x", "y", "z", "t"]})
    MET = vlv(**{q: np.repeat(getattr(MET, q)[1], n)
                 for q in ["x", "y", "z", "t"]})
    print(lep)
    t = time.time()
    betchart(lep, antilep, b, antib, MET)
    print("Took {} s for {} events".format(time.time() - t, n))
