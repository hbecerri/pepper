import numpy as np
import uproot
import uproot_methods
import coffea
from coffea.analysis_objects import JaggedCandidateArray
import awkward

from pepper.misc import jaggeddepth, jaggedfromnumpy


def _maybe_sample(s, size):
    if isinstance(s, coffea.hist.Hist):
        if s.dim() != 1 or s.dense_dim() != 1:
            raise ValueError("mass histogram has invalid dimensions")
        values = s.values()[()]
        centers = s.axes()[0].centers()
        p = values / values.sum()
        s = np.random.choice(centers, size, p=p)
    elif isinstance(s, uproot_methods.classes.TH1.Methods):
        values, edges = s.numpy()
        centers = (edges[1:] + edges[:-1]) / 2
        p = values / values.sum()
        s = np.random.choice(centers, size, p=p)
    elif isinstance(s, (int, float)):
        s = np.full(size, s)
    return s


def _random_orthogonal(vec):
    random = np.random.rand(*vec.shape)
    rnorm = random / np.linalg.norm(random, axis=-1, keepdims=True)
    vnorm = vec / np.linalg.norm(vec, keepdims=True)
    u = rnorm - (rnorm * vnorm).sum(axis=-1, keepdims=True) * vnorm
    unorm = u / np.linalg.norm(u, keepdims=True)
    return unorm


def _rotate_axis(vec, axis, angle):
    # Taken from uproot_methods.TVector3
    # Using TVector3 directly led to many kinds of complications
    vx, vy, vz = np.rollaxis(vec, -1)
    ux, uy, uz = np.rollaxis(vec, -1)
    c = np.cos(angle)
    s = np.sin(angle)
    c1 = 1 - c

    x = ((c + ux**2 * c1) * vx
         + (ux * uy * c1 - uz * s) * vy
         + (ux * uz * c1 + uy * s) * vz)
    y = ((ux * uy * c1 + uz * s) * vx
         + (c + uy**2 * c1) * vy
         + (uy * uz * c1 - ux * s) * vz)
    z = ((ux * uz * c1 - uy * s) * vx
         + (uy * uz * c1 + ux * s) * vy
         + (c + uz**2 * c1) * vz)

    return np.stack([x, y, z], axis=-1)


def _smear(fourvec, energyf, alpha, num):
    num_events = fourvec.size
    e = fourvec.E[:, None]
    p3 = np.stack([fourvec.x, fourvec.y, fourvec.z], axis=-1)[:, None, :]
    m = fourvec.mass[:, None]
    if num is None:
        return e, p3[..., 0], p3[..., 1], p3[..., 2]

    e = np.broadcast_to(e, (num_events, num))
    p3 = np.broadcast_to(p3, (num_events, num, 3))
    m = np.broadcast_to(m, (num_events, num))
    if energyf is not None and num is not None:
        e = e * _maybe_sample(energyf, (num_events, num))
        # Cap energy to something bit above the mass
        e[e < m] = 1.01 * m[e < m]
    if alpha is not None and num is not None:
        # Rotate around a random orthogonal axis by alpha
        r = _random_orthogonal(p3)
        p3 = _rotate_axis(p3, r, _maybe_sample(alpha, (num_events, num)))
    # Keep mass constant
    p3 = p3 * np.sqrt((e**2 - m**2) / (p3**2).sum(axis=-1))[..., None]

    return e, p3[..., 0], p3[..., 1], p3[..., 2]


def _roots_vectorized(poly, axis=-1):
    """Like numpy.roots just that it can take any number of axes, allowing to
    compute the roots of any number of polynomials at once"""
    if poly.ndim == 1:
        return np.roots(poly)
    # Bring input to two-dim shape
    poly = poly.swapaxes(axis, -1)
    shape = poly.shape
    poly = poly.reshape(-1, shape[-1])
    # Build companion matrix
    ones = np.ones(poly.shape[1] - 2, poly.dtype)
    companion = np.tile(np.diag(ones, -1), (poly.shape[0], 1, 1))
    companion[:, 0] = -poly[:, 1:] / poly[:, 0, None]
    # Find eigenvalues of companion matrix <=> Find roots of poly
    roots = np.linalg.eigvals(companion)
    num_roots = roots.shape[-1]
    # Bring roots to the original input shape
    roots = roots.reshape(shape[:-1] + (num_roots,)).swapaxes(-1, axis)

    return roots


def sonnenschein(lep, antilep, b, antib, met, mw=80.3, mt=172.5,
                 num_smear=None, energyfl=None, energyfj=None, alphal=None,
                 alphaj=None, hist_mlb=None):
    """Full kinematic reconstruction for dileptonic ttbar using Sonnenschein's
    method https://arxiv.org/pdf/hep-ph/0603011.pdf
    Arguments:
    lep -- TLorentzVectorArray holding one negativly charged lepton per event
    antilep -- TLorentzVectorArray holding one positively charged lepton per
               event
    b -- TLorentzVectorArray holding one negativly charged bottom quark per
         event
    antib -- TLorentzVectorArray holding one positively changed bottom quark
             per event
    met -- TLorentzVectorArray holding with one entry per event, yielding
           the MET pt and phi
    mw -- Mass of the W bosons. Either a number or a histogram, to sample from
    mt  -- Same as mwp for the top quarks
    num_smear -- Number of times an event is smeared. If None, smearing is off
    energyfl -- Histogram giving Ereco/Egen for the leptons. If None, lepton
                energy won't be smeared
    energyfj -- Same as energyfl for bottom quarks
    alphal -- Histogram giving the angle between reco and gen leptons. If None,
              lepton angles won't be smeared
    alphaj -- Same as alphal for bottom quarks
    hist_mlb -- Histogram of the lepton-bottom-quark-mass distribution. Is
                needed, if num_smear is not None
    Inputs should be Lorentz vectors rather than candidate arrays"""

    if jaggeddepth(lep) > 1:
        # Get rid of jagged dimension, as we have one particle per row and
        # variable
        lep = lep.flatten()
        antilep = antilep.flatten()
        b = b.flatten()
        antib = antib.flatten()
        met = met.flatten()

    # Allow num_smear to be 0
    if num_smear == 0:
        num_smear = None

    # Use 2d numpy arrays. Use first axis for events, second for smearing
    num_events = lep.size
    lE, lx, ly, lz = _smear(lep, energyfl, alphal, num_smear)
    alE, alx, aly, alz = _smear(antilep, energyfl, alphal, num_smear)
    bE, bx, by, bz = _smear(b, energyfj, alphaj, num_smear)
    abE, abx, aby, abz = _smear(antib, energyfj, alphaj, num_smear)
    # Even if num_smear is None, we have a smear axis. Update num_smear
    num_smear = max(lE.shape[1], bE.shape[1])
    # Unpack MET compontents and also propagate smearing to it
    METx = (met.x[:, None] + lx - lep.x[:, None] + alx - antilep.x[:, None]
                           + bx - b.x[:, None] + abx - antib.x[:, None])
    METy = (met.y[:, None] + ly - lep.y[:, None] + aly - antilep.y[:, None]
                           + by - b.y[:, None] + aby - antib.y[:, None])

    mwp = _maybe_sample(mw, (num_events, 1))
    mwm = _maybe_sample(mw, (num_events, 1))
    mat = _maybe_sample(mt, (num_events, 1))
    mt = _maybe_sample(mt, (num_events, 1))
    ml = np.sqrt(lE**2 - lx**2 - ly**2 - lz**2)
    mal = np.sqrt(alE**2 - alx**2 - aly**2 - alz**2)
    mb = np.sqrt(bE**2 - bx**2 - by**2 - bz**2)
    mab = np.sqrt(abE**2 - abx**2 - aby**2 - abz**2)

    if hist_mlb is not None:
        mlab = np.sqrt((lE + abE)**2 - (lx + abx)**2
                       - (ly + aby)**2 - (lz + abz)**2)
        malb = np.sqrt((alE + bE)**2 - (alx + bx)**2
                       - (aly + by)**2 - (alz + bz)**2)
        # Set nan, over and underflow to 0
        allvalues = np.r_[0, hist_mlb.values, 0, 0]
        plab = allvalues[np.searchsorted(hist_mlb.alledges, mlab) - 1]
        palb = allvalues[np.searchsorted(hist_mlb.alledges, malb) - 1]
        weights = plab * palb
    elif num_smear is not None and num_smear > 1:
        raise ValueError("Smearing is enabled but got None for hist_mlb")
    else:
        weights = np.ones_like(lE)

    a1 = ((bE + alE) * (mwp ** 2 - mal ** 2)
          - alE * (mt ** 2 - mb ** 2 - mal ** 2) + 2 * bE * alE ** 2
          - 2 * alE * (bx * alx + by * aly + bz * alz))
    a2 = 2 * (bE * alx - alE * bx)
    a3 = 2 * (bE * aly - alE * by)
    a4 = 2 * (bE * alz - alE * bz)

    b1 = ((abE + lE) * (mwm ** 2 - ml ** 2)
          - lE * (mat ** 2 - mab ** 2 - ml ** 2) + 2 * abE * lE ** 2
          - 2 * lE * (abx * lx + aby * ly + abz * lz))
    b2 = 2 * (abE * lx - lE * abx)
    b3 = 2 * (abE * ly - lE * aby)
    b4 = 2 * (abE * lz - lE * abz)

    c00 = (- 4 * (alE ** 2 - aly ** 2) - 4 * (alE ** 2 - alz ** 2)
           * (a3 / a4) ** 2 - 8 * aly * alz * a3 / a4)
    c10 = (- 8 * (alE ** 2 - alz ** 2) * a2 * a3 / (a4 ** 2) + 8 * alx * aly
           - 8 * alx * alz * a3 / a4 - 8 * aly * alz * a2 / a4)
    c11 = (4 * (mwp ** 2 - mal ** 2) * (aly - alz * a3 / a4)
           - 8 * (alE ** 2 - alz ** 2) * a1 * a3 / (a4 ** 2)
           - 8 * aly * alz * a1 / a4)
    c20 = (- 4 * (alE ** 2 - alx ** 2) - 4 * (alE ** 2 - alz ** 2)
           * (a2 / a4) ** 2 - 8 * alx * alz * a2 / a4)
    c21 = (4 * (mwp ** 2 - mal ** 2) * (alx - alz * a2 / a4)
           - 8 * (alE ** 2 - alz ** 2) * a1 * a2 / (a4 ** 2)
           - 8 * alx * alz * a1 / a4)
    c22 = ((mwp ** 2 - mal ** 2) ** 2 - 4 * (alE ** 2 - alz ** 2)
           * (a1 / a4) ** 2 - 4 * mwp ** 2 * alz * a1 / a4)

    d00 = (- 4 * (lE ** 2 - ly ** 2) - 4 * (lE ** 2 - lz ** 2)
           * (b3 / b4) ** 2 - 8 * ly * lz * b3 / b4)
    d10 = (- 8 * (lE ** 2 - lz ** 2) * b2 * b3 / (b4 ** 2)
           + 8 * lx * ly - 8 * lx * lz * b3 / b4 - 8 * ly * lz * b2 / b4)
    d11p = (4 * (mwm ** 2 - ml ** 2) * (ly - lz * b3 / b4)
            - 8 * (lE ** 2 - lz ** 2) * b1 * b3 / (b4 ** 2)
            - 8 * ly * lz * b1 / b4)
    d20 = (- 4 * (lE ** 2 - lx ** 2) - 4 * (lE ** 2 - lz ** 2)
           * (b2 / b4) ** 2 - 8 * lx * lz * b2 / b4)
    d21p = (4 * (mwm ** 2 - ml ** 2) * (lx - lz * b2 / b4)
            - 8 * (lE ** 2 - lz ** 2) * b1 * b2 / (b4 ** 2)
            - 8 * lx * lz * b1 / b4)
    d22p = ((mwm ** 2 - ml ** 2) ** 2 - 4 * (lE ** 2 - lz ** 2)
            * (b1 / b4) ** 2 - 4 * mwm ** 2 * lz * b1 / b4)

    d11 = - d11p - 2 * METy * d00 - METx * d10
    d21 = - d21p - 2 * METx * d20 - METy * d10
    d22 = (d22p + METx ** 2 * d20 + METy ** 2 * d00
           + METx * METy * d10 + METx * d21p + METy * d11p)

    h0 = (c00 ** 2 * d20 ** 2 + c10 * d20 * (c10 * d00 - c00 * d10)
          + c20 * d10 * (c00 * d10 - c10 * d00)
          + c20 * d00 * (c20 * d00 - 2 * c00 * d20))
    h1 = (c00 * d21 * (2 * c00 * d20 - c10 * d10)
          - c00 * d20 * (c11 * d10 + c10 * d11)
          + c00 * d10 * (2 * c20 * d11 + c21 * d10)
          - 2 * c00 * d00 * (c21 * d20 + c20 * d21)
          + c10 * d00 * (2 * c11 * d20 + c10 * d21)
          + c20 * d00 * (2 * c21 * d00 - c10 * d11)
          - d00 * d10 * (c11 * c20 + c10 * c21))
    # (note the sign of c20*d00*(...) is different to the appendix of
    # Sonnenschein's paper, and instead follows the implementation on github:
    # https://github.com/gerbaudo/ttbar-kinsol-comp,
    # which gives the right solution)

    h2 = (c00 ** 2 * (2 * d22 * d20 + d21 ** 2)
          - c00 * d21 * (c11 * d10 + c10 * d11)
          + c11 * d20 * (c11 * d00 - c00 * d11)
          + c00 * d10 * (c22 * d10 - c10 * d22)
          + c00 * d11 * (2 * c21 * d10 + c20 * d11)
          + (2 * c22 * c20 + c21 ** 2) * d00 ** 2
          - 2 * c00 * d00 * (c22 * d20 + c21 * d21 + c20 * d22)
          + c10 * d00 * (2 * c11 * d21 + c10 * d22)
          - d00 * d10 * (c11 * c21 + c10 * c22)
          - d00 * d11 * (c11 * c20 + c10 * c21))
    h3 = (c00 * d21 * (2 * c00 * d22 - c11 * d11)
          + c00 * d11 * (2 * c22 * d10 + c21 * d11)
          + c22 * d00 * (2 * c21 * d00 - c11 * d10)
          - c00 * d22 * (c11 * d10 + c10 * d11)
          - 2 * c00 * d00 * (c22 * d21 + c21 * d22)
          - d00 * d11 * (c11 * c21 + c10 * c22)
          + c11 * d00 * (c11 * d21 + 2 * c10 * d22))
    h4 = (c00 ** 2 * d22 ** 2 + c11 * d22 * (c11 * d00 - c00 * d11)
          + c00 * c22 * (d11 ** 2 - 2 * d00 * d22)
          + c22 * d00 * (c22 * d00 - c11 * d11))
    h = np.stack([h0, h1, h2, h3, h4], axis=-1)

    roots = _roots_vectorized(h)
    vpx = roots.real
    is_real = abs(roots.imag) < 10 ** -6

    c0 = c00[..., None]
    c1 = c10[..., None] * vpx + c11[..., None]
    c2 = c20[..., None] * vpx ** 2 + c21[..., None] * vpx + c22[..., None]
    d0 = d00[..., None]
    d1 = d10[..., None] * vpx + d11[..., None]
    d2 = d20[..., None] * vpx ** 2 + d21[..., None] * vpx + d22[..., None]

    vpy = (c0 * d2 - c2 * d0)/(c1 * d0 - c0 * d1)
    vpz = ((-a1[..., None] - a2[..., None] * vpx - a3[..., None] * vpy)
           / a4[..., None])
    vbarpx = METx[..., None] - vpx
    vbarpy = METy[..., None] - vpy
    vbarpz = ((-b1[..., None] - b2[..., None] * vbarpx - b3[..., None]
               * vbarpy) / b4[..., None])

    is_real = jaggedfromnumpy(is_real)
    vpx = jaggedfromnumpy(vpx)[is_real]
    vpy = jaggedfromnumpy(vpy)[is_real]
    vpz = jaggedfromnumpy(vpz)[is_real]
    vbarpx = jaggedfromnumpy(vbarpx)[is_real]
    vbarpy = jaggedfromnumpy(vbarpy)[is_real]
    vbarpz = jaggedfromnumpy(vbarpz)[is_real]
    weights = jaggedfromnumpy(weights)

    v = uproot_methods.TLorentzVectorArray.from_xyzm(
        vpx, vpy, vpz, vpz.zeros_like())
    av = uproot_methods.TLorentzVectorArray.from_xyzm(
        vbarpx, vbarpy, vbarpz, vbarpz.zeros_like())
    b = uproot_methods.TLorentzVectorArray.from_cartesian(
        jaggedfromnumpy(bx), jaggedfromnumpy(by), jaggedfromnumpy(bz),
        jaggedfromnumpy(bE))
    ab = uproot_methods.TLorentzVectorArray.from_cartesian(
        jaggedfromnumpy(abx), jaggedfromnumpy(aby), jaggedfromnumpy(abz),
        jaggedfromnumpy(abE))
    lep = uproot_methods.TLorentzVectorArray.from_cartesian(
        jaggedfromnumpy(lx), jaggedfromnumpy(ly), jaggedfromnumpy(lz),
        jaggedfromnumpy(lE))
    alep = uproot_methods.TLorentzVectorArray.from_cartesian(
        jaggedfromnumpy(alx), jaggedfromnumpy(aly), jaggedfromnumpy(alz),
        jaggedfromnumpy(alE))
    wp = v + alep
    # Doing alep + v (terms switched) causes bugs in uproot/awkward
    wm = av + lep
    t = wp + b
    at = wm + ab

    # Reduce solution axis and pick the solution with the smallest mtt
    has_solution = v.flatten().counts >= 1
    has_solution = awkward.JaggedArray.fromcounts(
        np.full(num_events, num_smear), has_solution)
    min_mtt = (t + at).mass.argmin()
    v = v[min_mtt][has_solution][:, :, 0]
    av = av[min_mtt][has_solution][:, :, 0]
    wp = wp[min_mtt][has_solution][:, :, 0]
    wm = wm[min_mtt][has_solution][:, :, 0]
    t = t[min_mtt][has_solution][:, :, 0]
    at = at[min_mtt][has_solution][:, :, 0]
    weights = weights[has_solution]

    # Undo smearing by averaging
    sum_weights = weights.sum()
    t = (t * weights / sum_weights).sum()
    at = (at * weights / sum_weights).sum()

    # Put it into a JaggedArrays
    has_solution = has_solution.any()
    cls = uproot_methods.classes.TLorentzVector.JaggedArrayMethods
    counts = has_solution.astype(int)
    t = cls.fromcounts(counts, t[has_solution])
    at = cls.fromcounts(counts, at[has_solution])

    return t, at


if __name__ == '__main__':
    # test case:
    lep = JaggedCandidateArray.candidatesfromcounts(
        [1], energy=np.array([165.33320]), px=np.array([26.923591]),
        py=np.array([16.170616]), pz=np.array([-162.3227]))
    antilep = JaggedCandidateArray.candidatesfromcounts(
        [1], energy=np.array([49.290821]), px=np.array([-34.58441]),
        py=np.array([-13.27824]), pz=np.array([-32.51431]))
    b = JaggedCandidateArray.candidatesfromcounts(
        [1], energy=np.array([205.54469]), px=np.array([99.415420]),
        py=np.array([-78.89404]), pz=np.array([-161.6102]))
    antib = JaggedCandidateArray.candidatesfromcounts(
        [1], energy=np.array([362.82086]), px=np.array([-49.87086]),
        py=np.array([91.930526]), pz=np.array([-347.3868]))
    nu = JaggedCandidateArray.candidatesfromcounts(
        [1], energy=np.array([70.848953]), px=np.array([34.521587]),
        py=np.array([-51.23474]), pz=np.array([-6.555319]))
    antinu = JaggedCandidateArray.candidatesfromcounts(
        [1], energy=np.array([13.760989]), px=np.array([11.179965]),
        py=np.array([-3.844941]), pz=np.array([7.0419898]))
    sump4 = nu.p4 + antinu.p4
    met = JaggedCandidateArray.candidatesfromcounts(
        [1], pt=sump4.pt[0], eta=np.zeros(1), phi=sump4.phi[0],
        mass=np.zeros(1))
    top, antitop = sonnenschein(
        lep["p4"], antilep["p4"], b["p4"], antib["p4"], met["p4"])
    print("MC Truth:", (antilep.p4 + nu.p4 + b.p4)[0],
                       (lep.p4 + antinu.p4 + antib.p4)[0])
    print("Reconstructed:", top, antitop)
    # top: [TLorentzVector(x=93.465, y=-160.14, z=-213.12, t=331)]
    # antitop: [TLorentzVector(x=-5.8792, y=120.99, z=-507.62, t=549.64)]

    import time
    n = 10000
    lep = uproot_methods.TLorentzVectorArray.from_cartesian(
        np.full(n, 26.923591), np.full(n, 16.170616), np.full(n, -162.3227),
        np.full(n, 165.33320))
    alep = uproot_methods.TLorentzVectorArray.from_cartesian(
        np.full(n, -34.58441), np.full(n, -13.27824), np.full(n, -32.51431),
        np.full(n, 49.290821))
    b = uproot_methods.TLorentzVectorArray.from_cartesian(
        np.full(n, 99.415420), np.full(n, -78.89404), np.full(n, -161.6102),
        np.full(n, 205.54469))
    ab = uproot_methods.TLorentzVectorArray.from_cartesian(
        np.full(n, -49.87086), np.full(n, 91.930526), np.full(n, -347.3868),
        np.full(n, 362.82086))
    met = uproot_methods.TLorentzVectorArray.from_cartesian(
        np.full(n, 45.701552), np.full(n, -55.079681), np.full(n, 0.),
        np.full(n, 71.5709656))

    t = time.time()
    sonnenschein(lep, alep, b, ab, met)
    print("Took {} s for {} events".format(time.time() - t, n))
