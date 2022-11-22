import numpy as np
import uproot
import awkward as ak

from pepper.misc import chunked_calls


def _maybe_sample(s, size, rng):
    if isinstance(s, uproot.model.Model) and s.classname.startswith("TH1"):
        values, edges = s.to_numpy()
        centers = (edges[1:] + edges[:-1]) / 2
        p = values / values.sum()
        s = rng.choice(centers, size, p=p)
    elif isinstance(s, (int, float)):
        s = np.full(size, s)
    elif isinstance(s, np.ndarray):
        s = s.reshape(size)
    return s


def _random_orthogonal(vec, rng):
    random = rng.random(vec.shape)
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


def _smear(fourvec, energyf, alpha, num, rng):
    num_events = len(fourvec)
    # Axes are [events, smearing, lorentz]
    e = np.asarray(fourvec.energy)[:, None]
    p3 = np.stack([np.asarray(fourvec.x), np.asarray(fourvec.y),
                   np.asarray(fourvec.z)], axis=-1)[:, None, :]
    m = np.asarray(fourvec.mass)[:, None]
    if num is None:
        return e, p3[..., 0], p3[..., 1], p3[..., 2]

    e = np.broadcast_to(e, (num_events, num))
    p3 = np.broadcast_to(p3, (num_events, num, 3))
    m = np.broadcast_to(m, (num_events, num))
    if energyf is not None and num is not None:
        e = e * _maybe_sample(energyf, (num_events, num), rng)
        # Make sure e and m still have same dtype, otherwise numerical problems
        m = m.astype(e.dtype)
        # Cap energy to something bit above the mass
        np.clip(e, 1.01 * m, None, out=e)
    if alpha is not None and num is not None:
        # Rotate around a random orthogonal axis by alpha
        r = _random_orthogonal(p3, rng)
        p3 = _rotate_axis(p3, r, _maybe_sample(alpha, (num_events, num), rng))
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


def _lorvecfromnumpy(x, y, z, t, behavior, name="LorentzVector"):
    return ak.zip(
        {"x": x, "y": y, "z": z, "t": t}, with_name="LorentzVector",
        behavior=behavior)


def _from_regular(array, axis=1, highlevel=True):
    """from_regular with multiple axis at once"""
    if isinstance(axis, int):
        return ak.from_regular(array, axis, highlevel)
    for i in axis:
        array = ak.from_regular(array, i, highlevel)
    return array


@chunked_calls("lep", True, 10000)
def sonnenschein(lep, antilep, b, antib, met, mwp=80.3, mwm=80.3, mt=172.5,
                 mat=172.5, num_smear=None, energyfl=None, energyfj=None,
                 alphal=None, alphaj=None, hist_mlb=None, rng=None):
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
    mwp, mwm -- Mass of the W+ and W- bosons. Either a number, an array with a
                number for each event or a histogram, to sample from
    mt, mat  -- Same as mwp/mwm for the top quark and antiquark
    num_smear -- Number of times an event is smeared. If None, smearing is off
    energyfl -- Histogram giving Ereco/Egen for the leptons. If None, lepton
                energy won't be smeared
    energyfj -- Same as energyfl for bottom quarks
    alphal -- Histogram giving the angle between reco and gen leptons. If None,
              lepton angles won't be smeared
    alphaj -- Same as alphal for bottom quarks
    hist_mlb -- uproot histogram of the lepton-bottom-quark-mass distribution.
                Is needed, if num_smear is not None
    rng -- A numpy.random.BitGenerator, if None a new one will be used
    """

    if rng is None:
        rng = np.random.default_rng()

    if lep.ndim > 1:
        # Get rid of jagged dimension, as we have one particle per row and
        # variable
        lep = ak.flatten(lep)
        antilep = ak.flatten(antilep)
        b = ak.flatten(b)
        antib = ak.flatten(antib)
        met = ak.flatten(met)

    # Allow num_smear to be 0
    if num_smear == 0:
        num_smear = None

    behav = lep.behavior
    # Use 2d numpy arrays. Use first axis for events, second for smearing
    num_events = len(lep)
    lE, lx, ly, lz = _smear(lep, energyfl, alphal, num_smear, rng)
    alE, alx, aly, alz = _smear(antilep, energyfl, alphal, num_smear, rng)
    bE, bx, by, bz = _smear(b, energyfj, alphaj, num_smear, rng)
    abE, abx, aby, abz = _smear(antib, energyfj, alphaj, num_smear, rng)
    # Even if num_smear is None, we have a smear axis. Update num_smear
    num_smear = max(lE.shape[1], bE.shape[1])
    # Unpack MET compontents and also propagate smearing to it
    METx = (met.x[:, None] - lx + lep.x[:, None] - alx + antilep.x[:, None]
                           - bx + b.x[:, None] - abx + antib.x[:, None])
    METx = np.asarray(METx)
    METy = (met.y[:, None] - ly + lep.y[:, None] - aly + antilep.y[:, None]
                           - by + b.y[:, None] - aby + antib.y[:, None])
    METy = np.asarray(METy)

    mwp = _maybe_sample(mwp, (num_events, 1), rng)
    mwm = _maybe_sample(mwm, (num_events, 1), rng)
    mat = _maybe_sample(mat, (num_events, 1), rng)
    mt = _maybe_sample(mt, (num_events, 1), rng)
    # Compute masses, make sure they are real
    lp = np.sqrt(lx**2 + ly**2 + lz**2)
    lE = np.where(lE < lp, lp, lE)
    ml = np.sqrt(lE**2 - lp**2)
    alp = np.sqrt(alx**2 + aly**2 + alz**2)
    alE = np.where(alE < alp, alp, alE)
    mal = np.sqrt(alE**2 - alp**2)
    bp = np.sqrt(bx**2 + by**2 + bz**2)
    bE = np.where(bE < bp, bp, bE)
    mb = np.sqrt(bE**2 - bp**2)
    abp = np.sqrt(abx**2 + aby**2 + abz**2)
    abE = np.where(abE < abp, abp, abE)
    mab = np.sqrt(abE**2 - abp**2)
    del lp, alp, bp, abp

    if hist_mlb is not None:
        mlab = np.sqrt((lE + abE)**2 - (lx + abx)**2
                       - (ly + aby)**2 - (lz + abz)**2)
        malb = np.sqrt((alE + bE)**2 - (alx + bx)**2
                       - (aly + by)**2 - (alz + bz)**2)
        # Set over and underflow to 0
        values, edges = hist_mlb.to_numpy(flow=True)
        values = np.r_[0, values[1:-1], 0]
        plab = values[np.digitize(mlab, edges) - 1]
        palb = values[np.digitize(malb, edges) - 1]
        weights = plab * palb
    elif num_smear is not None and num_smear > 1:
        raise ValueError("Smearing is enabled but got None for hist_mlb")
    else:
        weights = np.ones_like(lE)

    a1 = ((bE + alE) * (mwp ** 2 - mal ** 2)
          - alE * (mt ** 2 - mb ** 2 - mal ** 2) + 2 * bE * alE ** 2
          - 2 * alE * (bx * alx + by * aly + bz * alz))
    del mb
    a2 = 2 * (bE * alx - alE * bx)
    a3 = 2 * (bE * aly - alE * by)
    a4 = 2 * (bE * alz - alE * bz)

    b1 = ((abE + lE) * (mwm ** 2 - ml ** 2)
          - lE * (mat ** 2 - mab ** 2 - ml ** 2) + 2 * abE * lE ** 2
          - 2 * lE * (abx * lx + aby * ly + abz * lz))
    del mab
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
    del mal

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
    del ml

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
    del h0, h1, h2, h3, h4, h
    vpx = roots.real
    is_real = abs(roots.imag) < 10 ** -6
    del roots

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

    # Neutrino mass is assumed to be 0
    vE = np.sqrt(vpx ** 2 + vpy ** 2 + vpz ** 2)
    vbarE = np.sqrt(vbarpx ** 2 + vbarpy ** 2 + vbarpz ** 2)

    is_real = _from_regular(is_real, axis=(1, 2))
    v = _lorvecfromnumpy(vpx, vpy, vpz, vE, behav)[is_real]
    av = _lorvecfromnumpy(vbarpx, vbarpy, vbarpz, vbarE, behav)[is_real]
    b = _lorvecfromnumpy(bx, by, bz, bE, behav)
    ab = _lorvecfromnumpy(abx, aby, abz, abE, behav)
    lep = _lorvecfromnumpy(lx, ly, lz, lE, behav)
    alep = _lorvecfromnumpy(alx, aly, alz, alE, behav)
    wp = v + alep
    wm = av + lep
    t = wp + b
    at = wm + ab

    # Reduce solution axis and pick the solution with the smallest mtt
    # Note that there is a very small probability of having a weight of zero
    # and a root that looks real, we don't want these
    has_solution = ak.any(is_real, axis=2) & (weights != 0)
    min_mtt = ak.argmin((t + at).mass, axis=2, keepdims=True)
    t = ak.flatten(t[min_mtt][has_solution], axis=2)
    at = ak.flatten(at[min_mtt][has_solution], axis=2)
    weights = _from_regular(weights, axis=1)[has_solution]

    # Undo smearing by averaging, slice with has_solution again to not get
    # vectors with zeros if there is no solution
    sum_weights = ak.where(ak.num(weights) > 0, ak.sum(weights, axis=1), 1.)
    has_solution = ak.any(has_solution, axis=1, keepdims=True)

    t = ak.zip(
        {f: ak.sum(t[f] * weights, axis=1, keepdims=True) for f in t.fields},
        with_name="LorentzVector")[has_solution] / sum_weights
    at = ak.zip(
        {f: ak.sum(at[f] * weights, axis=1, keepdims=True) for f in at.fields},
        with_name="LorentzVector")[has_solution] / sum_weights

    # Top mass got changed by the averaging. Set to input mass again
    t["t"] = np.sqrt(mt ** 2 + t.rho2)
    at["t"] = np.sqrt(mat ** 2 + at.rho2)

    return t, at


if __name__ == '__main__':
    from coffea.nanoevents.methods import vector
    # test case:
    lep = ak.Array({
        "t": [165.33320], "x": [26.923591], "y": [16.170616],
        "z": [-162.3227]}, with_name="LorentzVector", behavior=vector.behavior)
    antilep = ak.Array({
        "t": [49.290821], "x": [-34.58441], "y": [-13.27824],
        "z": [-32.51431]}, with_name="LorentzVector", behavior=vector.behavior)
    b = ak.Array({
        "t": [205.54469], "x": [99.415420], "y": [-78.89404],
        "z": [-161.6102]}, with_name="LorentzVector", behavior=vector.behavior)
    antib = ak.Array({
        "t": [362.82086], "x": [-49.87086], "y": [91.930526],
        "z": [-347.3868]}, with_name="LorentzVector", behavior=vector.behavior)
    nu = ak.Array({
        "t": [70.848953], "x": [34.521587], "y": [-51.23474],
        "z": [-6.555319]}, with_name="LorentzVector", behavior=vector.behavior)
    antinu = ak.Array({
        "t": [13.760989], "x": [11.179965], "y": [-3.844941],
        "z": [7.0419898]}, with_name="LorentzVector", behavior=vector.behavior)
    sump4 = nu + antinu
    met = ak.Array({
        "pt": sump4.pt, "eta": [0.], "phi": sump4.phi, "mass": [0.]},
        with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
    top, antitop = sonnenschein(lep, antilep, b, antib, met)
    print("MC Truth:", (antilep + nu + b)[0],
                       (lep + antinu + antib)[0])
    print("Reconstructed:", top, antitop)
    # Reconstructed: [{x: 93.5, y: -160, z: -213, t: 331}]
    #                [{x: -5.88, y: 121, z: -508, t: 550}]

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
    sonnenschein(lep, antilep, b, antib, met)
    print("Took {} s for {} events".format(time.time() - t, n))
