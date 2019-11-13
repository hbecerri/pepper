from collections import OrderedDict

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from coffea.util import awkward
from coffea.util import numpy as np
import uproot_methods
from uproot_methods.classes.TLorentzVector import PtEtaPhiMassLorentzVector as LV

#Additional functions which should probably exist in coffea

def concatenate(arrays):
    flatarrays = [a.flatten() for a in arrays]
    n_arrays = len(arrays)

    # the first step is to get the starts and stops for the stacked structure
    counts = np.vstack([a.counts for a in arrays])
    flat_counts = counts.T.flatten()
    offsets = awkward.JaggedArray.counts2offsets(flat_counts)
    starts, stops = offsets[:-1], offsets[1:]

    n_content = sum([len(a) for a in flatarrays])
    content_type = type(flatarrays[0])

    # get masks for each of the arrays so we can fill the stacked content array at the right indices
    def get_mask(array_index):
        working_array = np.zeros(n_content + 1, dtype=awkward.JaggedArray.INDEXTYPE)
        starts_i = starts[i::n_arrays]
        stops_i = stops[i::n_arrays]
        not_empty = starts_i != stops_i
        working_array[starts_i[not_empty]] += 1
        working_array[stops_i[not_empty]] -= 1
        mask = np.array(np.cumsum(working_array)[:-1], dtype=awkward.JaggedArray.MASKTYPE)
        return mask
    
    if content_type == np.ndarray:
        content = np.zeros(n_content, dtype=get_dtype(flatarrays))
        for i in range(n_arrays):
            content[get_mask(i)] = flatarrays[i]

    elif isinstance(flatarrays[0], awkward.array.table.Table):
        tablecontent = OrderedDict()
        #tables = [a._content for a in flatarrays]

        # make sure all flatarrays have the same columns
        for i in range(len(flatarrays) - 1):
            if set(flatarrays[i]._contents) != set(flatarrays[i+1]._contents):
                raise ValueError("cannot concatenate flatarrays with different fields")
        
        # create empty arrays    for each column with the most general dtype
        for n in flatarrays[0]._contents:
            if not(n=='p4'):
                dtype = np.dtype(flatarrays[0]._contents[n][0])
                tablecontent[n] = np.zeros(n_content, dtype=dtype)

        for i in range(n_arrays):
            mask = get_mask(i)
            for n in flatarrays[0]._contents:
                if not(n=='p4'):
                    tablecontent[n][mask] = flatarrays[i][n]

        tablecontent['p4'] = uproot_methods.TLorentzVectorArray.from_ptetaphim(tablecontent['__fast_pt'], tablecontent['__fast_eta'], tablecontent['__fast_phi'], tablecontent['__fast_mass']) 

        content = awkward.array.table.Table(**tablecontent)

    else:
        raise NotImplementedError("concatenate with axis=1 is not implemented for " + type(arrays[0]).__name__)

    out= JaggedCandidateArray(starts[::n_arrays], stops[n_arrays-1::n_arrays], content)
    #print(out.type)
    return(out)


def LVwhere(condition, x, y):
    pt=np.empty(len(condition), dtype=float)
    eta=np.empty(len(condition), dtype=float)
    phi=np.empty(len(condition), dtype=float)
    mass=np.empty(len(condition), dtype=float)

    #condition=condition.astype(awkward.JaggedArray.MASKTYPmass)
    pt[condition]=x.pt
    pt[~condition]=y.pt
    eta[condition]=x.eta
    eta[~condition]=y.eta
    phi[condition]=x.phi
    phi[~condition]=y.phi
    mass[condition]=x.mass
    mass[~condition]=y.mass
    out=JaggedCandidateArray.candidatesfromcounts(np.ones_like(condition), pt=pt,eta=eta,phi=phi,mass=mass)
    return out


def Pairswhere(condition, x, y):
    counts=np.where(condition, x.counts, y.counts)
    pt0=np.empty(counts.sum(), dtype=float)
    eta0=np.empty(counts.sum(), dtype=float)
    phi0=np.empty(counts.sum(), dtype=float)
    mass0=np.empty(counts.sum(), dtype=float)
    
    pt1=np.empty(counts.sum(), dtype=float)
    eta1=np.empty(counts.sum(), dtype=float)
    phi1=np.empty(counts.sum(), dtype=float)
    mass1=np.empty(counts.sum(), dtype=float)
    
    offsets = awkward.JaggedArray.counts2offsets(counts)
    starts, stops = offsets[:-1], offsets[1:]
    
    working_array = np.zeros(counts.sum()+1, dtype=awkward.JaggedArray.INDEXTYPE)
    xstarts=starts[condition]
    xstops=stops[condition]
    not_empty = xstarts != xstops
    working_array[xstarts[not_empty]] += 1
    working_array[xstops[not_empty]] -= 1
    mask = np.array(np.cumsum(working_array)[:-1], dtype=awkward.JaggedArray.MASKTYPE)
    
    pt0[mask]=x[condition].i0.pt.flatten()
    pt0[~mask]=y[~condition].i0.pt.flatten()
    eta0[mask]=x[condition].i0.eta.flatten()
    eta0[~mask]=y[~condition].i0.eta.flatten()
    phi0[mask]=x[condition].i0.phi.flatten()
    phi0[~mask]=y[~condition].i0.phi.flatten()
    mass0[mask]=x[condition].i0.mass.flatten()
    mass0[~mask]=y[~condition].i0.mass.flatten()
    out0=JaggedCandidateArray.candidatesfromcounts(counts, pt=pt0,eta=eta0,phi=phi0,mass=mass0)
    
    pt1[mask]=x[condition].i1.pt.flatten()
    pt1[~mask]=y[~condition].i1.pt.flatten()
    eta1[mask]=x[condition].i1.eta.flatten()
    eta1[~mask]=y[~condition].i1.eta.flatten()
    phi1[mask]=x[condition].i1.phi.flatten()
    phi1[~mask]=y[~condition].i1.phi.flatten()
    mass1[mask]=x[condition].i1.mass.flatten()
    mass1[~mask]=y[~condition].i1.mass.flatten()
    out1=JaggedCandidateArray.candidatesfromcounts(counts, pt=pt1,eta=eta1,phi=phi1,mass=mass1)
    return out0, out1
