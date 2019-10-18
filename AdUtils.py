from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from coffea.util import awkward
from coffea.util import numpy as np
import uproot_methods
from uproot_methods.classes.TLorentzVector import PtEtaPhiMassLorentzVector as LV
from collections import OrderedDict

#Additional functions which should probably exist in coffea (currently just concatenate)

def concatenate(arrays):
  flatarrays = [a._content for a in arrays]
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
    
    # create empty arrays  for each column with the most general dtype
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
