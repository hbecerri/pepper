from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from coffea.analysis_objects import JaggedTLorentzVectorArray
from coffea.util import awkward
from coffea.util import numpy
from coffea.util import numpy as np
import uproot
import coffea.processor as processor
import matplotlib
import matplotlib.pyplot as plt

class ttbarProcessor(processor.ProcessorABC):
  def __init__(self):
    lbarbmass_axis = hist.Bin("Mlbarb", "Mlbarb [GeV]", 100, 0, 200)
    lbbarmass_axis = hist.Bin("Mlbbar", "Mlbbar [GeV]", 100, 0, 200)
    
    self._accumulator = processor.dict_accumulator({
      'Mlbarb': hist.Hist("Counts", lbarbmass_axis),
      'Mlbbar': hist.Hist("Counts", lbbarmass_axis)
    })
  
  @property
  def accumulator(self):
    return self._accumulator
  
  def process(self, df):
    output = self.accumulator.identity()
    
    dataset = df['dataset']
   
    genpart = JaggedCandidateArray.candidatesfromcounts(
      df['nGenPart'],
      pt=df['GenPart_pt'].content,
      eta=df['GenPart_eta'].content,
      phi=df['GenPart_phi'].content,
      mass=np.where((np.abs(df['GenPart_pdgId'].content)==5)&(df['GenPart_mass'].content==0),4.8, df['GenPart_mass'].content), #mass is only stored for masses greater than 10GeV- this is a fudge to set the b mass to the right value if not stored
      pdgId=df['GenPart_pdgId'].content,
      motherIdx=df['GenPart_genPartIdxMother'].content,
      status=df['GenPart_status'].content,
      statusFlags=df['GenPart_statusFlags'].content,
      )
    genMET= awkward.Table(pt=df['GenMET_pt'], phi=df['GenMET_phi'])
    
    genlep=genpart[((genpart.pdgId==11)|(genpart.pdgId==13))&(genpart[genpart.motherIdx].pdgId==-24)&(genpart[genpart[genpart.motherIdx].motherIdx].pdgId==-6)&(genpart.statusFlags>>12&1==1)]
    genantilep=genpart[((genpart.pdgId==-11)|(genpart.pdgId==-13))&(genpart[genpart.motherIdx].pdgId==24)&(genpart[genpart[genpart.motherIdx].motherIdx].pdgId==6)&(genpart.statusFlags>>12&1==1)]
    genb=genpart[(genpart.pdgId==5)&(genpart[genpart.motherIdx].pdgId==6)&(genpart.statusFlags>>12&1==1)]
    genantib=genpart[(genpart.pdgId==-5)&(genpart[genpart.motherIdx].pdgId==-6)&(genpart.statusFlags>>12&1==1)&(genpart.statusFlags>>7&1==1)]
    twogenleps=(genlep.counts>0)&(genantilep.counts>0)
    genlep=genlep[twogenleps]
    genantilep=genantilep[twogenleps]
    genb=genb[twogenleps]
    genantib=genantib[twogenleps]
    Mlbarb=(genantilep['p4']+genb['p4']).mass.content
    Mlbbar=(genlep['p4']+genantib['p4']).mass.content
    
    output['Mlbarb'].fill(Mlbarb=Mlbarb)
    output['Mlbbar'].fill(Mlbbar=Mlbbar)
    return output
    
  def postprocess(self, accumulator):
    return accumulator

fileset = {'Jets': ["/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/250000/5239A9C7-9C9D-2B4B-B337-374252B493F0.root"]}

output = processor.run_uproot_job(fileset,
                                 treename='Events',
                                 processor_instance=ttbarProcessor(),
                                 executor=processor.iterative_executor,
                                 executor_args={'workers':4},
                                 chunksize = 500000)

outfile = uproot.recreate('GenHists.root')
outfile['Mlbbar'] = hist.export1d(output['Mlbbar'])
outfile['Mlbarb'] = hist.export1d(output['Mlbarb'])
outfile.close()
