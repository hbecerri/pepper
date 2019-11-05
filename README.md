# DESY ttbarBSM coffea

A framework for analysing ttbarBSM events in nanoaod. Requires coffea (https://coffeateam.github.io/coffea/index.html) and run on CMSSW_10_2_16_UL
To run, simply use: python3 main.py

File Structure:

-main.py: reads in nanoaod to coffea JaggedArrays, performs cuts, calls reconstruction algorithm, and plots cutflows and mttbar masses

-AdUtils.py: Implements additional utils not yet available elsewhere: an equivalent of JaggedArray.concatenate for coffea's JCAs and an equivalent of np.where for the same

-KinRecoSonnenschein.py: Sonnenschein's kinematic reconstruction algoritm: https://arxiv.org/pdf/hep-ph/0603011v3.pdf

-betchartkinreco.py: Betchart's kinematic reconstruction algoritm: https://arxiv.org/pdf/1305.1878.pdf Not yet fully implemented

-genHists.py Produces hists of Gen Level information (currently just the Mlb distribution) to be used for selecting the correct jets, etc.

-gencomp.py: Old version of main.py, designed to run on ttbar samples and compare Mttbar for Gen and Reco
