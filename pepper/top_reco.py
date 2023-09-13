import numpy as np
# from coffea.analysis_objects import JaggedCandidateArray as Jca
import awkward as ak
import sys
sys.path.append("..")


def topreco(data):

        jets = data["Jet"]

        results = {}

        MET = data["MET"]
        jet_indices = ak.local_index(jets, axis=1)

        # Generate the indices of every pair;
        # indices because we'll be removing these elements later.
        #lepton_pairs = ak.argcombinations(leptons, 2)
        # Find the pair with SF,OS and mass closest to Z.
        #lepton_pairs_OSSF = lepton_pairs[cut_defs.os_sf_lepton_pair(
        #                                    leptons[lepton_pairs.slot0],
        #                                    leptons[lepton_pairs.slot1])]
        #closest_pair = lepton_pairs_OSSF[ak.argmin(np.abs((
        #    leptons[lepton_pairs_OSSF.slot0] +
        #    leptons[lepton_pairs_OSSF.slot1]).mass - 91.2), axis=1,
        #                                            keepdims=True)]
        # Get index of spare lepton (not from Z)
        # Padding with index 0 the elements where a match with Z
        # is not found. .
        #idxs_nonZ = lepton_indices[
        #    ~(lepton_indices == ak.flatten(ak.fill_none(
        #        ak.pad_none(closest_pair.slot0, 1), 0)))
        #]
        #idxs_nonZ = idxs_nonZ[
        #    ~(idxs_nonZ == ak.flatten(ak.fill_none(
        #        ak.pad_none(closest_pair.slot1, 1), 0)))
        #]
        lep_nonZ = data["Lepton"]

        # Convert to awkward
        ### leptons_ak1 = ak.from_ak(leptons)
        ## idxs_nonZ_ak1 = ak.from_ak(idxs_nonZ)
        ## leptons_nonZ_ak1 = leptons_ak1[idxs_nonZ_ak1]

        lep_nonZ = lep_nonZ[:, 0]

        # Get neutrinos
        #neutrinos = neutrinosolutions(lep_nonZ, MET)
        neutrinos = data["Neutrino1"]

        # Top's resolutions (GeV), computed with functions
        # "get*Res" in this script
        # Below is the sigma of a gaussian which fits the distributions
        rtl = 23.08
        rth = 31.37
        mt = 172.69

        #need at least 3 jets to reconstruct the hadronic top: in case
        #of 2 jets, only the leptonic top is reconstructed. In case of
        #exactly 3 jets, evaluate both and take the lowest chi2. When
        #there are 4 or more jets, reconstruct both
        has2jets = ak.num(jets, axis=1) == 2
        has3jets = ak.num(jets, axis=1) == 3
        has4jets = ak.num(jets, axis=1) >= 4

        #2 jets: assume it is tZq and only reconstruct leptonic top
        exactly_two_jets = ak.mask(jets, has2jets)
        twojets = ak.combinations(exactly_two_jets, 2)

        comb_twojets, comb_lepton, comb_nu = ak.unzip(ak.cartesian(
            [twojets, lep_nonZ, neutrinos], axis=1))
        comb_j1, comb_j2 = ak.unzip(comb_twojets)

        chi_square_1a = pow(((comb_lepton + comb_nu + comb_j1).mass - mt)/rtl, 2)

        minIdx1a = ak.argmin(chi_square_1a, axis=1, keepdims=True)
        jtop_1a = comb_j1[minIdx1a]
        leptop_1a = comb_lepton[minIdx1a]
        nutop_1a = comb_nu[minIdx1a]
        toplep_1a = jtop_1a + leptop_1a + nutop_1a

        chi_square_1b = pow(((comb_lepton + comb_nu + comb_j2).mass - mt)/rtl, 2)

        minIdx1b = ak.argmin(chi_square_1b, axis=1, keepdims=True)
        jtop_1b = comb_j2[minIdx1b]
        leptop_1b = comb_lepton[minIdx1b]
        nutop_1b = comb_nu[minIdx1b]
        toplep_1b = jtop_1b + leptop_1b + nutop_1b

        chi_square1a_ = chi_square_1a[minIdx1a]
        chi_square1b_ = chi_square_1b[minIdx1b]

        chi_square_2jets = ak.concatenate([chi_square1a_, chi_square1b_],
                                        axis=1)

        minIdx_2jets = ak.argmin(chi_square_2jets, axis=1, keepdims=True)
        minIdx_2jets = ak.fill_none(ak.pad_none(minIdx_2jets, 1), 0)

        # Extract topMass
        toplep_cand_2jets = ak.concatenate([toplep_1a, toplep_1b], axis=1)
        toplep_cand_2jets = ak.mask(toplep_cand_2jets, ak.num(toplep_cand_2jets)>0)
        toplep_2jets = toplep_cand_2jets[minIdx_2jets]

        #3 jets: reconstruct had and lep top, take the best chi
        exactly_three_jets = ak.mask(jets, has3jets)
        threejets = ak.combinations(exactly_three_jets, 3)

        comb_threejets, comb_lepton, comb_nu = ak.unzip(ak.cartesian(
            [threejets, lep_nonZ, neutrinos], axis=1))
        comb_j1, comb_j2, comb_j3 = ak.unzip(comb_threejets)

        #first try with leptonic top
        chi_square_1a = pow(((comb_lepton + comb_nu + comb_j1).mass - mt)/rtl, 2)

        minIdx1a = ak.argmin(chi_square_1a, axis=1, keepdims=True)
        jtop_1a = comb_j1[minIdx1a]
        leptop_1a = comb_lepton[minIdx1a]
        nutop_1a = comb_nu[minIdx1a]
        toplep_1a = jtop_1a + leptop_1a + nutop_1a

        chi_square_1b = pow(((comb_lepton + comb_nu + comb_j2).mass - mt)/rtl, 2)

        minIdx1b = ak.argmin(chi_square_1b, axis=1, keepdims=True)
        jtop_1b = comb_j2[minIdx1b]
        leptop_1b = comb_lepton[minIdx1b]
        nutop_1b = comb_nu[minIdx1b]
        toplep_1b = jtop_1b + leptop_1b + nutop_1b

        chi_square_1c = pow(((comb_lepton + comb_nu + comb_j3).mass - mt)/rtl, 2)

        minIdx1c = ak.argmin(chi_square_1c, axis=1, keepdims=True)
        jtop_1c = comb_j2[minIdx1c]
        leptop_1c = comb_lepton[minIdx1c]
        nutop_1c = comb_nu[minIdx1c]
        toplep_1c = jtop_1c + leptop_1c + nutop_1c

        #and now with the hadronic top
        chi_square_2 = pow(((comb_j1 + comb_j2 + comb_j3).mass - mt)/rth, 2)

        minIdx2 = ak.argmin(chi_square_2, axis=1, keepdims=True)
        j1top_2 = comb_j1[minIdx2]
        j2top_2 = comb_j2[minIdx2]
        j3top_2 = comb_j3[minIdx2]

        tophad_2 = j1top_2 + j2top_2 + j3top_2

        chi_square1a_ = chi_square_1a[minIdx1a]
        chi_square1b_ = chi_square_1b[minIdx1b]
        chi_square1c_ = chi_square_1c[minIdx1c]

        chi_square_3jets_lep = ak.concatenate([chi_square1a_,
                                        chi_square1b_,
                                        chi_square1c_],
                                        axis=1)

        chi_square_3jets_had = ak.concatenate([chi_square_2[minIdx2], ], axis=1)
        chi_square_3jets=ak.concatenate([chi_square_3jets_lep, chi_square_3jets_had], axis=1)
        minIdx_3jets = ak.argmin(chi_square_3jets, axis=1, keepdims=True)

        condition_had = (ak.argmin(chi_square_3jets_had, axis=1, keepdims=True)==ak.argmin(chi_square_3jets, axis=1, keepdims=True))
        condition_lep = (ak.argmin(chi_square_3jets_lep, axis=1, keepdims=True)==ak.argmin(chi_square_3jets, axis=1, keepdims=True))

        minIdx_3jets_lep = ak.where(condition_lep, ak.argmin(chi_square_3jets_lep, axis=1, keepdims=True), 0)
        minIdx_3jets_lep = ak.fill_none(ak.pad_none(minIdx_3jets_lep, 1), 0)

        minIdx_3jets_had = ak.where(condition_had, ak.argmin(chi_square_3jets_had, axis=1, keepdims=True), 0)
        minIdx_3jets_had = ak.fill_none(ak.pad_none(minIdx_3jets_had, 1), 0)

        # Extract topMass
        toplep_cand_3jets = ak.concatenate([toplep_1a, toplep_1b, toplep_1c], axis=1)
        toplep_cand_3jets = ak.mask(toplep_cand_3jets, ak.num(toplep_cand_3jets)>0)

        #only take events where the leptonic chi square is the minimum
        toplep_cand_3jets = ak.mask(toplep_cand_3jets, ak.flatten(condition_lep))
        toplep_3jets = toplep_cand_3jets[minIdx_3jets_lep]

        tophad_cand_3jets = ak.concatenate([tophad_2, ], axis=1)
        tophad_cand_3jets = ak.mask(tophad_cand_3jets, ak.num(tophad_cand_3jets)>0)

        #only take events where the hadronic chi square is the minimum
        tophad_cand_3jets = ak.mask(tophad_cand_3jets, ak.flatten(condition_had))
        tophad_3jets = tophad_cand_3jets[minIdx_3jets_had]

        #at least 4 jets: always reconstruct lep and had top
        #jets = ak.pad_none(jets, 1)
        atleast_four_jets = ak.mask(jets, has4jets)
        fourjets = ak.combinations(atleast_four_jets, 4)

        comb_fourjets, comb_lepton, comb_nu = ak.unzip(ak.cartesian(
            [fourjets, lep_nonZ, neutrinos], axis=1))
        comb_j1, comb_j2, comb_j3, comb_j4 = ak.unzip(comb_fourjets)

        #fourjets_idx = ak.argcombinations(jets, 4)
        #nu_idx = ak.argcombinations(neutrinos, 1)
        #comb_fourjets_idx, comb_lepton_idx, comb_nu_idx = ak.unzip(
        #    ak.cartesian([fourjets_idx, idxs_nonZ, nu_idx], axis=1))
        #comb_j1_idx, comb_j2_idx, comb_j3_idx, comb_j4_idx = ak.unzip(comb_fourjets_idx)

        #minimize leptonic and hadronic top together
        chi_square1a = (
            pow((((comb_lepton + comb_nu + comb_j1).mass - mt)/rtl), 2) +
            pow((((comb_j2 + comb_j3 + comb_j4).mass - mt)/rth), 2)
        )
        minIdx1a = ak.argmin(chi_square1a, axis=1, keepdims=True)
        j1top_1a = comb_j1[minIdx1a]
        j2top_1a = comb_j2[minIdx1a]
        j3top_1a = comb_j3[minIdx1a]
        j4top_1a = comb_j4[minIdx1a]
        leptop_1a = comb_lepton[minIdx1a]
        nutop_1a = comb_nu[minIdx1a]
        toplep_1a = j1top_1a + leptop_1a + nutop_1a
        tophad_1a = j2top_1a + j3top_1a + j4top_1a

        chi_square1b = (
            pow((((comb_lepton + comb_nu + comb_j2).mass - mt)/rtl), 2) +
            pow((((comb_j1 + comb_j3 + comb_j4).mass - mt)/rth), 2)
        )
        minIdx1b = ak.argmin(chi_square1b, axis=1, keepdims=True)
        j1top_1b = comb_j1[minIdx1b]
        j2top_1b = comb_j2[minIdx1b]
        j3top_1b = comb_j3[minIdx1b]
        j4top_1b = comb_j4[minIdx1b]
        leptop_1b = comb_lepton[minIdx1b]
        nutop_1b = comb_nu[minIdx1b]
        toplep_1b = j2top_1b + leptop_1b + nutop_1b
        tophad_1b = j1top_1b + j3top_1b + j4top_1b

        chi_square1c = (
            pow((((comb_lepton + comb_nu + comb_j3).mass - mt)/rtl), 2) +
            pow((((comb_j1 + comb_j2 + comb_j4).mass - mt)/rth), 2)
        )
        minIdx1c = ak.argmin(chi_square1c, axis=1, keepdims=True)
        j1top_1c = comb_j1[minIdx1c]
        j2top_1c = comb_j2[minIdx1c]
        j3top_1c = comb_j3[minIdx1c]
        j4top_1c = comb_j4[minIdx1c]
        leptop_1c = comb_lepton[minIdx1c]
        nutop_1c = comb_nu[minIdx1c]
        toplep_1c = j3top_1c + leptop_1c + nutop_1c
        tophad_1c = j1top_1c + j2top_1c + j4top_1c

        chi_square1d = (
            pow((((comb_lepton + comb_nu + comb_j4).mass - mt)/rtl), 2) +
            pow((((comb_j1 + comb_j2 + comb_j3).mass - mt)/rth), 2)
        )
        minIdx1d = ak.argmin(chi_square1d, axis=1, keepdims=True)
        j1top_1d = comb_j1[minIdx1d]
        j2top_1d = comb_j2[minIdx1d]
        j3top_1d = comb_j3[minIdx1d]
        j4top_1d = comb_j4[minIdx1d]
        leptop_1d = comb_lepton[minIdx1d]
        nutop_1d = comb_nu[minIdx1d]
        toplep_1d = j4top_1d + leptop_1d + nutop_1d
        tophad_1d = j1top_1d + j2top_1d + j3top_1d

        chi_square1a_ = chi_square1a[minIdx1a]
        chi_square1b_ = chi_square1b[minIdx1b]
        chi_square1c_ = chi_square1c[minIdx1c]
        chi_square1d_ = chi_square1d[minIdx1d]

        chi_square = ak.concatenate([chi_square1a_,
                                    chi_square1b_,
                                    chi_square1c_,
                                    chi_square1d_],
                                    axis=1)

        minIdx = ak.argmin(chi_square, axis=1, keepdims=True)
        minIdx = ak.fill_none(ak.pad_none(minIdx, 1), 0)

        # Extract topMass
        toplep_cand = ak.concatenate([toplep_1a, toplep_1b, toplep_1c, toplep_1d], axis=1)
        tophad_cand = ak.concatenate([tophad_1a, tophad_1b, tophad_1c, tophad_1d], axis=1)

        toplep_cand = ak.mask(toplep_cand, ak.num(toplep_cand)>0)
        toplep = toplep_cand[minIdx]

        tophad_cand = ak.mask(tophad_cand, ak.num(tophad_cand)>0)
        tophad = tophad_cand[minIdx]

        dummy_array_ptmass = np.empty(len(jets))
        dummy_array_ptmass.fill(-1)

        dummy_array_etaphi = np.empty(len(jets))
        dummy_array_etaphi.fill(0)

        #import pdb
        #pdb.set_trace()

        if len(dummy_array_ptmass) != 0:
            dummy_array_ptmass = ak.unflatten(ak.Array(dummy_array_ptmass), counts=1)

        if len(dummy_array_etaphi) != 0:
            dummy_array_etaphi = ak.unflatten(ak.Array(dummy_array_etaphi), counts=1)

        condition_lep = ak.fill_none(condition_lep, False)
        condition_had = ak.fill_none(condition_had, False)

        has3jets_lep = np.logical_and(has3jets, condition_lep)
        has3jets_had = np.logical_and(has3jets, condition_had)

        has2or4jets = np.logical_or(has2jets, has4jets)

        has_leptop = np.logical_or(has2or4jets, has3jets_lep)
        has_hadtop = np.logical_or(has4jets, has3jets_had)

        leptonicTop = ak.concatenate([toplep_2jets, toplep_3jets, toplep], axis=1)
        hadronicTop = ak.concatenate([tophad_3jets, tophad], axis=1)

        allTop = ak.concatenate([leptonicTop, hadronicTop], axis=1)

        results["mtoplep"] = ak.where(ak.flatten(has_leptop), leptonicTop.mass, dummy_array_ptmass)
        results["mtophad"] = ak.where(ak.flatten(has_hadtop), hadronicTop.mass, dummy_array_ptmass)
        results["pttoplep"] = ak.where(ak.flatten(has_leptop), leptonicTop.pt, dummy_array_ptmass)
        results["pttophad"] = ak.where(ak.flatten(has_hadtop), hadronicTop.pt, dummy_array_ptmass)
        results["etatoplep"] = ak.where(ak.flatten(has_leptop), leptonicTop.eta, dummy_array_etaphi)
        results["etatophad"] = ak.where(ak.flatten(has_hadtop), hadronicTop.eta, dummy_array_etaphi)
        results["phitoplep"] = ak.where(ak.flatten(has_leptop), leptonicTop.phi, dummy_array_etaphi)
        results["phitophad"] = ak.where(ak.flatten(has_hadtop), hadronicTop.phi, dummy_array_etaphi)

        chi_square = ak.pad_none(chi_square, 1)
        chi_square_3jets = ak.pad_none(chi_square_3jets, 1)
        chi_square_2jets = ak.pad_none(chi_square_2jets, 1)

        chiSquare = ak.concatenate([chi_square[minIdx], chi_square_3jets[minIdx_3jets], chi_square_2jets[minIdx_2jets]], axis=1)
        results["chisquare"] = chiSquare

        #take top with highest pT
        max_pt_indices = ak.argmax(allTop.pt, axis=1)
        allTop = allTop[ak.singletons(max_pt_indices)]
        results["TopHighestPt"] = allTop

        return results


