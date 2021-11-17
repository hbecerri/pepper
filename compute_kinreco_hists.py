#!/usr/bin/env python3

import os
import numpy as np
import awkward as ak
import uproot
import coffea

import pepper
from pepper.misc import coffeahist2hist


class Processor(pepper.ProcessorTTbarLL):
    def __init__(self, config, destdir):
        if "reco_algorithm" in config:
            del config["reco_algorithm"]
        if "reco_info_file" in config:
            del config["reco_info_file"]
        if "blinding_denom" in config:
            del config["blinding_denom"]
        config["compute_systematics"] = False
        super().__init__(config, None)

    def preprocess(self, datasets):
        return {"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8":
                datasets["TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"]}

    @property
    def accumulator(self):
        mlb_axis = coffea.hist.Bin(
            "mlb", r"$m_{\mathrm{lb}}$ (GeV)", 200, 0, 200)
        mlb = coffea.hist.Hist("Counts", mlb_axis)
        # W mass resolution in NanoAOD seems to be 0.25
        mw_axis = coffea.hist.Bin(
            "mw", r"$m_{\mathrm{W}}$ (GeV)", 160, 40, 120)
        mw = coffea.hist.Hist("Counts", mw_axis)
        mt_axis = coffea.hist.Bin(
            "mt", r"$m_{\mathrm{t}}$ (GeV)", 180, 160, 190)
        mt = coffea.hist.Hist("Counts", mt_axis)
        alphal_axis = coffea.hist.Bin("alpha", r"$\alpha$ (rad)", 20, 0, 0.02)
        alphal = coffea.hist.Hist("Counts", alphal_axis)
        alphaj_axis = coffea.hist.Bin("alpha", r"$\alpha$ (rad)", 100, 0, 0.2)
        alphaj = coffea.hist.Hist("Counts", alphaj_axis)
        energyfl_axis = coffea.hist.Bin(
            "energyf", r"$E_{\mathrm{gen}} / E_{\mathrm{reco}}", 200, 0.5, 1.5)
        energyfl = coffea.hist.Hist("Counts", energyfl_axis)
        energyfj_axis = coffea.hist.Bin(
            "energyf", r"$E_{\mathrm{gen}} / E_{\mathrm{reco}}", 200, 0, 3)
        energyfj = coffea.hist.Hist("Counts", energyfj_axis)
        return coffea.processor.dict_accumulator(
            {"mlb": mlb, "mw": mw, "mt": mt, "alphal": alphal,
             "alphaj": alphaj, "energyfl": energyfl, "energyfj": energyfj})

    def setup_outputfiller(self, data, dsname):
        return pepper.DummyOutputFiller(self.accumulator.identity())

    def setup_selection(self, data, dsname, is_mc, filler):
        return pepper.Selector(data, data["genWeight"])

    def process_selection(self, selector, dsname, is_mc, filler):
        selector.set_multiple_columns(self.build_gen_columns)
        selector.add_cut("Has gen particles", self.has_gen_particles)

        self.fill_before_selection(
            selector.data, selector.systematics, filler.output)

        super().process_selection(selector, dsname, is_mc, filler)

        self.fill_after_selection(
            selector.final, selector.final_systematics, filler.output)

    @staticmethod
    def sortby(data, field):
        sorting = ak.argsort(data[field], ascending=False)
        return data[sorting]

    def build_gen_columns(self, data):
        part = data["GenPart"]
        mass = np.asarray(ak.flatten(part.mass))
        # mass is only stored for masses greater than 10 GeV
        # this is a fudge to set the b mass to the right
        # value if not stored
        mass[(mass == 0) & (abs(ak.flatten(part["pdgId"])) == 5)] = 4.18
        mass[(mass == 0) & (abs(ak.flatten(part["pdgId"])) == 13)] = 0.102
        part["mass"] = ak.unflatten(mass, ak.num(part))
        # Fill none in order to not enter masked arrays regime
        part["motherid"] = ak.fill_none(part.parent.pdgId, 0)
        part = part[part["motherid"] != 0]
        part = part[part.hasFlags("isFirstCopy", "isHardProcess")]
        abspdg = abs(part.pdgId)
        sgn = np.sign(part.pdgId)

        cols = {}
        cols["genlepton"] = part[((abspdg == 11) | (abspdg == 13))
                                 & (part.motherid == sgn * -24)]
        cols["genlepton"] = self.sortby(cols["genlepton"], "pdgId")

        cols["genb"] = part[(abspdg == 5) & (part.motherid == sgn * 6)]
        cols["genb"] = self.sortby(cols["genb"], "pdgId")

        cols["genw"] = part[(abspdg == 24) & (part.motherid == sgn * 6)]
        cols["genw"] = self.sortby(cols["genw"], "pdgId")

        cols["gent"] = part[(abspdg == 6)]
        cols["gent"] = self.sortby(cols["gent"], "pdgId")

        return cols

    def has_gen_particles(self, data):
        return ((ak.num(data["genlepton"]) == 2)
                & (ak.num(data["genb"]) == 2)
                & (ak.num(data["genw"]) == 2)
                & (ak.num(data["gent"]) == 2))

    def fill_before_selection(self, data, sys, output):
        lep = data["genlepton"][:, 0]
        antilep = data["genlepton"][:, 1]
        b = data["genb"][:, 0]
        antib = data["genb"][:, 1]
        w = data["genw"]
        t = data["gent"]
        weight = np.asarray(sys["weight"])

        mlbarb = (antilep + b).mass
        mlbbar = (lep + antib).mass
        output["mlb"].fill(mlb=mlbarb, weight=weight)
        output["mlb"].fill(mlb=mlbbar, weight=weight)
        output["mw"].fill(mw=ak.flatten(w.mass), weight=np.repeat(weight, 2))
        output["mt"].fill(mt=ak.flatten(t.mass), weight=np.repeat(weight, 2))

    def match_leptons(self, data):
        recolep = self.sortby(data["Lepton"][:, :2], "pdgId")
        genlep = data["genlepton"]
        is_same_flavor = recolep.pdgId == genlep.pdgId
        is_close = recolep.delta_r(genlep) < 0.3
        is_matched = is_same_flavor & is_close

        return genlep[is_matched], recolep[is_matched]

    def match_jets(self, data):
        recojet = ak.with_name(data["Jet"][["pt", "eta", "phi", "mass"]],
                               "PtEtaPhiMLorentzVector")
        genb = ak.with_name(data["genb"][["pt", "eta", "phi", "mass"]],
                            "PtEtaPhiMLorentzVector")
        genbc, recojetc = ak.unzip(ak.cartesian([genb, recojet], nested=True))
        is_close = genbc.delta_r(recojetc) < 0.3
        is_matched = is_close & (ak.sum(is_close, axis=2) == 1)

        mrecojet = [recojet[is_matched[:, i]] for i in range(2)]
        mrecojet = ak.concatenate(mrecojet, axis=1)
        return genb[ak.any(is_matched, axis=2)], mrecojet

    @staticmethod
    def fill_alpha_energyf(gen, reco, weight, alphahist, energyfhist):
        deltaphi = gen.delta_phi(reco)
        energyf = gen.energy / reco.energy
        # axis=None to remove eventual masking
        rep = ak.fill_none(ak.num(deltaphi[~ak.is_none(deltaphi)]), 0)
        alphahist.fill(alpha=ak.flatten(deltaphi, axis=None),
                       weight=np.repeat(weight, rep))
        rep = ak.fill_none(ak.num(energyf[~ak.is_none(energyf)]), 0)
        energyfhist.fill(energyf=ak.flatten(energyf, axis=None),
                         weight=np.repeat(weight, rep))

    def fill_after_selection(self, data, sys, output):
        weight = np.asarray(ak.flatten(sys["weight"], axis=None))

        genlep, recolep = self.match_leptons(data)
        self.fill_alpha_energyf(
            genlep, recolep, weight, output["alphal"], output["energyfl"])

        genjet, recojet = self.match_jets(data)
        self.fill_alpha_energyf(
            genjet, recojet, weight, output["alphaj"], output["energyfj"])

    def save_output(self, output, dest):
        with uproot.recreate(os.path.join(dest, "kinreco.root")) as f:
            items = ("mlb", "mw", "mt", "alphal", "energyfl", "alphaj",
                     "energyfj")
            for key in items:
                f[key] = coffeahist2hist(output[key])


if __name__ == "__main__":
    from pepper import runproc
    runproc.run_processor(Processor, mconly=True)
