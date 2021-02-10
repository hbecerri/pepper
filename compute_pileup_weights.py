import os
import pepper
import coffea
import uproot3


class Processor(pepper.Processor):
    def __init__(self, config, eventdir):
        self.data_pu_hist = config["data_pu_hist"]
        self.data_pu_hist_up = config["data_pu_hist_up"]
        self.data_pu_hist_down = config["data_pu_hist_down"]

        with uproot3.open(self.data_pu_hist) as f:
            datahist = pepper.misc.rootimport(f["pileup"])
        with uproot3.open(self.data_pu_hist_up) as f:
            datahistup = pepper.misc.rootimport(f["pileup"])
        with uproot3.open(self.data_pu_hist_down) as f:
            datahistdown = pepper.misc.rootimport(f["pileup"])
        if len(datahist.axes()) != 1:
            raise pepper.config.ConfigError(
                "data_pu_hist has invalid number of axes. Only one axis is "
                "allowed.")
        if not datahist.compatible(datahistup):
            raise pepper.config.ConfigError(
                "data_pu_hist_up does not have the same shape as "
                "data_pu_hist.")
        if not datahist.compatible(datahistdown):
            raise pepper.config.ConfigError(
                "data_pu_hist_down does not have the same shape as "
                "data_pu_hist.")

        config["hists"] = {
            "pileup": pepper.HistDefinition({
                "bins": [
                    {
                        "name": datahist.axes()[0].name,
                        "label": ("True mean number interactions per bunch "
                                  "crossing"),
                        "n_or_arr": datahist.axes()[0].edges()
                    }
                ],
                "fill": {
                    datahist.axes()[0].name: [
                        "Pileup",
                        "nTrueInt"
                    ]
                }
            })
        }
        if "hists_to_do" in config:
            del config["hists_to_do"]
        config["compute_systematics"] = False
        # Treat all datasets as normal datasets, instead of using them as
        # systematic
        config["dataset_for_systematics"] = {}

        super().__init__(config, eventdir)

    def preprocess(self, datasets):
        # Only run over MC
        processed = {}
        for key, value in datasets.items():
            if key in self.config["mc_datasets"]:
                processed[key] = value
        return processed

    def setup_selection(self, data, dsname, is_mc, filler):
        # Ignore generator weights, because pileup is independent
        return pepper.Selector(data, on_update=filler.get_callbacks())

    def process_selection(self, selector, dsname, is_mc, filler):
        pass

    @staticmethod
    def _save_hists(hist, datahist, datahistup, datahistdown, filename):
        with uproot3.recreate(filename) as f:
            for idn in hist.identifiers("dataset"):
                dataset = idn.name
                hist_int = hist.integrate("dataset", dataset)
                hist_int.scale(1 / hist_int.project().values()[()])
                for datahist_i, suffix in [
                        (datahist, ""), (datahistup, "_up"),
                        (datahistdown, "_down")]:
                    ratio = pepper.misc.hist_divide(datahist_i, hist_int)
                    f[dataset + suffix] = pepper.misc.export(ratio)

    def save_output(self, output, dest):
        with uproot3.open(self.data_pu_hist) as f:
            datahist = pepper.misc.rootimport(f["pileup"])
        with uproot3.open(self.data_pu_hist_up) as f:
            datahistup = pepper.misc.rootimport(f["pileup"])
        with uproot3.open(self.data_pu_hist_down) as f:
            datahistdown = pepper.misc.rootimport(f["pileup"])

        # Normalize data histograms
        datahist.scale(1 / datahist.project().values()[()])
        datahistup.scale(1 / datahistup.project().values()[()])
        datahistdown.scale(1 / datahistdown.project().values()[()])

        mchist = output["hists"][("Before cuts", "pileup")]
        # Set underflow and 0 pileup bin to 0, which might be != 0 only for
        # buggy reasons in MC
        for idn in mchist.identifiers("dataset"):
            mchist._sumw[(idn,)][:2] = 0

        new_ds_axis = coffea.hist.Cat("dataset", "Dataset")
        mchist_allds = mchist.group(
            mchist.axis("dataset"), new_ds_axis,
            {"all_datasets": [i.name for i in mchist.identifiers("dataset")]})
        self._save_hists(
            mchist_allds, datahist, datahistup, datahistdown,
            os.path.join(dest, "pileup.root"))
        self._save_hists(
            mchist, datahist, datahistup, datahistdown,
            os.path.join(dest, "pileup_perdataset.root"))


if __name__ == "__main__":
    from pepper import runproc
    runproc.run_processor(
        Processor, "Create histograms needed for pileup reweighting",
        mconly=True)
