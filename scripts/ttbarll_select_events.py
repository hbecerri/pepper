#!/usr/bin/env python3

import pepper

if __name__ == "__main__":
    from pepper import runproc
    runproc.run_processor(
        pepper.ProcessorTTbarLL,
        "Select events from nanoAODs using the TTbarLL processor."
        "This will save cutflows, histograms and, if wished, per-event data. "
        "Histograms are saved in a Coffea format and are ready to be plotted "
        "by for example plot_control.py")
