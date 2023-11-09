# -*- coding: utf-8 -*-
"""
@author: mustafa hammood, 2023
"""
from SiEPIC.utils import get_layout_variables
#from layout_bragg import layout_bragg
import numpy as np
TECHNOLOGY, lv, ly, cell = get_layout_variables()

layout = layout_bragg(ly, TECHNOLOGY)
layout.cell_name = "bragg_c_te_uni"
layout.cell = cell
layout.io = "GC_TE_1550_8degOxide_BB"
layout.io_lib = "EBeam"
layout.bragg = "ebeam_bragg_apodized"
layout.bragg_lib = "EBeam_Beta"
layout.ybranch = "ebeam_y_1550"
layout.ybranch_lib = "EBeam"
layout.num_sweep = 20
layout.wavl = 1550
layout.pol = "TE"
layout.label = "device_bragg_strip"
layout.wg_type = "Strip TE 1550 nm, w=500 nm"
layout.label = "device_bragg_dWsweep"
layout.layer_floorplan = 'FloorPlan'

layout.number_of_periods = 300
layout.grating_period = 0.320  # µm
layout.wg_width = 0.5  # µm
layout.corrugation_width = np.linspace(0.0, 0.12, layout.num_sweep)  # µm
layout.sinusoidal = False  # sinusoidal corrugations option
layout.a = 2.8  # gaussian apodization index

layout.make()
layout.add_to_layout(cell)
