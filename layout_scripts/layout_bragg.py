# -*- coding: utf-8 -*-
"""
@author: mustafa hammood, 2023
"""
import numpy as np
import pya
from SiEPIC.extend import to_itype
from SiEPIC.scripts import connect_pins_with_waveguide, connect_cell


class layout_bragg:
    def __init__(self, ly, tech):
        # layout parameters
        self.ly = ly
        self.tech = tech
        self.cell = None
        self.cell_name = 'Top_bragg'
        self.io = "drp_cl_gc_8deg_te_se"
        self.io_lib = "DRPSUNY_Standard_whitebox"
        self.bragg = "waveguide_bragg"
        self.bragg_lib = "MH_Personal"
        self.ybranch = "drp_cl_y_splitter_tetm_se"
        self.ybranch_lib = "DRPSUNY_Standard_whitebox"
        self.num_sweep = 10
        self.wavl = 1550
        self.pol = "TE"
        self.label = "device_bragg"
        self.layer_floorplan = 'FP (82/0@1)'
        self.layer_text = 'Text'

        # pcell parameters
        self.number_of_periods = 100
        self.grating_period = 0.320  # µm
        self.wg_width = 0.5  # µm
        self.corrugation_width = 0.02  # µm
        self.sinusoidal = False
        self.a = 2.8

        # waveguide routing and placement parameters
        self.wg_type = 'DRP Si Strip TE 1550 nm, w=500'
        self.wg_radius = 5  # µm
        self.io_pitch = 127  # µm
        self.wg_spacing = 2.5  # µm
        self.wg_turnleft = 10  # additional left turn bias µm
        self.wg_turnright = 10  # additional right turn bias µm

    def make_dependent_params(self):
        """Define "dependent" waveguide routing variables."""
        self.route_up = self.io_pitch  # µm
        self.device_io_space = self.wg_radius*5  # µm
        self.io_column_space = self.wg_radius*14  # µm
        self.device_column_space = self.wg_radius+2  # µm

    def make_bragg(self, number_of_periods, grating_period, wg_width, corrugation_width, sinusoidal, a):
        """Create the device PCell."""
        pcell_params = {"number_of_periods": number_of_periods, "grating_period": grating_period,
                        "wg_width": wg_width, "corrugation_width": corrugation_width, "sinusoidal": sinusoidal, "index": a}
        return self.ly.create_cell(self.bragg, self.bragg_lib, pcell_params).cell_index()

    def make_params(self):
        """Convert the pcell params to list of size self.num_sweep."""
        if type(self.number_of_periods) not in [list, np.ndarray]:
            self.number_of_periods = [self.number_of_periods]*self.num_sweep
        if type(self.grating_period) not in [list, np.ndarray]:
            self.grating_period = [self.grating_period]*self.num_sweep
        if type(self.wg_width) not in [list, np.ndarray]:
            self.wg_width = [self.wg_width]*self.num_sweep
        if type(self.corrugation_width) not in [list, np.ndarray]:
            self.corrugation_width = [self.corrugation_width]*self.num_sweep
        if type(self.sinusoidal) not in [list, np.ndarray]:
            self.sinusoidal = [self.sinusoidal]*self.num_sweep
        if type(self.a) not in [list, np.ndarray]:
            self.a = [self.a]*self.num_sweep

    def make(self):
        """Make the layout cell."""
        dbu = self.ly.dbu
        top_cell = self.cell
        cell = self.cell.layout().create_cell(self.cell_name)
        cell_io = self.ly.create_cell(self.io, self.io_lib).cell_index()
        cell_ybranch = self.ly.create_cell(self.ybranch, self.ybranch_lib)
        width_ybranch = to_itype(cell_ybranch.bbox().width(), 1/dbu)
        self.make_dependent_params()
        self.make_params()  # convert parameters to lists in case they weren't already

        insts_bragg = []
        insts_ybranch = []
        for i in range(self.num_sweep):
            insts_io = []
            cell_bragg = self.make_bragg(
                self.number_of_periods[i], self.grating_period[i], self.wg_width[i], self.corrugation_width[i], self.sinusoidal[i], self.a[i])
            device_length = self.number_of_periods[i]*self.grating_period[i]
            x = -self.io_pitch -self.io_pitch/4 + width_ybranch  # starting coordinate for Ybranch + bragg
            y = i*self.device_column_space + self.device_io_space
            t = pya.Trans(pya.Trans.R0, to_itype(x, dbu), to_itype(y, dbu))
            insts_bragg.append(cell.insert(pya.CellInstArray(cell_bragg, t)))
            insts_ybranch.append(connect_cell(insts_bragg[-1], 'pin1', cell_ybranch, 'opt1'))

            # measurement labels
            device_label = f"opt_in_{self.pol}_{self.wavl}_{self.label}"
            device_attributes = f"_{self.number_of_periods[i]}N{int(self.grating_period[i]*1e3)}nmPeriod{int(self.wg_width[i]*1e3)}nmW{int(self.corrugation_width[i]*1e3)}nmdW{self.a[i]}Apo{int(self.sinusoidal[i])}"
            text = device_label + device_attributes
            text_size = 1.5/dbu
            # direct routing devices
            if i % 2 == 0:
                # instantiate IOs
                x = -self.io_pitch -self.io_pitch/4
                y = -i*self.io_column_space/2
                t = pya.Trans(pya.Trans.R90, to_itype(x, dbu), to_itype(y, dbu))
                insts_io.append(cell.insert(pya.CellInstArray(cell_io, t)))
                t = pya.Trans(pya.Trans.R90, to_itype(
                    x + self.io_pitch, dbu), to_itype(y, dbu))
                insts_io.append(cell.insert(pya.CellInstArray(cell_io, t)))
                t = pya.Trans(pya.Trans.R90, to_itype(
                    x + 2*self.io_pitch, dbu), to_itype(y, dbu))
                insts_io.append(cell.insert(pya.CellInstArray(cell_io, t)))

                # measurement text label on IO
                t = pya.Trans(pya.Trans.R0, to_itype(-self.io_pitch/4, dbu), to_itype(y, dbu))
                io_text = pya.Text(text.replace('.', 'p'), t)
                TextLayerN = cell.layout().layer(self.tech[self.layer_text])
                shape = cell.shapes(TextLayerN).insert(io_text)
                shape.text_size = text_size


                # connect IOs to devices
                # middle IO
                pt2_1 = self.wg_radius + self.wg_spacing
                pt2_2 = self.io_pitch + self.wg_radius + self.wg_turnleft + i*self.wg_spacing*2
                turtle1 = [pt2_1, 90, pt2_2, 90]  # inflection points
                wg_io_device2 = connect_pins_with_waveguide(
                    insts_io[1], 'opt1', insts_ybranch[-1], 'opt2', waveguide_type=self.wg_type, turtle_A=turtle1)  # center io
                # leftmost IO
                pt1_1 = pt2_1 - self.wg_spacing
                pt1_2 = pt2_2 - self.io_pitch + self.wg_spacing
                turtle1 = [pt1_1, 90, pt1_2, 90]  # inflection points
                wg_io_device1 = connect_pins_with_waveguide(
                    insts_io[0], 'opt1', insts_ybranch[-1], 'opt3', waveguide_type=self.wg_type, turtle_A=turtle1)  # top io

                # rightmost IO
                pt3_1 = self.wg_radius + self.wg_spacing
                pt3_2 = self.io_pitch/2 + self.wg_radius + self.wg_turnright + i*self.wg_spacing
                if device_length-np.abs(x) > pt3_2:
                  pt3_2 = device_length-np.abs(x) + self.wg_radius + self.wg_turnright + i*self.wg_spacing
                  print("long device exception handled")
                turtle1 = [pt3_1, -90, pt3_2, -90]  # inflection points
                wg_io_device3 = connect_pins_with_waveguide(
                    insts_io[2], 'opt1', insts_bragg[-1], 'pin2', waveguide_type=self.wg_type, turtle_A=turtle1)  # bottom io

            # backwards routing devices (interdigitated io layout)
            else:
                # instantiate IOs
                x = -self.io_pitch + self.io_pitch/2 -self.io_pitch/4
                y = -(i-1)*self.io_column_space/2
                t = pya.Trans(pya.Trans.R90, to_itype(x, dbu), to_itype(y, dbu))
                insts_io.append(cell.insert(pya.CellInstArray(cell_io, t)))
                t = pya.Trans(pya.Trans.R90, to_itype(
                    x + self.io_pitch, dbu), to_itype(y, dbu))
                insts_io.append(cell.insert(pya.CellInstArray(cell_io, t)))
                t = pya.Trans(pya.Trans.R90, to_itype(
                    x + 2*self.io_pitch, dbu), to_itype(y, dbu))
                insts_io.append(cell.insert(pya.CellInstArray(cell_io, t)))

                # measurement text label on IO
                t = pya.Trans(pya.Trans.R0, to_itype(self.io_pitch/4, dbu), to_itype(y, dbu))
                io_text = pya.Text(text.replace('.', 'p'), t)
                TextLayerN = cell.layout().layer(self.tech[self.layer_text])
                shape = cell.shapes(TextLayerN).insert(io_text)
                shape.text_size = text_size

                # connect IOs to devices
                # middle IO
                pt2_1x = pt2_1
                pt2_2x = self.io_pitch/4
                pt2_3x = self.io_column_space*3/4 + self.wg_spacing
                pt2_4x = pt2_2 + self.io_pitch/4 + 3*self.wg_spacing
                turtle1 = [pt2_1x, 90, pt2_2x, 90, pt2_3x, -90, pt2_4x, -90]  # inflection points
                wg_io_device2 = connect_pins_with_waveguide(
                    insts_io[1], 'opt1', insts_ybranch[-1], 'opt3', waveguide_type=self.wg_type, turtle_A=turtle1)  # top io

                # leftmost IO
                pt1_1x = self.wg_radius
                pt1_2x = pt2_2x
                pt1_3x = pt2_3x - 2*self.wg_spacing
                pt1_4x = pt1_2 + self.io_pitch/4 + self.wg_spacing
                turtle1 = [pt1_1x, 90, pt1_2x, 90, pt1_3x, -90, pt1_4x, -90]  # inflection points
                wg_io_device1 = connect_pins_with_waveguide(
                    insts_io[0], 'opt1', insts_ybranch[-1], 'opt2', waveguide_type=self.wg_type, turtle_A=turtle1)  # top io


                pt3_1x = pt3_1 - self.wg_spacing
                pt3_2x = pt3_2 - self.io_pitch/2 + self.wg_spacing
                turtle1 = [pt3_1x, -90, pt3_2x, -90]  # inflection points
                wg_io_device3 = connect_pins_with_waveguide(
                    insts_io[2], 'opt1', insts_bragg[-1], 'pin2', waveguide_type=self.wg_type, turtle_A=turtle1)  # bottom io


        self.bragg_cell = cell

    def add_to_layout(self, cell):
        t = pya.Trans(pya.Trans.R270, 0, 0)
        x = -pya.CellInstArray(self.bragg_cell.cell_index(), t).bbox(self.ly).p1.x
        y = -pya.CellInstArray(self.bragg_cell.cell_index(), t).bbox(self.ly).p1.y
        t = pya.Trans(pya.Trans.R270, x, y)
        cell.insert(pya.CellInstArray(self.bragg_cell.cell_index(), t))
        FloorplanLayer = self.bragg_cell.layout().layer(self.tech[self.layer_floorplan])
        cell.shapes(FloorplanLayer).insert(
            pya.Box(0, 0, self.bragg_cell.bbox().height(), self.bragg_cell.bbox().width()))
