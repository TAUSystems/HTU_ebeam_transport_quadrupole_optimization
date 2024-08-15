"""
Quick classes to store an electron beam's 6D phase space
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import median_filter
from PIL import Image

electron_rest_mass = 0.511e6  # eV
lightspeed_mps = 2.9979e8  # m/s

def calculate_geometric_emittance(x, xp):
    dx = x - np.average(x)
    dxp = xp - np.average(xp)
    sigma_x2 = np.average(np.square(dx))
    sigma_xp2 = np.average(np.square(dxp))
    sigma_xxp = np.average(dx * dxp)
    return np.sqrt(sigma_x2 * sigma_xp2 - np.square(sigma_xxp))


class Beam:
    """
    The coordinates are defined as follows:
    x, y, z: [m]
    xp, yp:  [rad]
    pz:      [eV]
    """

    def __init__(self, n_particles, total_charge_pc=1):
        self.n_particles = n_particles
        self.charge = total_charge_pc

        self.x = np.zeros(n_particles)
        self.xp = np.zeros(n_particles)
        self.y = np.zeros(n_particles)
        self.yp = np.zeros(n_particles)
        self.z = np.zeros(n_particles)
        self.pz = np.zeros(n_particles)

        self.x_units = "m"
        self.x_scale = 1
        self.y_units = "m"
        self.y_scale = 1
        self.xp_units = "rad"
        self.xp_scale = 1
        self.yp_units = "rad"
        self.yp_scale = 1

    def autoset_xy_scale(self):
        x_scale = self.find_scale_info(self.get_x())
        y_scale = self.find_scale_info(self.get_y())
        self.x_units = x_scale[0] + "m"
        self.x_scale = x_scale[1]
        self.y_units = y_scale[0] + "m"
        self.y_scale = y_scale[1]

        xp_scale = self.find_scale_info(self.get_xp())
        yp_scale = self.find_scale_info(self.get_yp())
        self.xp_units = xp_scale[0] + "rad"
        self.xp_scale = xp_scale[1]
        self.yp_units = yp_scale[0] + "rad"
        self.yp_scale = yp_scale[1]

    @staticmethod
    def find_scale_info(vals):
        order = np.abs(np.max(vals)) + np.abs(np.min(vals))
        if order < 1e-7:
            units = "n"
            scale = 1e9
        elif order < 1e-4:
            units = "u"
            scale = 1e6
        elif order < 1e-1:
            units = "m"
            scale = 1e3
        else:
            units = ""
            scale = 1
        return [units, scale]

    def set_phase_space(self, x, xp, y, yp, z, pz, autoscale_units=True):
        self.set_x(x)
        self.set_xp(xp)
        self.set_y(y)
        self.set_yp(yp)
        self.set_z(z)
        self.set_pz(pz)

        if autoscale_units:
            self.autoset_xy_scale()

    def calculate_x_emittance_n(self):
        emit_x = calculate_geometric_emittance(self.x, self.xp)
        return emit_x * self.calculate_gamma_lorentz()

    def calculate_y_emittance_n(self):
        emit_y = calculate_geometric_emittance(self.y, self.yp)
        return emit_y * self.calculate_gamma_lorentz()

    def calculate_gamma_lorentz(self):
        electron_rest_mass = 0.511e6  # eV
        beam_gamma = self.pz / electron_rest_mass
        return np.average(beam_gamma)

    def calculate_average_energy_eV(self):
        return np.average(self.pz)

    def calculate_energy_spread(self):
        average_energy = self.calculate_average_energy_eV()
        return np.sqrt(np.average(np.square(self.pz - average_energy))) / average_energy * 100

    def calculate_x_average(self):
        return np.average(self.x)

    def calculate_y_average(self):
        return np.average(self.y)

    def calculate_xp_average(self):
        return np.average(self.xp)

    def calculate_yp_average(self):
        return np.average(self.yp)

    def calculate_sigma_x(self):
        return np.std(self.x)

    def calculate_sigma_y(self):
        return np.std(self.y)

    def calculate_x_beta(self):
        sigma_x2 = np.average(np.square(self.x - self.calculate_x_average()))
        return sigma_x2 / calculate_geometric_emittance(self.x, self.xp)

    def calculate_y_beta(self):
        sigma_y2 = np.average(np.square(self.y - self.calculate_y_average()))
        return sigma_y2 / calculate_geometric_emittance(self.y, self.yp)

    def calculate_x_gamma(self):
        sigma_xp2 = np.average(np.square(self.xp - self.calculate_xp_average()))
        return sigma_xp2 / calculate_geometric_emittance(self.x, self.xp)

    def calculate_y_gamma(self):
        sigma_yp2 = np.average(np.square(self.yp - self.calculate_yp_average()))
        return sigma_yp2 / calculate_geometric_emittance(self.y, self.yp)

    def calculate_x_alpha(self):
        dx = self.x - self.calculate_x_average()
        dxp = self.xp - self.calculate_xp_average()
        sigma_xxp = -np.average(dx * dxp)
        return sigma_xxp / calculate_geometric_emittance(self.x, self.xp)

    def calculate_y_alpha(self):
        dy = self.y - self.calculate_y_average()
        dyp = self.yp - self.calculate_yp_average()
        sigma_yyp = -np.average(dy * dyp)
        return sigma_yyp / calculate_geometric_emittance(self.y, self.yp)

    def set_total_charge_pc(self, total_charge_pc):
        self.charge = total_charge_pc

    def set_x(self, x):
        self.x = np.copy(x)

    def set_xp(self, xp):
        self.xp = np.copy(xp)

    def set_y(self, y):
        self.y = np.copy(y)

    def set_yp(self, yp):
        self.yp = np.copy(yp)

    def set_z(self, z):
        self.z = np.copy(z)

    def set_pz(self, pz):
        self.pz = np.copy(pz)

    def get_n_particles(self):
        return self.n_particles

    def get_total_charge_pc(self):
        return self.charge

    def get_x(self):
        return self.x

    def get_xp(self):
        return self.xp

    def get_y(self):
        return self.y

    def get_yp(self):
        return self.yp

    def get_z(self):
        return self.z

    def get_pz(self):
        return self.pz

    def plot_transverse_phase_space(self, num_bins=50, title=None, do_scatter=False):
        fig, ax = plt.subplots(2, 2, figsize=(9, 7))
        mycmap = plt.get_cmap('viridis')
        mycmap.set_under(color='white')  # map 0 to this color

        if do_scatter:
            ax[0, 0].scatter(self.get_x()*self.x_scale, self.get_xp()*self.xp_scale, s=0.1, c='r')
        else:
            ax[0, 0].hist2d(self.get_x()*self.x_scale, self.get_xp()*self.xp_scale, bins=num_bins, cmap=mycmap)
        ax[0, 0].set_xlabel("x ["+self.x_units+"]")
        ax[0, 0].set_ylabel("xp ["+self.xp_units+"]")

        if do_scatter:
            ax[1, 0].scatter(self.get_x() * self.x_scale, self.get_yp() * self.yp_scale, s=0.1, c='r')
        else:
            ax[1, 0].hist2d(self.get_x()*self.x_scale, self.get_yp()*self.yp_scale, bins=num_bins, cmap=mycmap)
        ax[1, 0].set_xlabel("x ["+self.x_units+"]")
        ax[1, 0].set_ylabel("yp ["+self.yp_units+"]")

        if do_scatter:
            ax[0, 1].scatter(self.get_y() * self.y_scale, self.get_xp() * self.xp_scale, s=0.1, c='r')
        else:
            ax[0, 1].hist2d(self.get_y()*self.y_scale, self.get_xp()*self.xp_scale, bins=num_bins, cmap=mycmap)
        ax[0, 1].set_xlabel("y ["+self.y_units+"]")
        ax[0, 1].set_ylabel("xp ["+self.xp_units+"]")

        if do_scatter:
            ax[1, 1].scatter(self.get_y() * self.y_scale, self.get_yp() * self.yp_scale, s=0.1, c='r')
        else:
            ax[1, 1].hist2d(self.get_y()*self.y_scale, self.get_yp()*self.yp_scale, bins=num_bins, cmap=mycmap)
        ax[1, 1].set_xlabel("y ["+self.y_units+"]")
        ax[1, 1].set_ylabel("yp ["+self.yp_units+"]")

        fig.suptitle(title)
        fig.tight_layout()
        plt.show()

        return

    def plot_transverse_spot_image(self, num_bins=50, title=None, do_scatter=False, save=None, do_plot=True):
        max_extent = 7.68e-3  # /10000
        #num_bins = 50
        x_vals = self.get_x()
        y_vals = self.get_y()

        xedges = np.linspace(-max_extent, max_extent, num_bins+1)*self.x_scale
        yedges = np.linspace(-max_extent, max_extent, num_bins+1)*self.y_scale
        print(xedges[1]-xedges[0])

        screen = np.where((np.abs(x_vals) < max_extent) & (np.abs(y_vals) < max_extent))[0]

        if do_plot:
            plt.figure(figsize=(5, 4))
            mycmap = plt.get_cmap('viridis')
            mycmap.set_under(color='white')  # map 0 to this color

            """
            hist = plt.hist(x_vals*1e3,bins=200, label="Histogram of Beam")
            x = np.array(hist[1])
            vals = np.array(hist[0])
            sigma = np.std(x_vals)*1e3
            x0 = x[np.argmax(vals)]
            amp = np.max(vals)/3
            plt.plot(x, amp*np.exp(-0.5 * ((x-x0)/sigma)**2), label="Gaussian of Sigx")
            plt.title("Equivalent Sigma X")
            plt.xlabel("x (mm)")
            plt.ylabel("Counts (arb)")
            plt.tight_layout()
            plt.legend()
            plt.show()
            """
            if do_scatter:
                plt.scatter(x_vals[screen] * self.x_scale, y_vals[screen] * self.y_scale, c='r', s=0.01)
            else:
                hist = plt.hist2d(x_vals[screen]*self.x_scale, y_vals[screen]*self.y_scale, bins=(xedges, yedges),
                           cmap=mycmap,
                           #norm=colors.LogNorm(),
                           #vmax=100,
                           )
                #plt.colorbar()
                #plt.show()

                """
                array = np.array(hist[0])
                proj = np.sum(array, axis=0)
                axis = np.linspace(-max_extent, max_extent, len(xedges)-1)
    
                print(len(proj), len(axis))
                axis_average = np.average(axis, weights=proj)
                rms = np.sqrt(np.average((axis-axis_average) ** 2, weights=proj))
                print("rms", rms)
                process = np.log(array+1)-2
                process[np.where(process<0)]=0
                process = process.T
                plt.imshow(process)
                plt.show()
                """

            plt.xlim([-max_extent*self.x_scale, max_extent*self.x_scale])
            plt.ylim([-max_extent*self.y_scale, max_extent*self.y_scale])
            plt.xlabel("x ["+self.x_units+"]")
            plt.ylabel("y ["+self.y_units+"]")
            plt.title(title)
            plt.show()
        else:
            hist = plt.hist2d(x_vals[screen] * self.x_scale, y_vals[screen] * self.y_scale, bins=(xedges, yedges))

        if not do_scatter:
            image = hist[0].T/1000*65535
            image_sub = np.copy(image) - 1.0
            image_sub[np.where(image_sub < 0)] = 0
            image_filter = median_filter(image, size=3)
            image_filter_sub = np.copy(image_filter) - 1.0
            image_filter_sub[np.where(image_filter_sub < 0)] = 0
            if do_plot:
                plt.imshow(image)
                plt.show()

            if save is not None:
                def save_hist_image(filename, numpy_hist_array):
                    uint16_array = Image.fromarray(np.uint16(numpy_hist_array))
                    uint16_array.save(filename)

                save_hist_image(f"./save_images/simscreen_set{save}.png", image)
                #save_hist_image(f"./save_images/simscreen_sub_set{save}.png", image_sub)
                save_hist_image(f"./save_images/simscreen_filter_set{save}.png", image_filter)
                #save_hist_image(f"./save_images/simscreen_filter_sub_set{save}.png", image_filter_sub)

        return

    def plot_longitudinal_phase_space(self, num_bins=50, title=None):
        plt.figure(figsize=(5, 4))
        mycmap = plt.get_cmap('viridis')
        mycmap.set_under(color='white')  # map 0 to this color

        plt.hist(self.get_z()*1e6, bins=1000)
        plt.xlabel("z [um]")
        plt.ylabel("Num Particles")
        plt.yscale('log')
        plt.show()

        #plt.hist2d(self.get_z()*1e6, self.get_pz()/1e6, bins=num_bins, cmap=mycmap)
        plt.scatter(self.get_z()*1e6-np.average(self.get_z()), self.get_pz()/1e6)
        plt.xlabel("z [um]")
        plt.ylabel("pz [MeV]")
        plt.title(title)
        plt.show()

        return

    def plot_dispersion_phase_space(self, num_bins=50, title=None):
        scale = 1e3
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        mycmap = plt.get_cmap('viridis')
        mycmap.set_under(color='white')  # map 0 to this color

        ax[0].hist2d(self.get_pz()/1e6, self.get_x() * scale, bins=num_bins, cmap=mycmap)
        ax[0].set_ylabel("x [mm]")
        ax[0].set_xlabel("pz [MeV]")

        ax[1].hist2d(self.get_pz()/1e6, self.get_y() * scale, bins=num_bins, cmap=mycmap)
        ax[1].set_ylabel("y [mm]")
        ax[1].set_xlabel("pz [MeV]")

        plt.title(title)
        plt.show()

        return

    def plot_longitudinal_slice_emittance(self, bin_size_nm=400.0):
        # Initialize arrays
        beam_z = self.get_z()
        min_z = np.min(beam_z)
        max_z = np.max(beam_z)

        slice_bins = np.arange(start=min_z, stop=max_z, step=bin_size_nm*1e-9)
        number_bins = len(slice_bins)-1
        emit_x_arr = np.zeros(number_bins)
        emit_y_arr = np.zeros(number_bins)
        central_val = np.zeros(number_bins)
        num_parts = np.zeros(number_bins)
        ave_energy_arr = np.zeros(number_bins)
        rms_energy_arr = np.zeros(number_bins)

        # Separate population into bins

        beam_x = self.get_x()
        beam_xp = self.get_xp()
        beam_y = self.get_y()
        beam_yp = self.get_yp()
        beam_pz = self.get_pz()

        threshold = 0.005*len(beam_z)
        print(f"-Threshold is {int(threshold)} particles")

        for i in range(number_bins):
            central_val[i] = slice_bins[i]/2 + slice_bins[i+1]/2
            bin_sel = np.where((beam_z >= slice_bins[i]) & (beam_z <= slice_bins[i+1]))[0]
            num_parts[i] = len(bin_sel)

            # Ignore bins that have too few electrons to perform good statistics
            if num_parts[i] > threshold:
                # Calculate emittance for each bin
                ave_energy_arr[i] = np.average(beam_pz[bin_sel])
                bin_gamma = ave_energy_arr[i] / electron_rest_mass
                rms_energy_arr[i] = np.sqrt(np.average(np.square(beam_pz[bin_sel] - ave_energy_arr[i]))) / ave_energy_arr[i] * 100

                emit_x_bin = calculate_geometric_emittance(beam_x[bin_sel], beam_xp[bin_sel])
                emit_x_arr[i] = emit_x_bin * bin_gamma
                emit_y_bin = calculate_geometric_emittance(beam_y[bin_sel], beam_yp[bin_sel])
                emit_y_arr[i] = emit_y_bin * bin_gamma

        # Plot and display results
        nonzero = np.where(emit_x_arr > 0)[0]

        total_charge = self.get_total_charge_pc() * 1e-12
        current = total_charge * (num_parts/len(beam_z)) * lightspeed_mps / (bin_size_nm*1e-9)

        fig, ax = plt.subplots(2,1, sharex=True)

        ax1 = ax[0]
        ax1.plot(central_val[nonzero]*1e6, emit_x_arr[nonzero]*1e6, label="X Slice Emittance", c='r')
        ax1.plot(central_val[nonzero]*1e6, emit_y_arr[nonzero]*1e6, label="Y Slice Emittance", c='b')
        ax1.set_ylabel("Slice Emittance (um-rad)")
        ax1.tick_params(labelbottom=False)
        fig.suptitle(f"Longitudinal e-Beam Slice Statistics: {bin_size_nm} nm")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Slice Current (kA)")
        ax2.plot(central_val[nonzero]*1e6, current[nonzero]*1e-3, c='g', ls='--', label="Current")
        ax2.set_ylim([0,np.max(current)*1.20*1e-3])
        ax2.tick_params(labelbottom=False)

        ax1.legend(loc=2)
        ax2.legend(loc=1)

        ax3 = ax[1]
        ax3.plot(central_val[nonzero]*1e6, ave_energy_arr[nonzero]*1e-6, c='k', label='Ave.')
        ax3.set_ylabel("Slice Energy (MeV)")
        ax3.set_xlabel("Longitudinal Slice (um)")

        ax4 = ax3.twinx()
        ax4.plot(central_val[nonzero]*1e6, rms_energy_arr[nonzero], c='k', ls='dotted', label="RMS")
        ax4.set_ylabel("Energy Spread (%)")

        ax3.legend(loc=2)
        ax4.legend(loc=1)

        fig.tight_layout()

        plt.subplots_adjust(hspace=0)
        plt.show()

        return

    def plot_energy_slice_emittance(self, bin_size_mev=0.5):
        # Initialize arrays
        beam_pz = self.get_pz()
        min_pz = np.min(beam_pz)
        max_pz = np.max(beam_pz)

        energy_bins = np.arange(start=min_pz, stop=max_pz, step=bin_size_mev*1e6)
        number_bins = len(energy_bins)-1
        emit_x_arr = np.zeros(number_bins)
        emit_y_arr = np.zeros(number_bins)
        central_ev = np.zeros(number_bins)
        num_parts = np.zeros(number_bins)

        # Separate population into bins

        beam_x = self.get_x()
        beam_xp = self.get_xp()
        beam_y = self.get_y()
        beam_yp = self.get_yp()
        threshold = 0.01*len(beam_pz)
        print(f"-Threshold is {int(threshold)} particles")

        for i in range(number_bins):
            central_ev[i] = energy_bins[i]/2 + energy_bins[i+1]/2
            bin_sel = np.where((beam_pz >= energy_bins[i]) & (beam_pz <= energy_bins[i+1]))[0]
            num_parts[i] = len(bin_sel)

            # Ignore bins that have too few electrons to perform good statistics
            if num_parts[i] > threshold:
                # Calculate emittance for each bin
                bin_gamma = central_ev[i] / electron_rest_mass

                emit_x_bin = calculate_geometric_emittance(beam_x[bin_sel], beam_xp[bin_sel])
                emit_x_arr[i] = emit_x_bin * bin_gamma
                emit_y_bin = calculate_geometric_emittance(beam_y[bin_sel], beam_yp[bin_sel])
                emit_y_arr[i] = emit_y_bin * bin_gamma

        # Plot and display results
        nonzero = np.where(emit_x_arr > 0)[0]

        fig, ax1 = plt.subplots()
        ax1.plot(central_ev[nonzero]/1e6, emit_x_arr[nonzero]*1e6, label="X Slice Emittance", c='r')
        ax1.plot(central_ev[nonzero]/1e6, emit_y_arr[nonzero]*1e6, label="Y Slice Emittance", c='b')
        ax1.set_xlabel("Energy (MeV)")
        ax1.set_ylabel("Energy Slice Emittance (um-rad)")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Number of Particles")
        ax2.plot(central_ev[nonzero]/1e6, num_parts[nonzero], c='g', ls='--', label="Num. Parts.")
        ax2.set_ylim([0,np.max(num_parts)*1.20])

        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()

        return

    def print_statistics(self):
        print("Num Particles: ", self.n_particles)
        print("x_ave,   y_ave:  ", self.calculate_x_average(), self.calculate_y_average())
        print("px_ave,  py_ave: ", self.calculate_xp_average(), self.calculate_yp_average())
        print("sig_x,   sig_y:  ", self.calculate_sigma_x(), self.calculate_sigma_y())
        print("beta_x,  beta_y: ", self.calculate_x_beta(), self.calculate_y_beta())
        print("alpha_x, alpha_y:", self.calculate_x_alpha(), self.calculate_y_alpha())
        print("gamma_x, gamma_y:", self.calculate_x_gamma(), self.calculate_y_gamma())
        print("emitn_x, emitn_y:", self.calculate_x_emittance_n(), self.calculate_y_emittance_n())
        print("gamma_L, Energy, E_spread(%): ", self.calculate_gamma_lorentz(),
              self.calculate_average_energy_eV() / 1e6, self.calculate_energy_spread())
        return

class NumpyBeam(Beam):

    def __init__(self, filename, adjust_to_average_energy_MeV=None, is_alternate_ordering=False):
        """
        Initialize a beam using a 6D phase space saved in a .npy file
        :param filename: Filename to .npy file with 6D beam distribution, by default we are assuming that the file is
                         saved as x,xp,y,yp,z,pz with pz saved as eV, but some older version are saved otherwise
        :param adjust_to_average_energy_MeV: Optional, legacy parameter to convert from fraction offset to eV
        :param is_alternate_ordering: Optional, legacy flag for older files saved as x,y,z,xp,yp,pz
        """
        beam_data = np.load(filename)
        super().__init__(np.shape(beam_data)[1])

        if adjust_to_average_energy_MeV is None:
            energy_eV = beam_data[5]
        else:
            energy_eV = (adjust_to_average_energy_MeV * (beam_data[5] + 1)) * 1e6
        if is_alternate_ordering:
            self.set_phase_space(beam_data[0], beam_data[3], beam_data[1], beam_data[4], beam_data[2], energy_eV)
        else:
            self.set_phase_space(beam_data[0], beam_data[1], beam_data[2], beam_data[3], beam_data[4], energy_eV)


class SDDSBeam(Beam):

    def __init__(self, beam_data):
        """
        Initialize a beam using a 6D phase space read from a .proc.plainbin file using utils function
        :param beam_data: the output of the utils function
        """
        super().__init__(np.shape(beam_data)[0])

        gamma = beam_data[:, 5]
        electron_rest_mass = 0.511e6  # eV
        energy_eV = gamma * electron_rest_mass

        beam_t = beam_data[:, 4]
        beam_z = (beam_t - np.average(beam_t)) * 2.998e8

        self.set_phase_space(x=beam_data[:, 0], xp=beam_data[:, 1],
                             y=beam_data[:, 2], yp=beam_data[:, 3],
                             z=beam_z, pz=energy_eV)
