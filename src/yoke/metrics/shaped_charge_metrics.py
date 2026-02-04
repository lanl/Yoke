"""Metric functions related to perturbed layer interface data."""

import numpy as np
import scipy as sp


# function for reading the npz files
def singlePVIarray(
    npzfile: str = "./lsc_nonconvex_pvi_idx00115.npz", FIELD: str = "rho"
) -> np.ndarray:
    """Function to grab single array from NPZ.

    Args:
       npzfile (str): File name for NPZ.
       FIELD (str): Field to return array for.

    Returns:
       field (np.array): Array of hydro-dynamic field for plotting

    """
    NPZ = np.load(npzfile)
    arrays_dict = dict()
    for key in NPZ.keys():
        arrays_dict[key] = NPZ[key]

    NPZ.close()

    return arrays_dict[FIELD]


class SCmetrics:
    """LSC-NPZ initialization class.

    Primarily called by SCmetrics(filename), where filename is the name of the npz file.
    Variable liner is for the name of the shaped-charge liner in the Pagosa simulations
    (more specifically, as named in the npz files).
    """

    def __init__(self, filename: str, liner: str = "throw") -> None:
        """Initialize SCmetrics object."""
        self.filename = filename

        # initialize density and vf of liner
        self.density = self.get_field("density_" + liner)

        # initialize volume fraction for liner and W-velocity
        self.vofm = self.get_field("vofm_" + liner)
        self.Wvelocity = self.get_field("Wvelocity")

        # get mesh coordinates
        self.Rcoord = singlePVIarray(npzfile=filename, FIELD="Rcoord")
        self.Zcoord = singlePVIarray(npzfile=filename, FIELD="Zcoord")
        # extend coordinate vectors to contain end points of mesh
        self.Rcoord = np.append(self.Rcoord, 2 * self.Rcoord[-1] - self.Rcoord[-2])
        self.Zcoord = np.append(self.Zcoord, 2 * self.Zcoord[-1] - self.Zcoord[-2])

        # initialize other fields such as volume,
        # volume is the cell volume for 2D cylindrical meshes
        self.volume = self.compute_volume()

        # create regions field using density of liner
        # this returns density field of connected components
        # that lie on the vertical axis (axis of symmetry)
        self.regions = self.compute_regions(mask=True)

        # initialize some vars to None and only compute them once
        self.jet_mass = None
        self.jet_kinetic_energy = None
        self.HE_mass = None

        # hard-coding names of some materials
        self.HE_field_name = "density_maincharge"
        self.HE_vofm_field_name = "vofm_maincharge"

    # function for getting a field from a npz file
    # function checks for nans, as they can occur, for example,
    # in the density fields
    def get_field(self, field_name: str) -> np.ndarray:
        """Return single hydrofield."""
        field = singlePVIarray(npzfile=self.filename, FIELD=field_name)
        field_map = np.zeros(field.shape)
        Dind = np.where(np.isfinite(field))
        field_map[Dind] = field[Dind]

        return field_map

    # function to compute and return HE mass
    def get_HE_mass(self) -> float:
        """Return the mass of the high explosive (HE) material."""
        if self.HE_mass is None:
            HEdensity = self.get_field(self.HE_field_name)
            HEvofm = self.get_field(self.HE_vofm_field_name)
            return np.sum(self.volume * HEdensity * HEvofm)
        else:
            return self.HE_mass

    def get_jet_width_stats(self, vel_thres: float = 0.0) -> tuple[float, float, float]:
        """Function to compute jet width statistics.

        Function returns avg width, std dev of width, and max width.

        Variable vel_thres sets the velocity threshold and the
        "jet" is only considered when its velocity exceeds the
        threshold.
        """
        if vel_thres > 0.0:
            # get jet locations above threshold
            Vind = np.where((self.regions) & (self.Wvelocity > vel_thres))
            skeleton = np.zeros(self.regions.shape)
            skeleton[Vind] = self.regions[Vind]
        else:
            skeleton = self.regions

        # get jet width as a function of z (vertical axis)
        Rcoord_map = np.repeat(
            np.reshape(self.Rcoord[1:], (1, -1)), skeleton.shape[0], axis=0
        )
        Rcoord_mask = skeleton * Rcoord_map
        width = np.max(Rcoord_mask, axis=0)

        # compute stats
        # multiplying by 2 to consider a "true" width instead of a "radius",
        # since we're looking at 2D cylindrical simulations
        avg_width = 2.0 * np.mean(width)
        std_width = 2.0 * np.std(width)
        max_width = 2.0 * np.max(width)

        return avg_width, std_width, max_width

    def get_jet_rho_velsq_2D(self, vel_thres: float = 0.1) -> float:
        """Function to compute cumulative value of jet density times.

        Computes velocity squared over a 2D cross section of jet. This is primarly
        intended for 2D axi-symmetric calculations.

        Variable vel_thres allows for parts of jet below the threshold to be ignored.
        """
        Vind = np.where((self.Wvelocity >= vel_thres) & (self.regions))
        return np.sum(self.density[Vind] * np.square(self.Wvelocity[Vind]))

    def get_jet_sqrt_rho_vel_2D(self, vel_thres: float = 0.1) -> float:
        """Function to compute cumulative value of jet sqrt(density).

        Computes velocity squared over a 2D cross section of jet.
        This is primarly intended for 2D axi-symmetric calculations.

        Variable vel_thres allows for parts of jet below the
        threshold to be ignored.
        """
        Vind = np.where((self.Wvelocity > vel_thres) & (self.regions))
        return np.sum(np.sqrt(self.density[Vind]) * self.Wvelocity[Vind])

    def get_jet_kinetic_energy(self, vel_thres: float = 0.1) -> float:
        """Function to compute jet kinetic energy of effective jet.

        This differs from function get_jet_rho_velsq_2D in that
        it computes the kinetic energy of the actual 3D jet object.
        """
        eff_jet_mass_map = self.get_eff_jet_mass_map(vel_thres=vel_thres)
        return 0.5 * np.sum(eff_jet_mass_map * np.square(self.Wvelocity))

    def get_jet_sqrt_kinetic_energy(self, vel_thres: float = 0.1) -> float:
        """Function to compute spatially integrated quantity.

        Computes sqrt(0.5 * mass) * velocity.

        This differs from function get_jet_sqrt_rho_velsq_2D in that
        it computes the quantity for the actual 3D jet object.
        """
        eff_jet_mass_map = np.sqrt(self.get_eff_jet_mass_map(vel_thres=vel_thres))
        return 0.70710678 * np.sum(np.sqrt(eff_jet_mass_map) * self.Wvelocity)

    def compute_volume(self) -> float:
        """Function to compute volume for 2D axis-symmetric grid cells."""
        surf_area = np.pi * (np.square(self.Rcoord[1:]) - np.square(self.Rcoord[0:-1]))
        height = self.Zcoord[1:] - self.Zcoord[0:-1]
        volume = np.matmul(
            np.reshape(height, (len(height), 1)),
            np.reshape(surf_area, (1, len(surf_area))),
        )
        return volume

    def get_jet_mass(self) -> float:
        """Compute and return jet mass."""
        if self.jet_mass is None:
            return np.sum(self.volume * self.density * self.vofm)
        else:
            return self.jet_mass

    def compute_regions(self, mask: bool = False) -> np.ndarray:
        """Returns the connected components.

        Connected components that touch the
        central axis (axis of symmetry for 2D runs).
        Initial field is taken as the density-liner field.

        If mask is True, then only return zero/one with
        one representing an on-axis jet component.
        Otherwise, each different connected component will be
        labeled with an "ID" (just a number) for the component.
        """
        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        field_regions, n_regions = sp.ndimage.label(self.density, structure)

        # get region labels for regions that are on-axis
        axis_regions = np.unique(field_regions[:, 0])

        # removing connected components/regions that are not on-axis
        count = 1  # label for first connected component, needs to
        # be greater than zero.
        # A value of zero represents locations that are
        # not part of any connected component.
        for ilabel in range(1, np.max(field_regions) + 1, 1):
            Aind = np.where(field_regions == ilabel)
            if ilabel not in axis_regions:  # region is not on-axis
                field_regions[Aind] = 0
            else:  # region is on-axis
                field_regions[Aind] = count
                if not mask:  # increment label for next region
                    count = count + 1

        return field_regions

    def max_regions(self, field: str) -> float:
        """Function to compute maximum field value over contiguous jet.

        To be included in the contiguous jet, a location must
        be in a connected component that is touching the "central"
        axis (which is the 2D axis of symmetry).
        """
        Vind = np.where(self.regions > 0)  # get on-axis jet regions
        maxv = np.max(field[Vind])
        return maxv

    def avg_regions(self, field: str, thresh: float = 0.0) -> float:
        """Function to compute average field value.

        Here average is taken over connected jet regions that are on-axis.
        """
        Vind = np.where((self.regions) > 0 and (field >= thresh))
        avg = np.mean(field[Vind])
        return avg

    def max_Wvelocity(self) -> float:
        """Function to return maximum vertical velocity."""
        return self.max_regions(self.Wvelocity)

    def avg_Wvelocity(self, Wthresh: float = 0.0) -> float:
        """Function to return average vertical velocity."""
        return self.avg_regions(self.Wvelocity, Wthresh=Wthresh)

    def get_eff_jet_mass(self, vel_thres: float = 0.1, asPercent: bool = False) -> float:
        """Function to compute effective jet mass.

        Effective jet mass is mass of jet with Wvelocity above
        a threshold and for a connected component that lies on
        the vertical axis.
        """
        eff_jet_mass = np.sum(self.get_eff_jet_mass_map())
        if asPercent:
            return eff_jet_mass / self.get_jet_mass()
        else:
            return eff_jet_mass

    def get_eff_jet_mass_map(self, vel_thres: float = 0.1) -> np.ndarray:
        """Function to compute effective jet mass map.

        Return effective jet mass for each cell/zone in simulation.
        """
        Vind = np.where((self.Wvelocity >= vel_thres) & (self.regions))
        eff_jet_mass_map = self.volume[Vind] * self.density[Vind] * self.vofm[Vind]
        return eff_jet_mass_map
