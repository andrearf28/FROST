import numpy as np
from typing import List, Tuple, Optional
from FROST_module import FROSTModule
from geometry_utils import calculate_photon_yield_for_module
from scipy import interpolate 
import os  

def interpolate_temporal_profile_file(input_filename, output_filename, bin_width=6.5, interpolation_factor=10):
    """
    Create an interpolated version of a temporal profile file and save it to disk.
    
    Args:
        input_filename: path to original file
        output_filename: path to save interpolated file
        bin_width: width of each time bin in ns (default: 6.5)
        interpolation_factor: number of interpolated points between original bins (default: 10)
    
    Returns:
        output_filename: path to the created interpolated file
    """
    print(f"\n🔄 Interpolating temporal profile:")
    print(f"   Input:  {input_filename}")
    print(f"   Output: {output_filename}")
    
    # Load photon counts (single column)
    photon_counts = np.loadtxt(input_filename)
    
    # Convert negative values to zero
    photon_counts[photon_counts < 0] = 0
    
    # Generate time array: START of each bin (not center)
    n_bins = len(photon_counts)
    times_original = np.arange(n_bins) * bin_width
    
    print(f"   Original bins: {n_bins}")
    
    if interpolation_factor > 1:
        # Create interpolation function (cubic spline for smooth curves)
        f_interp = interpolate.interp1d(
            times_original, 
            photon_counts, 
            kind='cubic',
            bounds_error=False,
            fill_value=0
        )
        
        # Create new time array with more points
        new_bin_width = bin_width / interpolation_factor
        n_new_bins = (n_bins - 1) * interpolation_factor + 1
        times = np.linspace(times_original[0], times_original[-1], n_new_bins)
        
        # Interpolate photon counts
        photon_counts_interp = f_interp(times)
        
        # Ensure no negative values after interpolation
        photon_counts_interp[photon_counts_interp < 0] = 0
        
        print(f"   Interpolated bins: {n_new_bins} (factor={interpolation_factor}x)")
        print(f"   New bin width: {new_bin_width:.3f} ns")
        
        # Save to file
        np.savetxt(output_filename, photon_counts_interp, fmt='%.6f')
        print(f"   ✅ Saved interpolated profile to: {output_filename}")
        
        return output_filename
    else:
        print(f"   ⚠️  No interpolation (factor=1), copying original file")
        np.savetxt(output_filename, photon_counts, fmt='%.6f')
        return output_filename


def load_temporal_profile(filename, bin_width=6.5):
    """
    Load temporal profile from txt file.
    Expected format: single column with number of photons per bin.
    Each line represents a time bin of 6.5 ns (default).
    
    Args:
        filename: path to file
        bin_width: width of each time bin in ns (default: 6.5)
    
    Returns:
        times: array of time values (ns) - START of each bin
        probabilities: normalized probability distribution
    """
    # Load photon counts (single column)
    photon_counts = np.loadtxt(filename)
    
    # Convert negative values to zero
    photon_counts[photon_counts < 0] = 0
    
    # Generate time array: START of each bin (not center)
    n_bins = len(photon_counts)
    times = np.arange(n_bins) * bin_width
    
    # Normalize to get probability distribution
    total = np.sum(photon_counts)
    if total > 0:
        probabilities = photon_counts / total
    else:
        raise ValueError(f"File {filename} contains no photons (sum = 0 after removing negatives)")
    
    print(f"   Loaded {len(times)} bins up to {times[-1]:.1f} ns")
    
    return times, probabilities


def sample_from_temporal_profile(times, probabilities, n_samples):
    """
    Sample emission times from a discrete temporal profile.
    
    Args:
        times: array of time values (ns)
        probabilities: probability for each time bin
        n_samples: number of samples to generate
    
    Returns:
        array of sampled emission times (ns)
    """
    # Ensure no negative probabilities
    probabilities = np.copy(probabilities)  # Don't modify the original
    probabilities[probabilities < 0] = 0
    
    # Re-normalize just in case
    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities = probabilities / prob_sum
    else:
        raise ValueError("All probabilities are zero or negative")
    
    # Sample directly from the distribution
    # The selected time IS the emission time (no offset needed)
    sampled_times = np.random.choice(times, size=n_samples, p=probabilities)
    
    return sampled_times


# ===== MAIN CLASS =====

class DetectorSimulation:
    def __init__(self, detector_dimensions, module_dimensions):
        """
        Initialize the FROST detector.
        
        Args:
            detector_dimensions: (length_x, height_y, depth_z) in meters
                - length_x: horizontal dimension in X (12m)
                - height_y: vertical HEIGHT dimension (12m)
                - depth_z: horizontal dimension in Z, beam axis (55m)
            module_dimensions: (width, height) of FROST module in meters
        """
        self.detector_length = detector_dimensions[0]  # X - horizontal (12m)
        self.detector_height = detector_dimensions[1]  # Y - HEIGHT vertical (12m)
        self.detector_depth = detector_dimensions[2]   # Z - beam axis (55m)
        
        self.module_width = module_dimensions[0]
        self.module_height = module_dimensions[1]
        
        self.modules: List[FROSTModule] = []
        
        # Physical parameters 
        self.n_photons_per_mev = 25000
        self.frost_efficiency = 0.05
        
        # LAr temporal parameters
        self.tau_fast = 6      # ns
        self.tau_slow = 100.0    # ns
        self.fast_fraction = 0.23
        
        # WLS parameters
        self.tau_wls = 1.2       # ns
        self.tau_ptp = 1.45      # ns
        
        # LAr propagation parameters
        self.n_lar = 1.23
        self.rayleigh_length = 8.0  # m
        self.rayleigh_power = 1.5
        self.c_light = 0.3          # m/ns
        
    def create_uniform_module_distribution(self,
                                          n_x=None,
                                          n_y=None,
                                          n_z=None):
        """
        Create a uniform distribution of FROST modules on the 4 lateral walls.
        
        Only creates modules on ±X and ±Z walls (the 4 vertical lateral walls).
        NO modules are created on ±Y walls (top/bottom).
        
        Args:
            n_x: Number of modules along X axis (12m)
            n_y: Number of modules along Y axis (12m height)
            n_z: Number of modules along Z axis (55m, beam axis)
        """
        module_id = 0
        
        # Default values
        if n_x is None:
            n_x = int(self.detector_length / self.module_width)
        if n_y is None:
            n_y = int(self.detector_height / self.module_height)
        if n_z is None:
            n_z = int(self.detector_depth / self.module_width)
        
        # Calculate spacing
        space_x = self.detector_length / n_x if n_x > 0 else 0
        space_y = self.detector_height / n_y if n_y > 0 else 0
        space_z = self.detector_depth / n_z if n_z > 0 else 0
        
        print(f"\n📐 DETECTOR CONFIGURATION:")
        print(f"   Dimensions: X={self.detector_length}m, Y={self.detector_height}m (height), Z={self.detector_depth}m (beam axis)")
        print(f"\n📊 Modules per axis: n_x={n_x}, n_y={n_y}, n_z={n_z}")
        print(f"   Spacing: X={space_x:.3f}m, Y={space_y:.3f}m, Z={space_z:.3f}m")
        
        # Calculate total range occupied by modules (centered)
        total_range_x = (n_x - 1) * space_x if n_x > 1 else 0
        total_range_y = (n_y - 1) * space_y if n_y > 1 else 0
        total_range_z = (n_z - 1) * space_z if n_z > 1 else 0
        
        print(f"   Occupied range (centered): X={total_range_x:.3f}m, Y={total_range_y:.3f}m, Z={total_range_z:.3f}m")
        
        # ====== X+ WALL (RIGHT) ======
        if n_y > 0 and n_z > 0:
            x_wall = self.detector_length / 2.0
            normal = np.array([-1.0, 0.0, 0.0])
            
            for iy in range(n_y):
                for iz in range(n_z):
                    # Modules are centered on the wall
                    y_center = -total_range_y/2.0 + iy * space_y
                    z_center = -total_range_z/2.0 + iz * space_z
                    position = np.array([x_wall, y_center, z_center])
                    
                    module = FROSTModule(
                        position=position,
                        normal=normal,
                        width=self.module_width,
                        height=self.module_height,
                        efficiency=self.frost_efficiency,
                        module_id=module_id
                    )
                    self.modules.append(module)
                    module_id += 1
        
        # ====== X- WALL (LEFT) ======
        if n_y > 0 and n_z > 0:
            x_wall = -self.detector_length / 2.0
            normal = np.array([+1.0, 0.0, 0.0])
            
            for iy in range(n_y):
                for iz in range(n_z):
                    y_center = -total_range_y/2.0 + iy * space_y
                    z_center = -total_range_z/2.0 + iz * space_z
                    position = np.array([x_wall, y_center, z_center])
                    
                    module = FROSTModule(
                        position=position,
                        normal=normal,
                        width=self.module_width,
                        height=self.module_height,
                        efficiency=self.frost_efficiency,
                        module_id=module_id
                    )
                    self.modules.append(module)
                    module_id += 1
        
        # ====== Z+ WALL (BACK) ======
        if n_x > 0 and n_y > 0:
            z_wall = self.detector_depth / 2.0
            normal = np.array([0.0, 0.0, -1.0])
            
            for ix in range(n_x):
                for iy in range(n_y):
                    x_center = -total_range_x/2.0 + ix * space_x
                    y_center = -total_range_y/2.0 + iy * space_y
                    position = np.array([x_center, y_center, z_wall])
                    
                    module = FROSTModule(
                        position=position,
                        normal=normal,
                        width=self.module_width,
                        height=self.module_height,
                        efficiency=self.frost_efficiency,
                        module_id=module_id
                    )
                    self.modules.append(module)
                    module_id += 1
        
        # ====== Z- WALL (FRONT) ======
        if n_x > 0 and n_y > 0:
            z_wall = -self.detector_depth / 2.0
            normal = np.array([0.0, 0.0, +1.0])
            
            for ix in range(n_x):
                for iy in range(n_y):
                    x_center = -total_range_x/2.0 + ix * space_x
                    y_center = -total_range_y/2.0 + iy * space_y
                    position = np.array([x_center, y_center, z_wall])
                    
                    module = FROSTModule(
                        position=position,
                        normal=normal,
                        width=self.module_width,
                        height=self.module_height,
                        efficiency=self.frost_efficiency,
                        module_id=module_id
                    )
                    self.modules.append(module)
                    module_id += 1
        
        # NO Y+ or Y- walls (top/bottom) are created
        
        total_modules = len(self.modules)
        if total_modules == 0:
            print("⚠️ WARNING: No modules were created.")
        else:
            print(f"\n✅ Created {total_modules} FROST modules")
            print(f"   X± walls (lateral): {2 * n_y * n_z if n_y > 0 and n_z > 0 else 0} modules")
            print(f"   Z± walls (lateral): {2 * n_x * n_y if n_x > 0 and n_y > 0 else 0} modules")
            print(f"   Y± walls (top/bottom): 0 modules (not created)")

    def calculate_light_yield_for_nearby_modules(self,
                                                 generation_point: np.ndarray,
                                                 kinetic_energy: float,
                                                 max_distance: float = 6.0):
        """
        Calculate light yield only for nearby modules (within max_distance).
        
        Args:
            generation_point: np.array([x, y, z]) generation point
            kinetic_energy: deposited energy (MeV)
            max_distance: maximum distance to consider a module (m)
        """
        n_modules_considered = 0
        
        for module in self.modules:
            distance = module.get_distance_to_point(generation_point)
            
            if distance <= max_distance:
                n_photons = calculate_photon_yield_for_module(
                    module,
                    generation_point,
                    kinetic_energy,
                    self.n_photons_per_mev
                )
                module.set_photon_yield(n_photons)
                n_modules_considered += 1
            else:
                # Module too far: zero photons
                module.set_photon_yield(0.0)
        
        print(f"✅ Calculated light yield for {n_modules_considered}/{len(self.modules)} modules "
              f"within {max_distance} m")
        
    def propagate_photons_temporally(self, generation_time, generation_point, 
                                     use_wls_smearing=False, wls_histogram_file=None,
                                     ar_fraction=0.6, xe_fraction=0.4,
                                     ar_tau_fast=6.0, ar_tau_slow=1600.0, ar_fast_fraction=0.3,
                                     xe_tau_fast=2.2, xe_tau_slow=27.0, xe_fast_fraction=0.3,
                                     ar_rayleigh_length=0.90, xe_rayleigh_length=0.30,
                                     use_temporal_profiles=False,
                                     xe_temporal_profile_file=None,
                                     ar_temporal_profile_file_total=None,
                                     ar_temporal_profile_file_xenon=None,
                                     time_range=(0, 100),
                                     interpolation_factor=10,
                                     interpolated_profiles_dir="Data/interpolated"):  # <-- AÑADIR parámetro
        """
        Propagates photons temporally with two components: Argon (128nm) and Xenon (175nm).
        Can use exponentials or temporal profiles from files.
        
        Args:
            generation_time: Event generation time (ns)
            generation_point: Generation point (x, y, z) in meters
            use_wls_smearing: If True, applies WLS temporal smearing
            wls_histogram_file: File with WLS temporal histogram
            ar_fraction: Fraction of Argon photons (default: 0.6)
            xe_fraction: Fraction of Xenon photons (default: 0.4)
            ar_tau_fast: Argon fast time in ns (default: 6.0)
            ar_tau_slow: Argon slow time in ns (default: 1600.0)
            ar_fast_fraction: Argon fast component fraction (default: 0.3)
            xe_tau_fast: Xenon fast time in ns (default: 2.2)
            xe_tau_slow: Xenon slow time in ns (default: 27.0)
            xe_fast_fraction: Xenon fast component fraction (default: 0.3)
            ar_rayleigh_length: Rayleigh length for Ar in METERS (default: 0.90 m = 90 cm)
            xe_rayleigh_length: Rayleigh length for Xe in METERS (default: 0.30 m = 30 cm)
            use_temporal_profiles: If True, use txt files; if False, use exponentials
            xe_temporal_profile_file: File with Xenon temporal profile (direct)
            ar_temporal_profile_file_total: Total profile file for Argon calculation
            ar_temporal_profile_file_xenon: Xenon file to subtract from total (to get Argon)
            time_range: tuple (tmin, tmax) for simulation window in ns
            interpolation_factor: number of interpolated points between original bins (default: 10)
            interpolated_profiles_dir: directory to save interpolated profiles (default: "Data/interpolated")
        """
        # If WLS smearing is requested, load the histogram
        cdf_wls = None
        bin_edges_wls = None
        
        if use_wls_smearing and wls_histogram_file is not None:
            cdf_wls, bin_edges_wls = self._load_wls_histogram(wls_histogram_file)
            print("⚠️ Using Geant4 WLS histogram: pTP and WLS exponential times are DISABLED")
        
        # Load temporal profiles if needed
        ar_times, ar_probs = None, None
        xe_times, xe_probs = None, None
        
        if use_temporal_profiles:
            # Create directory for interpolated profiles if it doesn't exist
            os.makedirs(interpolated_profiles_dir, exist_ok=True)
            
            # Calculate new bin width after interpolation
            original_bin_width = 6.5  # ns
            new_bin_width = original_bin_width / interpolation_factor if interpolation_factor > 1 else original_bin_width
            
            # Load Xenon profile (DIRECT) - with interpolation
            if xe_temporal_profile_file:
                # Create interpolated file
                base_name = os.path.basename(xe_temporal_profile_file)
                name_without_ext = os.path.splitext(base_name)[0]
                xe_interp_file = os.path.join(
                    interpolated_profiles_dir, 
                    f"{name_without_ext}_interp{interpolation_factor}x.txt"
                )
                
                # Interpolate and save
                interpolate_temporal_profile_file(
                    xe_temporal_profile_file,
                    xe_interp_file,
                    bin_width=original_bin_width,
                    interpolation_factor=interpolation_factor
                )
                
                # Load interpolated profile
                xe_times, xe_probs = load_temporal_profile(xe_interp_file, bin_width=new_bin_width)
                print(f"✅ Loaded Xenon temporal profile: {len(xe_times)} bins, sum={np.sum(xe_probs):.6f}")
            
            # Load and process Argon profile (file_total - file_xenon) - with interpolation
            if ar_temporal_profile_file_total and ar_temporal_profile_file_xenon:
                # Interpolate total file
                base_name_total = os.path.basename(ar_temporal_profile_file_total)
                name_without_ext_total = os.path.splitext(base_name_total)[0]
                total_interp_file = os.path.join(
                    interpolated_profiles_dir,
                    f"{name_without_ext_total}_interp{interpolation_factor}x.txt"
                )
                
                interpolate_temporal_profile_file(
                    ar_temporal_profile_file_total,
                    total_interp_file,
                    bin_width=original_bin_width,
                    interpolation_factor=interpolation_factor
                )
                
                # Interpolate xenon file (for subtraction)
                base_name_xe_sub = os.path.basename(ar_temporal_profile_file_xenon)
                name_without_ext_xe_sub = os.path.splitext(base_name_xe_sub)[0]
                xe_sub_interp_file = os.path.join(
                    interpolated_profiles_dir,
                    f"{name_without_ext_xe_sub}_interp{interpolation_factor}x.txt"
                )
                
                interpolate_temporal_profile_file(
                    ar_temporal_profile_file_xenon,
                    xe_sub_interp_file,
                    bin_width=original_bin_width,
                    interpolation_factor=interpolation_factor
                )
                
                # Load interpolated profiles
                times_total, probs_total = load_temporal_profile(total_interp_file, bin_width=new_bin_width)
                times_xe_sub, probs_xe_sub = load_temporal_profile(xe_sub_interp_file, bin_width=new_bin_width)
                
                if not np.allclose(times_total, times_xe_sub):
                    raise ValueError("Argon profile files must have same time binning")
                
                ar_times = times_total
                ar_probs = probs_total - probs_xe_sub
                
                n_negative = np.sum(ar_probs < 0)
                if n_negative > 0:
                    print(f"⚠️  Warning: {n_negative} bins with negative values in Argon profile (total - xenon). Setting to zero.")
                ar_probs[ar_probs < 0] = 0
                
                total = np.sum(ar_probs)
                if total > 0:
                    ar_probs = ar_probs / total
                    print(f"✅ Loaded Argon temporal profile: {len(ar_times)} bins, sum={np.sum(ar_probs):.6f}")
                else:
                    raise ValueError("Argon profile after subtraction is empty (all values <= 0)")
        
        # ===== SINGLE LOOP OVER MODULES =====
        for module in self.modules:
            n_photons_total = int(round(module.n_photons_arriving))
            
            if n_photons_total <= 0:
                module.photon_times = np.array([])
                module.n_photons_detected = 0
                continue
            
            # Get distance to module
            distance = np.linalg.norm(module.position - generation_point)
            
            # Split photons into Argon and Xenon
            n_photons_ar = int(round(n_photons_total * ar_fraction))
            n_photons_xe = int(round(n_photons_total * xe_fraction))
            
            # Ensure the sum is correct
            if n_photons_ar + n_photons_xe < n_photons_total:
                n_photons_ar += (n_photons_total - n_photons_ar - n_photons_xe)
            elif n_photons_ar + n_photons_xe > n_photons_total:
                if n_photons_xe > 0:
                    n_photons_xe -= (n_photons_ar + n_photons_xe - n_photons_total)
                else:
                    n_photons_ar -= (n_photons_ar + n_photons_xe - n_photons_total)
            
            photon_times = []
            photon_types = []  # 0 = Ar, 1 = Xe  <-- CORRECCIÓN: Añadir []
            
            # ===== ARGON PHOTONS (128 nm) =====
            if n_photons_ar > 0:
                if use_temporal_profiles and ar_times is not None:
                    emission_times_ar = sample_from_temporal_profile(ar_times, ar_probs, n_photons_ar)
                    photon_times.extend(emission_times_ar)
                    photon_types.extend([0] * n_photons_ar)
                else:
                    n_ar_fast = int(round(n_photons_ar * ar_fast_fraction))
                    n_ar_slow = n_photons_ar - n_ar_fast
                    
                    if n_ar_fast > 0:
                        emission_times_ar_fast = np.random.exponential(ar_tau_fast, n_ar_fast)
                        photon_times.extend(emission_times_ar_fast)
                        photon_types.extend([0] * n_ar_fast)
                    
                    if n_ar_slow > 0:
                        emission_times_ar_slow = np.random.exponential(ar_tau_slow, n_ar_slow)
                        photon_times.extend(emission_times_ar_slow)
                        photon_types.extend([0] * n_ar_slow)
            
            # ===== XENON PHOTONS (175 nm) =====
            if n_photons_xe > 0:
                if use_temporal_profiles and xe_times is not None:
                    emission_times_xe = sample_from_temporal_profile(xe_times, xe_probs, n_photons_xe)
                    photon_times.extend(emission_times_xe)
                    photon_types.extend([1] * n_photons_xe)
                else:
                    n_xe_fast = int(round(n_photons_xe * xe_fast_fraction))
                    n_xe_slow = n_photons_xe - n_xe_fast
                    
                    if n_xe_fast > 0:
                        emission_times_xe_fast = np.random.exponential(xe_tau_fast, n_xe_fast)
                        photon_times.extend(emission_times_xe_fast)
                        photon_types.extend([1] * n_xe_fast)
                    
                    if n_xe_slow > 0:
                        emission_times_xe_slow = np.random.exponential(xe_tau_slow, n_xe_slow)
                        photon_times.extend(emission_times_xe_slow)
                        photon_types.extend([1] * n_xe_slow)
            
            # Convert to numpy arrays
            photon_times = np.array(photon_times)
            photon_types = np.array(photon_types, dtype=int)
            
            # ===== RAYLEIGH SCATTERING IN LAr - TEMPORAL SMEARING =====
            # Similar to FROST_simu.py: prop_lar() adds temporal dispersion
            delta_t_rayleigh = (distance ** self.rayleigh_power) * self.n_lar / (self.c_light * self.rayleigh_length)
            rayleigh_delays = np.random.normal(loc=0, scale=delta_t_rayleigh, size=len(photon_times))
            photon_times += rayleigh_delays
            
            # ===== APPLY WLS SMEARING (OPTIONAL) =====
            if use_wls_smearing and cdf_wls is not None:
                wls_delays = np.array([self._sample_wls_time(cdf_wls, bin_edges_wls) 
                                      for _ in range(len(photon_times))])
                photon_times += wls_delays
            
            # ===== APPLY RAYLEIGH SCATTERING - ATTENUATION =====
            # Now apply survival probability based on distance
            survival_prob = np.zeros(len(photon_types))
            ar_mask = photon_types == 0
            xe_mask = photon_types == 1
            
            survival_prob[ar_mask] = np.exp(-distance / ar_rayleigh_length)
            survival_prob[xe_mask] = np.exp(-distance / xe_rayleigh_length)
            
            survived = np.random.random(len(photon_times)) < survival_prob
            photon_times = photon_times[survived]
            
            # Add generation_time at the end
            photon_times += generation_time
            
            # Save to module (sin filtro temporal)
            module.photon_times = photon_times
            module.n_photons_detected = len(photon_times)
    
    def _load_wls_histogram(self, filename: str):
        """
        Load WLS plate propagation histogram from file.
        Compatible with FROST_simu.py format.
        
        Returns:
            (cdf, bin_edges): tuple with CDF and bin edges for inverse sampling
        """
        import os
        
        if not os.path.exists(filename):
            print(f"⚠️ Warning: WLS histogram file not found: {filename}")
            return None, None
        
        values = []
        counts = []
        
        with open(filename, "r") as f:
            for line in f:
                if line[0] == '#':
                    continue
                if line.strip():
                    v, c = line.split()
                    values.append(float(v))
                    counts.append(int(c))
        
        # Create histogram
        hist, bin_edges = np.histogram(values, bins=len(values), weights=counts)
        
        # Rebinning (same factor as FROST_simu.py)
        hist, bin_edges = self._rebin_histogram(hist, bin_edges, factor=20)
        
        # Normalize
        bin_widths = np.diff(bin_edges)
        hist_normalized = hist / np.sum(hist * bin_widths)
        
        # Calculate CDF
        cdf = np.cumsum(hist_normalized * bin_widths)
        cdf = cdf / cdf[-1]  # ensure last value is 1
        
        return cdf, bin_edges
    
    def _rebin_histogram(self, hist, bin_edges, factor):
        """
        Reduce the number of bins by combining adjacent bins.
        """
        if len(hist) % factor != 0:
            # Truncate to be divisible
            new_len = (len(hist) // factor) * factor
            hist = hist[:new_len]
            bin_edges = bin_edges[:new_len + 1]
        
        new_hist = hist.reshape(-1, factor).sum(axis=1)
        new_bin_edges = bin_edges[::factor]
        
        if len(new_bin_edges) != len(new_hist) + 1:
            new_bin_edges = np.append(new_bin_edges, bin_edges[-1])
        
        return new_hist, new_bin_edges
    
    def _sample_wls_time(self, cdf, bin_edges):
        """
        Sample a random time from the WLS distribution using CDF inversion.
        """
        u = np.random.rand()
        delta_t = np.interp(u, cdf, bin_edges[1:])
        return delta_t

    def apply_electronics_to_all_modules(self, sipm_tau_rise=1.5, sipm_tau_decay=15.0, 
                                        spe_amplitude=10.0, adc_sampling_time=1.0,
                                        time_range=(0, 100), baseline_noise_sigma=0.0):
        """
        Apply electronic response to all modules.
        
        Args:
            (see FROSTModule.apply_electronics for parameter documentation)
        """
        for module in self.modules:
            # First create temporal histogram
            n_bins = int((time_range[1] - time_range[0]) / adc_sampling_time)
            module.create_temporal_histogram(time_range, n_bins)
            
            # Then apply electronics
            module.apply_electronics(
                sipm_tau_rise=sipm_tau_rise,
                sipm_tau_decay=sipm_tau_decay,
                spe_amplitude=spe_amplitude,
                adc_sampling_time=adc_sampling_time,
                time_range=time_range,
                baseline_noise_sigma=baseline_noise_sigma
            )
            
    def sum_waveforms(self, module_indices=None):
        """
        Sum waveforms from specified modules.
        
        Args:
            module_indices: list of module indices to sum. If None, sum all.
            
        Returns:
            (time_axis, summed_waveform): tuple with time axis and summed waveform
        """
        if module_indices is None:
            modules_to_sum = self.modules
        else:
            modules_to_sum = [self.modules[i] for i in module_indices]
        
        # Verify all have waveforms
        valid_modules = [m for m in modules_to_sum if m.waveform is not None]
        
        if len(valid_modules) == 0:
            print("⚠️ No modules with waveforms to sum")
            return None, None
        
        # Assume all have the same time axis
        time_axis = valid_modules[0].time_axis
        summed_waveform = np.zeros_like(time_axis, dtype=float)
        
        for module in valid_modules:
            if len(module.waveform) == len(summed_waveform):
                summed_waveform += module.waveform
            else:
                print(f"⚠️ Module {module.module_id} has different waveform length, skipping")
        
        print(f"✅ Summed waveforms from {len(valid_modules)} modules")
        return time_axis, summed_waveform
    
    def get_modules_with_signal(self, min_photons=1):
        """
        Return list of modules that detected at least min_photons photons.
        
        Args:
            min_photons: minimum photon threshold
            
        Returns:
            list of FROSTModule
        """
        return [m for m in self.modules if m.n_photons_detected >= min_photons]
