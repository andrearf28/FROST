import numpy as np
import matplotlib.pyplot as plt
from detector_simulation import DetectorSimulation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== CONFIGURATION PARAMETERS ====================

# Detector geometry
detector_dimensions = (12.0, 12.0, 55.0)  # (X, Y, Z) in meters
module_dimensions = (0.252, 0.252)          # (width, height) in meters

# Module distribution
n_modules_x = 6  # Number of modules along X
n_modules_y = 6   # Number of modules along Y (height)
n_modules_z = 27  # Number of modules along Z (beam axis)

# Physical simulation parameters
max_distance_light_yield = 10.0  # meters - Maximum distance to calculate light yield

# Particle energies
kaon_energy = 105   # MeV
muon_energy = 152   # MeV
k_decay_time = 20.0 # ns

# Data files
wls_histogram_file = "FROST_252x252_sipms_42_2boards_temporal_histogram.txt"
use_wls_smearing = False

# Light emission parameters - NEW
# Argon (128 nm)
ar_fraction = 0.60         # 60% of photons are Argon
ar_tau_fast = 6.0           # ns - Fast component of Argon
ar_tau_slow = 1500.0        # ns - Slow component of Argon
ar_fast_fraction = 0.23     # 30% of Ar photons are fast
ar_rayleigh_length = 1   # m - Rayleigh length for Ar @ 128nm 

# Xenon (175 nm)
xe_fraction = 0.40        # 40% of photons are Xenon
xe_tau_fast = 3        # ns - Fast component of Xenon
xe_tau_slow = 100         # ns - Slow component of Xenon
xe_fast_fraction = 0.23     # 30% of Xe photons are fast
xe_rayleigh_length = 8   # m - Rayleigh length for Xe @ 175nm 

# NEW: Temporal profile configuration
use_temporal_profiles = True # True: use txt files, False: use exponentials (tau_fast/tau_slow)
interpolation_factor = 10  # <-- Number of interpolated points between original bins (1 = no interpolation)
interpolated_profiles_dir = "Data/interpolated"  # <-- Directory to save interpolated profiles

# Xenon temporal profile file (single file with 6.6 ns binning) - DIRECT
xe_temporal_profile_file = "Data/3Q.txt"  # Format: time(ns) probability

# Argon temporal profile files (subtraction: file_total - file_xenon)
ar_temporal_profile_file_total = "Data/3nonQ.txt"     # Total profile (Ar + Xe)
ar_temporal_profile_file_xenon = "Data/3Q.txt"     # Xenon profile (to subtract)

# ======================================================================

def simulate_event(detector, generation_point, kaon_energy, muon_energy, 
                   k_decay_time, generate_muon=True, wls_file=None, use_wls_smearing=False):
    """
    Simulates a complete event (kaon or kaon+muon).
    
    Returns:
        detector with updated modules
    """
    # Clean previous state of modules
    for module in detector.modules:
        module.n_photons_arriving = 0.0
        module.n_photons_detected = 0
        module.photon_times = None
        module.temporal_histogram = None
        module.waveform = None
    
    # ========== CALCULATE LIGHT YIELD ==========
    detector.calculate_light_yield_for_nearby_modules(
        generation_point=generation_point,
        kinetic_energy=kaon_energy,
        max_distance=max_distance_light_yield
    )
   
    for module in detector.modules:
        module.n_photons_arriving = int(round(module.n_photons_arriving))
    
    # ========== PROPAGATE KAON PHOTONS ==========
    detector.propagate_photons_temporally(
        generation_time=0.0,
        generation_point=generation_point,
        use_wls_smearing=use_wls_smearing,
        wls_histogram_file=wls_file if use_wls_smearing else None,
        ar_fraction=ar_fraction,
        xe_fraction=xe_fraction,
        ar_tau_fast=ar_tau_fast,
        ar_tau_slow=ar_tau_slow,
        ar_fast_fraction=ar_fast_fraction,
        xe_tau_fast=xe_tau_fast,
        xe_tau_slow=xe_tau_slow,
        xe_fast_fraction=xe_fast_fraction,
        ar_rayleigh_length=ar_rayleigh_length,
        xe_rayleigh_length=xe_rayleigh_length,
        use_temporal_profiles=use_temporal_profiles,
        xe_temporal_profile_file=xe_temporal_profile_file,
        ar_temporal_profile_file_total=ar_temporal_profile_file_total,
        ar_temporal_profile_file_xenon=ar_temporal_profile_file_xenon,
        time_range=(0, 100),
        interpolation_factor=interpolation_factor,  # <-- AÑADIR
        interpolated_profiles_dir=interpolated_profiles_dir  # <-- AÑADIR
    )
    
    # Save kaon photon_times
    kaon_photon_times = {}
    for module in detector.modules:
        if module.photon_times is not None and len(module.photon_times) > 0:
            kaon_photon_times[module.module_id] = module.photon_times.copy()
    
    # ========== IF THERE'S MUON, ADD ITS PHOTONS ==========
    if generate_muon:
        detector.calculate_light_yield_for_nearby_modules(
            generation_point=generation_point,
            kinetic_energy=muon_energy,
            max_distance=max_distance_light_yield
        )
        
        # EXPLICIT CONVERSION TO INTEGERS - ALSO HERE
        for module in detector.modules:
            module.n_photons_arriving = int(round(module.n_photons_arriving))
        
        detector.propagate_photons_temporally(
            generation_time=k_decay_time,
            generation_point=generation_point,
            use_wls_smearing=use_wls_smearing,
            wls_histogram_file=wls_file if use_wls_smearing else None,
            ar_fraction=ar_fraction,
            xe_fraction=xe_fraction,
            ar_tau_fast=ar_tau_fast,
            ar_tau_slow=ar_tau_slow,
            ar_fast_fraction=ar_fast_fraction,
            xe_tau_fast=xe_tau_fast,
            xe_tau_slow=xe_tau_slow,
            xe_fast_fraction=xe_fast_fraction,
            ar_rayleigh_length=ar_rayleigh_length,
            xe_rayleigh_length=xe_rayleigh_length,
            use_temporal_profiles=use_temporal_profiles,
            xe_temporal_profile_file=xe_temporal_profile_file,
            ar_temporal_profile_file_total=ar_temporal_profile_file_total,
            ar_temporal_profile_file_xenon=ar_temporal_profile_file_xenon,
            time_range=(0, 100),
            interpolation_factor=interpolation_factor,  # <-- AÑADIR
            interpolated_profiles_dir=interpolated_profiles_dir  # <-- AÑADIR
        )
        
        # Combine kaon and muon photons
        for module in detector.modules:
            if module.module_id in kaon_photon_times:
                if module.photon_times is not None and len(module.photon_times) > 0:
                    # Combine both arrays
                    combined = np.concatenate([
                        kaon_photon_times[module.module_id],
                        module.photon_times
                    ])
                    module.photon_times = combined
                    module.n_photons_detected = len(combined)
                else:
                    # Only kaon photons
                    module.photon_times = kaon_photon_times[module.module_id]
                    module.n_photons_detected = len(module.photon_times)
    
    # ========== APPLY ELECTRONICS ==========
    # Cambiar time_range a un rango más amplio para capturar toda la waveform
    detector.apply_electronics_to_all_modules(
        sipm_tau_rise=1.5,
        sipm_tau_decay=3,
        spe_amplitude=10.0,
        adc_sampling_time=1,
        time_range=(0, 500),  # <-- CAMBIAR: de 100 a 500 ns (o más si es necesario)
        baseline_noise_sigma=0.01
    )
    
    return detector


def plot_module_comparison(module, kaon_only_signal, kaon_only_histogram,
                           kaon_muon_signal, kaon_muon_histogram, 
                           time_axis, row, col, fig, title, ncols):
    """
    Adds a subplot showing kaon-only vs kaon+muon signal.
    """
    # If no signal, just show the title
    if kaon_muon_signal is None or len(kaon_muon_signal) == 0:
        # Add text indicating no signal
        fig.add_annotation(
            text="No signal",
            xref=f"x{((row-1)*5 + col)}" if ((row-1)*5 + col) > 1 else "x",
            yref=f"y{((row-1)*5 + col)}" if ((row-1)*5 + col) > 1 else "y",
            x=0.5,
            y=0.5,
            xanchor='center',
            yanchor='middle',
            showarrow=False,
            font=dict(size=14, color="gray")
        )
    else:
        # Add temporal photon histogram (kaon+muon, blue bars)
        fig.add_trace(go.Bar(
            x=time_axis,
            y=kaon_muon_histogram,
            name='Photon histogram',
            marker_color='lightblue',
            showlegend=(row==1 and col==1)
        ), row=row, col=col)
        
        # Add kaon+muon signal (red line)
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=kaon_muon_signal,
            mode='lines+markers',
            name='Kaon + Muon',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            showlegend=(row==1 and col==1)
        ), row=row, col=col)
        
        # Add kaon-only signal (blue line)
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=kaon_only_signal,
            mode='lines+markers',
            name='Kaon only',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            showlegend=(row==1 and col==1)
        ), row=row, col=col)


def main():

    print("="*80)
    print("FROST DETECTOR SIMULATION")
    print("="*80)
    
    # ========== DETECTOR CONFIGURATION ==========
    detector = DetectorSimulation(
        detector_dimensions=detector_dimensions,
        module_dimensions=module_dimensions
    )
    
    detector.create_uniform_module_distribution(
        n_x=n_modules_x,
        n_y=n_modules_y,
        n_z=n_modules_z
    )
    
    # ========== DETAILED GEOMETRY INFORMATION ==========
    print("\n" + "="*80)
    print("📐 DETECTOR GEOMETRY")
    print("="*80)
    
    print(f"\n🔷 Total active volume dimensions:")
    print(f"   • X axis (horizontal): {detector.detector_length:.2f} m")
    print(f"   • Y axis (vertical height): {detector.detector_height:.2f} m")
    print(f"   • Z axis (beam axis): {detector.detector_depth:.2f} m")
    print(f"   • Total volume: {detector.detector_length * detector.detector_height * detector.detector_depth:.2f} m³")
    
    print(f"\n📦 Each FROST module dimensions:")
    print(f"   • Width: {detector.module_width:.3f} m ({detector.module_width*100:.1f} cm)")
    print(f"   • Height: {detector.module_height:.3f} m ({detector.module_height*100:.1f} cm)")
    print(f"   • Active area per module: {detector.module_width * detector.module_height:.4f} m² ({detector.module_width * detector.module_height * 10000:.1f} cm²)")
    
    # Calculate spacing between modules
    space_x = detector.detector_length / n_modules_x if n_modules_x > 0 else 0
    space_y = detector.detector_height / n_modules_y if n_modules_y > 0 else 0
    space_z = detector.detector_depth / n_modules_z if n_modules_z > 0 else 0
    
    # Calculate occupied ranges (centered)
    total_range_x = (n_modules_x - 1) * space_x if n_modules_x > 1 else 0
    total_range_y = (n_modules_y - 1) * space_y if n_modules_y > 1 else 0
    total_range_z = (n_modules_z - 1) * space_z if n_modules_z > 1 else 0
    
    print(f"\n📊 Module distribution per axis:")
    print(f"   • X axis: {n_modules_x} modules")
    print(f"     - Spacing between centers: {space_x:.3f} m ({space_x*100:.1f} cm)")
    print(f"     - Total occupied range: {total_range_x:.3f} m (from first to last module)")
    print(f"     - Module positions: from x={-total_range_x/2:.2f}m to x={+total_range_x/2:.2f}m")
    
    print(f"   • Y axis: {n_modules_y} modules")
    print(f"     - Spacing between centers: {space_y:.3f} m ({space_y*100:.1f} cm)")
    print(f"     - Total occupied range: {total_range_y:.3f} m (from first to last module)")
    print(f"     - Module positions: from y={-total_range_y/2:.2f}m to y={+total_range_y/2:.2f}m")
    
    print(f"   • Z axis: {n_modules_z} modules")
    print(f"     - Spacing between centers: {space_z:.3f} m ({space_z*100:.1f} cm)")
    print(f"     - Total occupied range: {total_range_z:.3f} m (from first to last module)")
    print(f"     - Module positions: from z={-total_range_z/2:.2f}m to z={+total_range_z/2:.2f}m")
    
    print(f"\n🔢 Module distribution per wall:")
    n_modules_x_walls = 2 * n_modules_y * n_modules_z
    n_modules_z_walls = 2 * n_modules_x * n_modules_y
    total_modules = n_modules_x_walls + n_modules_z_walls
    
    print(f"   • X+ wall (right): {n_modules_y}×{n_modules_z} = {n_modules_y * n_modules_z} modules")
    print(f"   • X- wall (left): {n_modules_y}×{n_modules_z} = {n_modules_y * n_modules_z} modules")
    print(f"   • Z+ wall (back): {n_modules_x}×{n_modules_y} = {n_modules_x * n_modules_y} modules")
    print(f"   • Z- wall (front): {n_modules_x}×{n_modules_y} = {n_modules_x * n_modules_y} modules")
    print(f"   • Y± walls (top/bottom): 0 modules")
    print(f"   • TOTAL: {total_modules} modules")
    
    # Calculate total covered area
    area_per_wall_x = n_modules_y * n_modules_z * detector.module_width * detector.module_height
    area_per_wall_z = n_modules_x * n_modules_y * detector.module_width * detector.module_height
    total_active_area = 2 * (area_per_wall_x + area_per_wall_z)
    
    # Calculate total available area on walls
    area_wall_x = detector.detector_height * detector.detector_depth
    area_wall_z = detector.detector_length * detector.detector_height
    total_wall_area = 2 * (area_wall_x + area_wall_z)
    
    coverage_percent = (total_active_area / total_wall_area) * 100
    
    print(f"\n📏 Area coverage:")
    print(f"   • Total active area (SiPMs): {total_active_area:.3f} m² ({total_active_area*10000:.1f} cm²)")
    print(f"   • Total wall area: {total_wall_area:.3f} m²")
    print(f"   • Coverage: {coverage_percent:.2f}%")
    
    print(f"\n⚙️ Simulation parameters:")
    print(f"   • Detection efficiency (frost_efficiency): {detector.frost_efficiency*100:.1f}%")
    print(f"   • Maximum distance for light yield: {max_distance_light_yield:.1f} m")
    print(f"   • LAr refractive index: {detector.n_lar}")
    print(f"   • Rayleigh length: {detector.rayleigh_length} m")
    
    # Random generation point - CORRECTED
    generation_point = np.array([
        np.random.uniform(-detector.detector_length/2, detector.detector_length/2),
        np.random.uniform(-detector.detector_height/2, detector.detector_height/2),
        np.random.uniform(-detector.detector_depth/2, detector.detector_depth/2)
    ])
    
    print(f"\n📍 Event generation point:")
    print(f"   • Coordinates: ({generation_point[0]:.2f}, {generation_point[1]:.2f}, {generation_point[2]:.2f}) m")
    print(f"   • Distance to detector center: {np.linalg.norm(generation_point):.2f} m")
    
    # Calculate distances to each wall (PERPENDICULAR)
    dist_to_x_plus = abs(generation_point[0] - detector.detector_length/2)
    dist_to_x_minus = abs(generation_point[0] + detector.detector_length/2)
    dist_to_y_plus = abs(generation_point[1] - detector.detector_height/2)
    dist_to_y_minus = abs(generation_point[1] + detector.detector_height/2)
    dist_to_z_plus = abs(generation_point[2] - detector.detector_depth/2)
    dist_to_z_minus = abs(generation_point[2] + detector.detector_depth/2)
    
    print(f"   • Perpendicular distances to each wall:")
    print(f"     - X+ wall: {dist_to_x_plus:.2f} m")
    print(f"     - X- wall: {dist_to_x_minus:.2f} m")
    print(f"     - Y+ wall (ceiling): {dist_to_y_plus:.2f} m")
    print(f"     - Y- wall (floor): {dist_to_y_minus:.2f} m")
    print(f"     - Z+ wall: {dist_to_z_plus:.2f} m")
    print(f"     - Z- wall: {dist_to_z_minus:.2f} m")
    
    min_dist_to_walls = min(dist_to_x_plus, dist_to_x_minus, dist_to_y_plus, 
                            dist_to_y_minus, dist_to_z_plus, dist_to_z_minus)
    print(f"     - Closest wall (perpendicular distance): {min_dist_to_walls:.2f} m")
    
    print(f"\n⚛️ Particle energies:")
    print(f"   • Kaon: {kaon_energy} MeV")
    print(f"   • Muon: {muon_energy} MeV")
    print(f"   • K+ decay time: {k_decay_time} ns")
    
    print(f"\n💡 Light emission parameters:")
    print(f"   • Using temporal profiles from files: {use_temporal_profiles}")
    
    if use_temporal_profiles:
        print(f"   • Xenon (175 nm): {xe_fraction*100:.0f}% of photons")
        print(f"     - Temporal profile file: {xe_temporal_profile_file}")
        print(f"     - Rayleigh length: {xe_rayleigh_length:.2f} m")
        print(f"   • Argon (128 nm): {ar_fraction*100:.0f}% of photons")
        print(f"     - Temporal profile (file_total - file_xenon):")
        print(f"       * Total file: {ar_temporal_profile_file_total}")
        print(f"       * Xenon file: {ar_temporal_profile_file_xenon}")
        print(f"     - Rayleigh length: {ar_rayleigh_length:.2f} m")
    else:
        print(f"   • Argon (128 nm): {ar_fraction*100:.0f}% of photons")
        print(f"     - τ_fast: {ar_tau_fast} ns ({ar_fast_fraction*100:.0f}% of Ar photons)")
        print(f"     - τ_slow: {ar_tau_slow} ns ({(1-ar_fast_fraction)*100:.0f}% of Ar photons)")
        print(f"     - Rayleigh length: {ar_rayleigh_length:.2f} m")
        print(f"   • Xenon (175 nm): {xe_fraction*100:.0f}% of photons")
        print(f"     - τ_fast: {xe_tau_fast} ns ({xe_fast_fraction*100:.0f}% of Xe photons)")
        print(f"     - τ_slow: {xe_tau_slow} ns ({(1-xe_fast_fraction)*100:.0f}% of Xe photons)")
        print(f"     - Rayleigh length: {xe_rayleigh_length:.2f} m")
    
    # ========== SIMULATION 1: KAON ONLY ==========
    print("\n" + "="*80)
    print("SIMULATING: Kaon only (no decay)")
    print("="*80)
    
    # CORRECTED: Use total energy (kaon + muon) for kaon only
    total_energy = kaon_energy + muon_energy
    
    detector_kaon_only = simulate_event(
        detector, generation_point, 
        total_energy,  # <-- Change: was kaon_energy before
        muon_energy,   # This parameter not used when generate_muon=False
        k_decay_time, 
        generate_muon=False, 
        wls_file=wls_histogram_file, 
        use_wls_smearing=use_wls_smearing
    )
    
    # Save kaon-only results
    kaon_only_results = {}
    for module in detector_kaon_only.modules:
        if module.n_photons_detected > 0:
            kaon_only_results[module.module_id] = {
                'waveform': module.waveform.copy() if module.waveform is not None else None,
                'histogram': module.temporal_histogram[1].copy() if module.temporal_histogram is not None else None,
                'time_axis': module.time_axis.copy() if module.time_axis is not None else None,
                'position': module.position.copy(),
                'n_photons': module.n_photons_detected
            }
    
    # ========== SIMULATION 2: KAON + MUON ==========
    print("\n" + "="*80)
    print("SIMULATING: Kaon + Muon (with decay)")
    print("="*80)
    
    detector_kaon_muon = simulate_event(
        detector, generation_point, kaon_energy, muon_energy,
        k_decay_time, generate_muon=True,
        wls_file=wls_histogram_file, use_wls_smearing=use_wls_smearing
    )
    
    # ========== SELECT MODULES TO PLOT ==========
    # Calculate modules with signal
    modules_with_signal = sorted(
        [m for m in detector_kaon_muon.modules if m.n_photons_detected > 0],
        key=lambda m: m.n_photons_detected,
        reverse=True
    )
    
    # CHANGE: Now we plot ALL modules, sorted by ID
    all_modules_sorted = sorted(detector_kaon_muon.modules, key=lambda m: m.module_id)
    modules_to_plot = all_modules_sorted  # Show all modules
    
    n_modules_to_plot = len(modules_to_plot)
    n_modules_with_signal = len(modules_with_signal)
    
    # ========== IMPROVED TERMINAL INFORMATION ==========
    print("\n" + "="*80)
    print("📊 DETECTED SIGNALS SUMMARY")
    print("="*80)
    
    total_photons = sum(m.n_photons_detected for m in detector_kaon_muon.modules)
    
    print(f"\n🔢 Global statistics:")
    print(f"   • Total installed modules: {len(detector.modules)}")
    print(f"   • Modules with signal: {n_modules_with_signal}/{len(detector.modules)} ({100*n_modules_with_signal/len(detector.modules):.1f}%)")
    print(f"   • Total photons detected: {total_photons}")
    
    if n_modules_with_signal == 0:
        print("\n⚠️ No module received signal")
        print("   Possible causes:")
        print("   - Generation point too far from walls")
        print("   - max_distance too small in calculate_light_yield")
        print("   - frost_efficiency too low")
        return
    
    # Signal distribution per wall
    print(f"\n📍 Signal distribution per wall:")
    walls = {'X+': 0, 'X-': 0, 'Y+': 0, 'Y-': 0, 'Z+': 0, 'Z-': 0}
    photons_per_wall = {'X+': 0, 'X-': 0, 'Y+': 0, 'Y-': 0, 'Z+': 0, 'Z-': 0}
    
    tolerance = 0.01
    for m in modules_with_signal:
        pos = m.position
        wall = "?"
        
        if abs(pos[0] - detector.detector_length/2) < tolerance:
            wall = "X+"
        elif abs(pos[0] + detector.detector_length/2) < tolerance:
            wall = "X-"
        elif abs(pos[1] - detector.detector_height/2) < tolerance:
            wall = "Y+"
        elif abs(pos[1] + detector.detector_height/2) < tolerance:
            wall = "Y-"
        elif abs(pos[2] - detector.detector_depth/2) < tolerance:
            wall = "Z+"
        elif abs(pos[2] + detector.detector_depth/2) < tolerance:
            wall = "Z-"
        
        if wall in walls:
            walls[wall] += 1
            photons_per_wall[wall] += m.n_photons_detected
    
    for wall in ['X+', 'X-', 'Z+', 'Z-']:  # Only lateral walls
        if walls[wall] > 0:
            avg_photons = photons_per_wall[wall] / walls[wall]
            print(f"   • {wall} wall: {walls[wall]} modules, {photons_per_wall[wall]} photons (average: {avg_photons:.1f} ph/module)")
    
    # ALL modules with signal (not just top 10)
    print(f"\n🏆 All modules with signal ({n_modules_with_signal} modules):")
    for i, m in enumerate(modules_with_signal, 1):
        d = np.linalg.norm(m.position - generation_point)
        pos = m.position
        
        # Determine wall
        wall = "?"
        if abs(pos[0] - detector.detector_length/2) < tolerance:
            wall = "X+"
        elif abs(pos[0] + detector.detector_length/2) < tolerance:
            wall = "X-"
        elif abs(pos[2] - detector.detector_depth/2) < tolerance:
            wall = "Z+"
        elif abs(pos[2] + detector.detector_depth/2) < tolerance:
            wall = "Z-"
        
        print(f"   {i:2d}. ID={m.module_id:3d} | {wall} wall | Pos=({pos[0]:5.1f}, {pos[1]:5.1f}, {pos[2]:6.1f})m | d={d:5.2f}m | {m.n_photons_detected:4d} ph")
    
    # ========== IMPROVED ANALYSIS OF ADJACENT MODULES WITHOUT SIGNAL ==========
    if n_modules_with_signal > 0:
        print(f"\n🔍 COMPLETE analysis of adjacent modules without signal:")
        print(f"   (Analyzing ALL modules with signal and their neighbors)")
        
        # Analyze adjacent modules for EACH module with signal
        for idx, signal_module in enumerate(modules_with_signal[:5], 1):  # Analyze top 5 with signal
            sig_pos = signal_module.position
            sig_wall = "?"
            
            if abs(sig_pos[0] - detector.detector_length/2) < tolerance:
                sig_wall = "X+"
            elif abs(sig_pos[0] + detector.detector_length/2) < tolerance:
                sig_wall = "X-"
            elif abs(sig_pos[2] - detector.detector_depth/2) < tolerance:
                sig_wall = "Z+"
            elif abs(sig_pos[2] + detector.detector_depth/2) < tolerance:
                sig_wall = "Z-"
            
            print(f"\n   {'='*70}")
            print(f"   Module #{idx} with signal:")
            print(f"   • ID={signal_module.module_id}, {sig_wall} wall, Pos=({sig_pos[0]:.1f}, {sig_pos[1]:.1f}, {sig_pos[2]:.1f})m")
            print(f"   • Distance to point: {np.linalg.norm(sig_pos - generation_point):.2f}m")
            print(f"   • Photons detected: {signal_module.n_photons_detected}")
            
            # Search for adjacent modules
            adjacent_modules = []
            for m in detector.modules:
                if m.module_id == signal_module.module_id:
                    continue
                
                # Same perpendicular axis to wall (same wall)
                if abs(m.position[0] - sig_pos[0]) < tolerance or \
                   abs(m.position[2] - sig_pos[2]) < tolerance:
                    # Calculate 2D distance in wall plane
                    if sig_wall in ['X+', 'X-']:
                        dist_in_plane = np.sqrt((m.position[1] - sig_pos[1])**2 + 
                                               (m.position[2] - sig_pos[2])**2)
                    else:  # Z+ or Z-
                        dist_in_plane = np.sqrt((m.position[0] - sig_pos[0])**2 + 
                                               (m.position[1] - sig_pos[1])**2)
                    
                    if 0 < dist_in_plane < 3*max(space_y, space_z):  # Close neighbors
                        dist_to_gen = np.linalg.norm(m.position - generation_point)
                        adjacent_modules.append({
                            'module': m,
                            'dist_in_plane': dist_in_plane,
                            'dist_to_gen': dist_to_gen
                        })
            
            # Sort by in-plane distance
            adjacent_modules.sort(key=lambda x: x['dist_in_plane'])
            
            # Count how many have and don't have signal
            n_adjacent_with_signal = sum(1 for adj in adjacent_modules if adj['module'].n_photons_detected > 0)
            n_adjacent_without_signal = len(adjacent_modules) - n_adjacent_with_signal
            
            print(f"\n   Adjacent modules found: {len(adjacent_modules)}")
            print(f"   • WITH signal: {n_adjacent_with_signal}")
            print(f"   • WITHOUT signal: {n_adjacent_without_signal}")
            
            if n_adjacent_without_signal > 0:
                print(f"\n   Adjacent modules WITHOUT signal (sorted by proximity):")
                
                without_signal_count = 0
                for adj in adjacent_modules:
                    m = adj['module']
                    if m.n_photons_detected == 0:
                        without_signal_count += 1
                        
                        print(f"\n   {without_signal_count}. ID={m.module_id:3d} | ✗ NO signal")
                        print(f"      Pos=({m.position[0]:5.1f}, {m.position[1]:5.1f}, {m.position[2]:6.1f})m")
                        print(f"      Dist. to module with signal: {adj['dist_in_plane']:.2f}m (in plane)")
                        print(f"      Dist. to gen. point: {adj['dist_to_gen']:.2f}m")
                        
                        # Detailed diagnosis
                        if adj['dist_to_gen'] > max_distance_light_yield:
                            print(f"      ⚠️  CAUSE: Distance > max_distance ({max_distance_light_yield:.1f}m)")
                            print(f"          → Light yield not calculated for this module")
                        else:
                            # Calculate approximate solid angle
                            solid_angle = (detector.module_width * detector.module_height) / (adj['dist_to_gen']**2)
                            print(f"      ⚠️  PROBABLE CAUSE: Very small solid angle")
                            print(f"          → Solid angle ≈ {solid_angle:.6f} sr")
                            print(f"          → Few photons generated towards this module")
                            print(f"          → Photons didn't arrive or were absorbed/reflected")
            else:
                print(f"\n   ✅ ALL adjacent modules have signal")
        
        # Global adjacency statistics
        print(f"\n   {'='*70}")
        print(f"\n   📊 Global statistics of modules without signal:")
        
        n_without_signal = len(detector.modules) - n_modules_with_signal
        
        if n_without_signal == 0:
            print(f"   • ✅ ALL modules ({len(detector.modules)}) have signal!")
            print(f"   • ⚠️ This is unusual. Probably max_distance is set too high.")
        else:
            n_too_far = sum(1 for m in detector.modules 
                           if m.n_photons_detected == 0 and 
                           np.linalg.norm(m.position - generation_point) > max_distance_light_yield)
            n_within_range = n_without_signal - n_too_far
            
            print(f"   • Total without signal: {n_without_signal}")
            print(f"   • Outside max_distance: {n_too_far} ({100*n_too_far/n_without_signal:.1f}%)")
            print(f"   • Within max_distance but no signal: {n_within_range} ({100*n_within_range/n_without_signal:.1f}%)")
    
    # ========== CALCULATE SUBPLOT LAYOUT ==========
    ncols = 5
    nrows = int(np.ceil(n_modules_to_plot / ncols))
    
    print(f"\n📈 Generating visualization:")
    print(f"   • Layout: {nrows} rows × {ncols} columns = {n_modules_to_plot} modules")
    
    # Optimized spacing - DYNAMICALLY ADJUSTED
    # Maximum allowed spacing is 1/(nrows-1) according to Plotly
    max_vertical_spacing = 1.0 / (nrows - 1) if nrows > 1 else 0.0
    max_horizontal_spacing = 1.0 / (ncols - 1) if ncols > 1 else 0.0
    
    # Use the minimum between desired and maximum allowed
    vertical_spacing = min(0.03, max_vertical_spacing * 0.8) if nrows > 1 else 0.0
    horizontal_spacing = min(0.03, max_horizontal_spacing * 0.8) if ncols > 1 else 0.0
    
    print(f"   • Vertical spacing: {vertical_spacing:.4f}")
    print(f"   • Horizontal spacing: {horizontal_spacing:.4f}")
    
    # ========== CREATE FIGURE WITH DYNAMIC SUBPLOTS ==========
    subplot_titles = [''] * n_modules_to_plot
    
    fig = make_subplots(
        rows=nrows, 
        cols=ncols,
        subplot_titles=subplot_titles,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing
    )
    
    for idx, module in enumerate(modules_to_plot):
        row = (idx // ncols) + 1
        col = (idx % ncols) + 1
        
        # Get current module data (kaon+muon)
        kaon_muon_signal = module.waveform
        kaon_muon_histogram = module.temporal_histogram[1]
        time_axis = module.time_axis
        
        # Get kaon-only data for this module
        if module.module_id in kaon_only_results:
            kaon_only_signal = kaon_only_results[module.module_id]['waveform']
            kaon_only_histogram = kaon_only_results[module.module_id]['histogram']
        else:
            kaon_only_signal = np.zeros_like(kaon_muon_signal)
            kaon_only_histogram = np.zeros_like(kaon_muon_histogram)
        
        # Get module information - CORRECT DISTANCE
        distance = np.linalg.norm(module.position - generation_point)
        pos = module.position
        
        # Determine which wall the module is on
        wall = "?"
        if abs(pos[0] - detector.detector_length/2) < tolerance:
            wall = "X+"
        elif abs(pos[0] + detector.detector_length/2) < tolerance:
            wall = "X-"
        elif abs(pos[1] - detector.detector_height/2) < tolerance:
            wall = "Y+"
        elif abs(pos[1] + detector.detector_height/2) < tolerance:
            wall = "Y-"
        elif abs(pos[2] - detector.detector_depth/2) < tolerance:
            wall = "Z+"
        elif abs(pos[2] + detector.detector_depth/2) < tolerance:
            wall = "Z-"
        
        # Create title BEFORE calling plot_module_comparison
        title_text = f"<b>ID: {module.module_id}</b> | ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m | d={distance:.2f}m | {module.n_photons_detected} ph"
        
        # Plot comparison in corresponding subplot
        plot_module_comparison(
            module, kaon_only_signal, kaon_only_histogram,
            kaon_muon_signal, kaon_muon_histogram,
            time_axis, row, col, fig, title_text, ncols
        )
        
        # Add annotation using PAPER coordinates - CORRECTED
        subplot_height = (1.0 - (nrows - 1) * vertical_spacing) / nrows
        subplot_width = (1.0 - (ncols - 1) * horizontal_spacing) / ncols
        
        x_paper = (col - 1) * (subplot_width + horizontal_spacing) + subplot_width / 2
        y_paper = 1.0 - ((row - 1) * (subplot_height + vertical_spacing))
        
        # Adjust offset: larger for first row (to avoid collision with main title)
        y_offset = 0.015 if row == 1 else 0.008
        
        fig.add_annotation(
            text=title_text,
            xref="paper",
            yref="paper",
            x=x_paper,
            y=y_paper + y_offset,
            xanchor='center',
            yanchor='bottom',
            showarrow=False,
            font=dict(size=8)
        )
    
    # ========== CONFIGURE LAYOUT ==========
    for i in range(1, nrows + 1):
        for j in range(1, ncols + 1):
            if i == nrows:
                fig.update_xaxes(
                    title_text="Time (ns)", 
                    row=i, col=j, 
                    title_font=dict(size=9),
                    range=[0, 500]  # <-- CAMBIAR: de 100 a 500 ns
                )
            else:
                fig.update_xaxes(
                    title_text="", 
                    row=i, col=j,
                    range=[0, 500]  # <-- CAMBIAR: de 100 a 500 ns
                )
            
            if j == 1:
                fig.update_yaxes(
                    title_text="Amp", 
                    row=i, col=j, 
                    title_font=dict(size=9),
                    # range=[ymin, ymax]  # Opcional: descomentar para fijar eje Y
                )
            else:
                fig.update_yaxes(
                    title_text="", 
                    row=i, col=j,
                    # range=[ymin, ymax]  # Opcional: descomentar para fijar eje Y
                )
    
    plot_height = max(225 * nrows, 500)
    
    # ========== IMPROVED TITLE WITH ALL DISTANCE INFO ==========
    # Calculate minimum distances to each wall from generation point
    closest_wall_name = "?"
    closest_wall_dist = min_dist_to_walls
    
    if abs(dist_to_x_plus - min_dist_to_walls) < 0.01:
        closest_wall_name = "X+"
    elif abs(dist_to_x_minus - min_dist_to_walls) < 0.01:
        closest_wall_name = "X-"
    elif abs(dist_to_y_plus - min_dist_to_walls) < 0.01:
        closest_wall_name = "Y+"
    elif abs(dist_to_y_minus - min_dist_to_walls) < 0.01:
        closest_wall_name = "Y-"
    elif abs(dist_to_z_plus - min_dist_to_walls) < 0.01:
        closest_wall_name = "Z+"
    elif abs(dist_to_z_minus - min_dist_to_walls) < 0.01:
        closest_wall_name = "Z-"
    
    # Calculate distances for modules
    all_distances = [np.linalg.norm(m.position - generation_point) for m in detector.modules]
    min_mod_dist = min(all_distances)
    
    if n_modules_with_signal > 0:
        distances_with_signal = [np.linalg.norm(m.position - generation_point) 
                                for m in modules_with_signal]
        max_signal_dist = max(distances_with_signal)
    else:
        max_signal_dist = 0.0
    
    fig.update_layout(
        title_text=(
            f"<b>FROST Detector - {n_modules_to_plot} modules ({n_modules_with_signal} with signal)</b><br>"
            f"<sub>Detector: {detector.detector_length:.0f}×{detector.detector_height:.0f}×{detector.detector_depth:.0f}m | "
            f"Modules: {detector.module_width*100:.0f}×{detector.module_height*100:.0f}cm | "
            f"Efficiency: {detector.frost_efficiency*100:.0f}% | "
            f"max_dist: {max_distance_light_yield:.1f}m</sub><br>"
            f"<sub>K+ decay: {k_decay_time}ns | "
            f"Gen. point: ({generation_point[0]:.1f}, {generation_point[1]:.1f}, {generation_point[2]:.1f})m → "
            f"{closest_wall_name} wall ({closest_wall_dist:.2f}m) | "
            f"Closest module: {min_mod_dist:.2f}m | "
            f"Signal up to: {max_signal_dist:.2f}m</sub>"
        ),
        title_font=dict(size=11),
        template='plotly_white',
        height=plot_height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9)
        ),
        margin=dict(l=40, r=20, t=200, b=40)
    )
    
    # Smaller subplot titles
    for annotation in fig.layout.annotations:
        annotation.font.size = 8
    
    # ========== SAVE FIRST PLOT (ALL MODULES) ==========
    fig.write_html('frost_detector_all_modules.html')
    print(f"   • File saved: 'frost_detector_all_modules.html'")
    
    # ========== CREATE SECOND PLOT ONLY WITH MODULES WITH SIGNAL ==========
    print(f"\n📈 Generating visualization of modules with signal only:")
    
    modules_with_signal_to_plot = modules_with_signal
    n_signal_modules = len(modules_with_signal_to_plot)
    
    if n_signal_modules > 0:
        ncols_signal = 5
        nrows_signal = int(np.ceil(n_signal_modules / ncols_signal))
        
        print(f"   • Layout: {nrows_signal} rows × {ncols_signal} columns = {n_signal_modules} modules with signal")
        
        # Calcular espaciado dinámicamente también para el segundo plot
        max_v_spacing = 1.0 / (nrows_signal - 1) if nrows_signal > 1 else 0.0
        max_h_spacing = 1.0 / (ncols_signal - 1) if ncols_signal > 1 else 0.0
        
        v_spacing = min(0.03, max_v_spacing * 0.8) if nrows_signal > 1 else 0.0
        h_spacing = min(0.03, max_h_spacing * 0.8) if ncols_signal > 1 else 0.0
        
        subplot_titles_signal = [''] * n_signal_modules
        
        fig_signal = make_subplots(
            rows=nrows_signal, 
            cols=ncols_signal,
            subplot_titles=subplot_titles_signal,
            vertical_spacing=v_spacing,
            horizontal_spacing=h_spacing
        )
        
        for idx, module in enumerate(modules_with_signal_to_plot):
            row = (idx // ncols_signal) + 1
            col = (idx % ncols_signal) + 1
            
            # Obtener datos del módulo actual (kaón+muón)
            kaon_muon_signal = module.waveform
            kaon_muon_histogram = module.temporal_histogram[1]
            time_axis = module.time_axis
            
            # Obtener datos del kaón solo para este módulo
            if module.module_id in kaon_only_results:
                kaon_only_signal = kaon_only_results[module.module_id]['waveform']
                kaon_only_histogram = kaon_only_results[module.module_id]['histogram']
            else:
                kaon_only_signal = np.zeros_like(kaon_muon_signal)
                kaon_only_histogram = np.zeros_like(kaon_muon_histogram)
            
            # Obtener información del módulo
            distance = np.linalg.norm(module.position - generation_point)
            pos = module.position
            
            # Determinar en qué pared está el módulo
            wall = "?"
            if abs(pos[0] - detector.detector_length/2) < tolerance:
                wall = "X+"
            elif abs(pos[0] + detector.detector_length/2) < tolerance:
                wall = "X-"
            elif abs(pos[1] - detector.detector_height/2) < tolerance:
                wall = "Y+"
            elif abs(pos[1] + detector.detector_height/2) < tolerance:
                wall = "Y-"
            elif abs(pos[2] - detector.detector_depth/2) < tolerance:
                wall = "Z+"
            elif abs(pos[2] + detector.detector_depth/2) < tolerance:
                wall = "Z-"
            
            # Título con module_id, coordenadas y distancia
            title_text = f"<b>ID: {module.module_id}</b> | ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m | d={distance:.2f}m | {module.n_photons_detected} ph"
            
            plot_module_comparison(
                module, kaon_only_signal, kaon_only_histogram,
                kaon_muon_signal, kaon_muon_histogram,
                time_axis, row, col, fig_signal, title_text, ncols_signal
            )
            
            # Añadir anotación - CORREGIDO
            v_spacing = 0.04 if nrows_signal > 1 else 0.0
            h_spacing = 0.03 if ncols_signal > 1 else 0.0
            
            subplot_height = (1.0 - (nrows_signal - 1) * v_spacing) / nrows_signal
            subplot_width = (1.0 - (ncols_signal - 1) * h_spacing) / ncols_signal
            
            x_paper = (col - 1) * (subplot_width + h_spacing) + subplot_width / 2
            y_paper = 1.0 - ((row - 1) * (subplot_height + v_spacing))
            
            # Mayor offset para primera fila
            y_offset = 0.015 if row == 1 else 0.008
            
            fig_signal.add_annotation(
                text=title_text,
                xref="paper",
                yref="paper",
                x=x_paper,
                y=y_paper + y_offset,
                xanchor='center',
                yanchor='bottom',
                showarrow=False,
                font=dict(size=8)
            )
        
        # Configurar layout del segundo plot
        for i in range(1, nrows_signal + 1):
            for j in range(1, ncols_signal + 1):
                if i == nrows_signal:
                    fig_signal.update_xaxes(
                        title_text="Time (ns)", 
                        row=i, col=j, 
                        title_font=dict(size=9),
                        range=[0, 500]  # <-- CAMBIAR: de 100 a 500 ns
                    )
                else:
                    fig_signal.update_xaxes(
                        title_text="", 
                        row=i, col=j,
                        range=[0, 500]  # <-- CAMBIAR: de 100 a 500 ns
                    )
                
                if j == 1:
                    fig_signal.update_yaxes(
                        title_text="Amp", 
                        row=i, col=j, 
                        title_font=dict(size=9),
                        # range=[ymin, ymax]  # Opcional: descomentar para fijar eje Y
                    )
                else:
                    fig_signal.update_yaxes(
                        title_text="", 
                        row=i, col=j,
                        # range=[ymin, ymax]  # Opcional: descomentar para fijar eje Y
                    )

        plot_height_signal = max(220 * nrows_signal, 500)
        
        fig_signal.update_layout(
            title_text=(
                f"<b>FROST Detector - {n_signal_modules} modules WITH SIGNAL out of {len(detector.modules)} total modules</b><br>"
                f"<sub>Detector: {detector.detector_length:.0f}×{detector.detector_height:.0f}×{detector.detector_depth:.0f}m | "
                f"Modules: {detector.module_width*100:.0f}×{detector.module_height*100:.0f}cm | "
                f"Efficiency: {detector.frost_efficiency*100:.0f}% | "
                f"max_dist: {max_distance_light_yield:.1f}m</sub><br>"
                f"<sub>K+ decay: {k_decay_time}ns | "
                f"Gen. point: ({generation_point[0]:.1f}, {generation_point[1]:.1f}, {generation_point[2]:.1f})m → "
                f"{closest_wall_name} wall ({closest_wall_dist:.2f}m) | "
                f"Closest module: {min_mod_dist:.2f}m | "
                f"Signal up to: {max_signal_dist:.2f}m</sub>"
            ),
            title_font=dict(size=11),
            template='plotly_white',
            height=plot_height_signal,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=9)
            ),
            margin=dict(l=40, r=20, t=200, b=40)  
        )
        
        for annotation in fig_signal.layout.annotations:
            annotation.font.size = 8
        
        fig_signal.write_html('frost_detector_with_signal.html')
        print(f"   • File saved: 'frost_detector_with_signal.html'")
    
    # ========== CREATE THIRD PLOT: SUMMED SIGNALS ==========
    min_photons_threshold = 40
    max_distance_threshold = 3.5  # meters - Maximum distance to include in sum
    
    # Filter modules by photon count AND distance
    modules_with_enough_signal = [
        m for m in modules_with_signal 
        if m.n_photons_detected > min_photons_threshold and 
           np.linalg.norm(m.position - generation_point) <= max_distance_threshold
    ]
    n_modules_above_threshold = len(modules_with_enough_signal)
    
    if n_modules_above_threshold > 0:
        print(f"\n📈 Generating summed signals plot:")
        print(f"   • Modules with signal: {n_modules_with_signal}")
        print(f"   • Modules with >{min_photons_threshold} photons: {sum(1 for m in modules_with_signal if m.n_photons_detected > min_photons_threshold)}")
        print(f"   • Modules within {max_distance_threshold}m: {sum(1 for m in modules_with_signal if np.linalg.norm(m.position - generation_point) <= max_distance_threshold)}")
        print(f"   • Modules meeting BOTH criteria: {n_modules_above_threshold}")
        print(f"   • Summing {n_modules_above_threshold} modules")
        
        # CORRECCIÓN: Obtener el primer módulo para saber las dimensiones
        first_module = modules_with_enough_signal[0]
        time_axis_sum = first_module.time_axis.copy()
        
        # Inicializar arrays para las sumas
        summed_waveform_kaon_muon = np.zeros_like(time_axis_sum, dtype=float)
        summed_waveform_kaon_only = np.zeros_like(time_axis_sum, dtype=float)
        summed_histogram_kaon_muon = np.zeros_like(first_module.temporal_histogram[1], dtype=float)
        summed_histogram_kaon_only = np.zeros_like(first_module.temporal_histogram[1], dtype=float)
        
        # Sumar las señales preservando el bineado original de cada canal
        for m in modules_with_enough_signal:
            # Sumar waveform K+μ (desde detector_kaon_muon)
            if m.waveform is not None:
                summed_waveform_kaon_muon += m.waveform
            
            # Sumar histograma K+μ
            if m.temporal_histogram is not None:
                summed_histogram_kaon_muon += m.temporal_histogram[1]
            
            # Sumar waveform K only (desde kaon_only_results)
            if m.module_id in kaon_only_results:
                if kaon_only_results[m.module_id]['waveform'] is not None:
                    summed_waveform_kaon_only += kaon_only_results[m.module_id]['waveform']
                
                # Sumar histograma K only
                if kaon_only_results[m.module_id]['histogram'] is not None:
                    summed_histogram_kaon_only += kaon_only_results[m.module_id]['histogram']
        
        # Crear figura para señales sumadas
        fig_summed = make_subplots(
            rows=1, 
            cols=1,
            subplot_titles=[f'Summed signals from modules with >{min_photons_threshold} photons']
        )
        
        # Añadir histograma sumado K+μ (barras azul claro)
        fig_summed.add_trace(go.Bar(
            x=time_axis_sum,
            y=summed_histogram_kaon_muon,
            name='Summed photon histogram (K+μ)',
            marker_color='lightblue',
            opacity=0.6
        ))
        
        # Añadir señal sumada kaón+muón (línea roja)
        fig_summed.add_trace(go.Scatter(
            x=time_axis_sum,
            y=summed_waveform_kaon_muon,
            mode='lines',
            name='Summed K+μ signal',
            line=dict(color='red', width=3)
        ))
        
        # Añadir señal sumada kaón only (línea azul)
        fig_summed.add_trace(go.Scatter(
            x=time_axis_sum,
            y=summed_waveform_kaon_only,
            mode='lines',
            name='Summed K only signal',
            line=dict(color='blue', width=3)
        ))
        
        # Configuración del plot sumado
        fig_summed.update_xaxes(
            title_text="Time (ns)",
            range=[0, 500]  # <-- CAMBIAR: de 100 a 500 ns
        )
        fig_summed.update_yaxes(
            title_text="Amplitude (summed)",
            # range=[ymin, ymax]  # Opcional: descomentar para fijar eje Y
        )

        # Calcular estadísticas de las señales sumadas
        max_amplitude_km = np.max(summed_waveform_kaon_muon)
        max_amplitude_k = np.max(summed_waveform_kaon_only)
        total_photons_sum = sum(m.n_photons_detected for m in modules_with_enough_signal)
        
        fig_summed.update_layout(
            title_text=(
                f"<b>FROST Detector - Summed Signals ({n_modules_above_threshold} modules)</b><br>"
                f"<sub>Filters: >{min_photons_threshold} photons AND distance ≤{max_distance_threshold}m | "
                f"Total photons: {total_photons_sum} | "
                f"Max amplitude K+μ: {max_amplitude_km:.1f} | "
                f"Max amplitude K: {max_amplitude_k:.1f}</sub><br>"
                f"<sub>Detector: {detector.detector_length:.0f}×{detector.detector_height:.0f}×{detector.detector_depth:.0f}m | "
                f"K+ decay: {k_decay_time}ns | "
                f"Gen. point: ({generation_point[0]:.1f}, {generation_point[1]:.1f}, {generation_point[2]:.1f})m</sub>"
            ),
            title_font=dict(size=13),
            template='plotly_white',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="right",
                x=0.98,
                font=dict(size=11)
            ),
            margin=dict(l=60, r=40, t=140, b=60)
        )
        
        # Guardar plot sumado
        fig_summed.write_html('frost_detector_summed_signals.html')
        print(f"   • File saved: 'frost_detector_summed_signals.html'")
        
        # Mostrar estadísticas en terminal
        print(f"\n📊 Estadísticas de señales sumadas:")
        print(f"   • Filtros aplicados:")
        print(f"     - Fotones detectados > {min_photons_threshold}")
        print(f"     - Distancia al punto de generación ≤ {max_distance_threshold} m")
        print(f"   • Módulos que cumplen ambos criterios: {n_modules_above_threshold}")
        print(f"   • Total de fotones sumados: {total_photons_sum}")
        print(f"   • Amplitud máxima K+μ: {max_amplitude_km:.2f}")
        print(f"   • Amplitud máxima K only: {max_amplitude_k:.2f}")
        print(f"   • Ratio K+μ/K: {max_amplitude_km/max_amplitude_k:.3f}" if max_amplitude_k > 0 else "   • Ratio K+μ/K: N/A")
        
        # Listar módulos usados en la suma
        print(f"\n   Módulos incluidos en la suma:")
        for i, m in enumerate(modules_with_enough_signal[:10], 1):  # Mostrar primeros 10
            dist = np.linalg.norm(m.position - generation_point)
            print(f"   {i:2d}. ID={m.module_id:3d} | {m.n_photons_detected:4d} fotones | "
                  f"Distancia: {dist:.2f}m")
        if n_modules_above_threshold > 10:
            print(f"   ... y {n_modules_above_threshold - 10} módulos más")
        
        # Mostrar el plot sumado
        fig_summed.show()
    else:
        print(f"\n⚠️ No hay módulos que cumplan los criterios:")
        print(f"   - Fotones > {min_photons_threshold}")
        print(f"   - Distancia ≤ {max_distance_threshold} m")
        print(f"   No se puede crear el plot de señales sumadas.")
    
    # Mostrar el plot de módulos con señal (si existe)
    if n_signal_modules > 0:
        fig_signal.show()
    
    # Mostrar el primer plot (todos los módulos)
    fig.show()
    
    print("\n✅ Simulación completada exitosamente!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
