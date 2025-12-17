import numpy as np
from typing import List, Tuple, Optional
from FROST_module import FROSTModule
from geometry_utils import calculate_photon_yield_for_module

class DetectorSimulation:
    def __init__(self, detector_dimensions, module_dimensions):
        """
        Inicializa el detector FROST.
        
        Args:
            detector_dimensions: (length_x, height_y, depth_z) en metros
                - length_x: dimensión horizontal en X (12m)
                - height_y: dimensión vertical ALTURA (12m)
                - depth_z: dimensión horizontal en Z, eje del haz (55m)
            module_dimensions: (width, height) del módulo FROST en metros
        """
        self.detector_length = detector_dimensions[0]  # X - horizontal (12m)
        self.detector_height = detector_dimensions[1]  # Y - ALTURA vertical (12m)
        self.detector_depth = detector_dimensions[2]   # Z - eje del haz (55m)
        
        self.module_width = module_dimensions[0]
        self.module_height = module_dimensions[1]
        
        self.modules: List[FROSTModule] = []
        
        # Parámetros físicos 
        self.n_photons_per_mev = 25000
        self.frost_efficiency = 0.05
        
        # Parámetros temporales LAr
        self.tau_fast = 0.01      # ns
        self.tau_slow = 3000.0    # ns
        self.fast_fraction = 0.23
        
        # Parámetros WLS
        self.tau_wls = 1.2       # ns
        self.tau_ptp = 1.45      # ns
        
        # Parámetros propagación LAr
        self.n_lar = 1.23
        self.rayleigh_length = 8.0  # m
        self.rayleigh_power = 1.5
        self.c_light = 0.3          # m/ns
        
    def create_uniform_module_distribution(self,
                                          n_x=None,
                                          n_y=None,
                                          n_z=None):
        """
        Crea una distribución uniforme de módulos FROST en las 4 paredes laterales.
        
        Solo se crean módulos en las paredes ±X y ±Z (las 4 paredes laterales verticales).
        NO se crean módulos en las paredes ±Y (arriba/abajo).
        
        Args:
            n_x: Número de módulos a lo largo del eje X (12m)
            n_y: Número de módulos a lo largo del eje Y (12m altura)
            n_z: Número de módulos a lo largo del eje Z (55m, eje del haz)
        """
        module_id = 0
        
        # Valores por defecto
        if n_x is None:
            n_x = int(self.detector_length / self.module_width)
        if n_y is None:
            n_y = int(self.detector_height / self.module_height)
        if n_z is None:
            n_z = int(self.detector_depth / self.module_width)
        
        # Calcular espaciados
        space_x = self.detector_length / n_x if n_x > 0 else 0
        space_y = self.detector_height / n_y if n_y > 0 else 0
        space_z = self.detector_depth / n_z if n_z > 0 else 0
        
        print(f"\n📐 CONFIGURACIÓN DEL DETECTOR:")
        print(f"   Dimensiones: X={self.detector_length}m, Y={self.detector_height}m (altura), Z={self.detector_depth}m (eje del haz)")
        print(f"\n📊 Módulos por eje: n_x={n_x}, n_y={n_y}, n_z={n_z}")
        print(f"   Espaciado: X={space_x:.3f}m, Y={space_y:.3f}m, Z={space_z:.3f}m")
        
        # Calcular el rango total que ocupan los módulos (centrados)
        total_range_x = (n_x - 1) * space_x if n_x > 1 else 0
        total_range_y = (n_y - 1) * space_y if n_y > 1 else 0
        total_range_z = (n_z - 1) * space_z if n_z > 1 else 0
        
        print(f"   Rango ocupado (centrado): X={total_range_x:.3f}m, Y={total_range_y:.3f}m, Z={total_range_z:.3f}m")
        
        # ====== PARED X+ (DERECHA) ======
        if n_y > 0 and n_z > 0:
            x_wall = self.detector_length / 2.0
            normal = np.array([-1.0, 0.0, 0.0])
            
            for iy in range(n_y):
                for iz in range(n_z):
                    # Los módulos están centrados en la pared
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
        
        # ====== PARED X- (IZQUIERDA) ======
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
        
        # ====== PARED Z+ (TRASERA) ======
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
        
        # ====== PARED Z- (FRONTAL) ======
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
        
        # NO SE CREAN paredes Y+ ni Y- (techo/suelo)
        
        total_modules = len(self.modules)
        if total_modules == 0:
            print("⚠️ WARNING: No se crearon módulos.")
        else:
            print(f"\n✅ Creados {total_modules} módulos FROST")
            print(f"   Paredes X± (laterales): {2 * n_y * n_z if n_y > 0 and n_z > 0 else 0} módulos")
            print(f"   Paredes Z± (laterales): {2 * n_x * n_y if n_x > 0 and n_y > 0 else 0} módulos")
            print(f"   Paredes Y± (arriba/abajo): 0 módulos (no se crean)")

    def calculate_light_yield_for_nearby_modules(self,
                                                 generation_point: np.ndarray,
                                                 kinetic_energy: float,
                                                 max_distance: float = 6.0):
        """
        Calcula el light yield solo para módulos cercanos (dentro de max_distance).
        
        Args:
            generation_point: np.array([x, y, z]) punto de generación
            kinetic_energy: energía depositada (MeV)
            max_distance: distancia máxima para considerar un módulo (m)
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
                # Módulo muy lejos: cero fotones
                module.set_photon_yield(0.0)
        
        print(f"✅ Calculated light yield for {n_modules_considered}/{len(self.modules)} modules "
              f"within {max_distance} m")
        
    def propagate_photons_temporally(self, 
                                     generation_time: float = 0.0,
                                     generation_point: np.ndarray = np.array([0., 0., 0.]),
                                     use_wls_smearing: bool = False,
                                     wls_histogram_file: str = None):
        """
        Calcula la distribución temporal de fotones para cada módulo.
        
        Dos modos de operación:
        
        1) use_wls_smearing=False (modo analítico):
           - Centelleo rápido/lento en LAr (tau_fast, tau_slow)
           - Propagación en LAr con dispersión Rayleigh
           - Conversión en pTP (tau_ptp)
           - Conversión en WLS (tau_wls)
        
        2) use_wls_smearing=True (modo Geant4):
           - Centelleo rápido/lento en LAr (tau_fast, tau_slow)
           - Propagación en LAr con dispersión Rayleigh
           - Smearing por propagación en WLS plate desde histograma Geant4
             (el histograma YA incluye los efectos de pTP + WLS, no se añaden)
        
        Args:
            generation_time: tiempo inicial de generación (ns)
            generation_point: punto de generación (para calcular propagación en LAr)
            use_wls_smearing: si True, usa histograma Geant4 (que ya incluye pTP+WLS)
            wls_histogram_file: path al archivo del histograma temporal de WLS
        """
        # Si se solicita WLS smearing, cargar el histograma
        cdf_wls = None
        bin_edges_wls = None
        
        if use_wls_smearing and wls_histogram_file is not None:
            cdf_wls, bin_edges_wls = self._load_wls_histogram(wls_histogram_file)
            print("⚠️ Using Geant4 WLS histogram: pTP and WLS exponential times are DISABLED")
        
        for module in self.modules:
            if module.n_photons_detected == 0:
                module.set_temporal_distribution(np.array([]))
                continue
            
            # Distancia de propagación en LAr
            distance = module.get_distance_to_point(generation_point)
            
            # Generar tiempos de llegada para cada fotón
            photon_times = []
            
            # Número de fotones rápidos y lentos
            n_fast = int(module.n_photons_detected * self.fast_fraction)
            n_slow = module.n_photons_detected - n_fast
            
            # -------- FOTONES DE CENTELLEO LENTO --------
            for _ in range(n_slow):
                t = generation_time
                
                # 1. Tiempo de emisión lenta del LAr
                t += np.random.exponential(self.tau_slow)
                
                # 2. Propagación en LAr (dispersión Rayleigh)
                delta_t_rayleigh = (distance ** self.rayleigh_power) * self.n_lar / (self.c_light * self.rayleigh_length)
                t += np.random.normal(loc=0, scale=delta_t_rayleigh)
                
                # 3. Efectos en FROST (dos opciones mutuamente excluyentes)
                if use_wls_smearing and cdf_wls is not None:
                    # OPCIÓN A: Usar histograma Geant4 (ya incluye pTP + WLS)
                    t += self._sample_wls_time(cdf_wls, bin_edges_wls)
                else:
                    # OPCIÓN B: Modelo analítico (pTP + WLS exponenciales)
                    t += np.random.exponential(self.tau_ptp)  # Conversión en pTP
                    t += np.random.exponential(self.tau_wls)  # Conversión en WLS
                
                photon_times.append(t)
            
            # -------- FOTONES DE CENTELLEO RÁPIDO --------
            for _ in range(n_fast):
                t = generation_time
                
                # 1. Tiempo de emisión rápida del LAr
                t += np.random.exponential(self.tau_fast)
                
                # 2. Propagación en LAr (dispersión Rayleigh)
                delta_t_rayleigh = (distance ** self.rayleigh_power) * self.n_lar / (self.c_light * self.rayleigh_length)
                t += np.random.normal(loc=0, scale=delta_t_rayleigh)
                
                # 3. Efectos en FROST (dos opciones mutuamente excluyentes)
                if use_wls_smearing and cdf_wls is not None:
                    # OPCIÓN A: Usar histograma Geant4 (ya incluye pTP + WLS)
                    t += self._sample_wls_time(cdf_wls, bin_edges_wls)
                else:
                    # OPCIÓN B: Modelo analítico (pTP + WLS exponenciales)
                    t += np.random.exponential(self.tau_ptp)  # Conversión en pTP
                    t += np.random.exponential(self.tau_wls)  # Conversión en WLS
                
                photon_times.append(t)
            
            module.set_temporal_distribution(np.array(photon_times))
    
    def _load_wls_histogram(self, filename: str):
        """
        Carga el histograma de propagación en WLS plate desde archivo.
        Compatible con el formato de FROST_simu.py.
        
        Returns:
            (cdf, bin_edges): tupla con CDF y bordes de bins para muestreo inverso
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
        
        # Crear histograma
        hist, bin_edges = np.histogram(values, bins=len(values), weights=counts)
        
        # Rebinning (mismo factor que FROST_simu.py)
        hist, bin_edges = self._rebin_histogram(hist, bin_edges, factor=20)
        
        # Normalizar
        bin_widths = np.diff(bin_edges)
        hist_normalized = hist / np.sum(hist * bin_widths)
        
        # Calcular CDF
        cdf = np.cumsum(hist_normalized * bin_widths)
        cdf = cdf / cdf[-1]  # asegurar que el último valor es 1
        
        return cdf, bin_edges
    
    def _rebin_histogram(self, hist, bin_edges, factor):
        """
        Reduce el número de bins combinando bins adyacentes.
        """
        if len(hist) % factor != 0:
            # Truncar para que sea divisible
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
        Muestrea un tiempo aleatorio de la distribución WLS usando inversión de CDF.
        """
        u = np.random.rand()
        delta_t = np.interp(u, cdf, bin_edges[1:])
        return delta_t

    def apply_electronics_to_all_modules(self,
                                        sipm_tau_rise: float = 1.5,
                                        sipm_tau_decay: float = 15.0,
                                        spe_amplitude: float = 10.0,
                                        adc_sampling_time: float = 1.0,
                                        time_range: Tuple[float, float] = (0, 100),
                                        baseline_noise_sigma: float = 0.0):
        """
        Aplica la respuesta electrónica a todos los módulos.
        
        Args:
            (ver FROSTModule.apply_electronics para documentación de parámetros)
        """
        for module in self.modules:
            # Primero crear histograma temporal
            n_bins = int((time_range[1] - time_range[0]) / adc_sampling_time)
            module.create_temporal_histogram(time_range, n_bins)
            
            # Luego aplicar electrónica
            module.apply_electronics(
                sipm_tau_rise=sipm_tau_rise,
                sipm_tau_decay=sipm_tau_decay,
                spe_amplitude=spe_amplitude,
                adc_sampling_time=adc_sampling_time,
                time_range=time_range,
                baseline_noise_sigma=baseline_noise_sigma
            )
            
    def sum_waveforms(self, 
                     module_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Suma las waveforms de módulos especificados.
        
        Args:
            module_indices: lista de índices de módulos a sumar. Si None, suma todos.
            
        Returns:
            (time_axis, summed_waveform): tupla con eje temporal y waveform sumada
        """
        if module_indices is None:
            modules_to_sum = self.modules
        else:
            modules_to_sum = [self.modules[i] for i in module_indices]
        
        # Verificar que todos tengan waveforms
        valid_modules = [m for m in modules_to_sum if m.waveform is not None]
        
        if len(valid_modules) == 0:
            print("⚠️ No modules with waveforms to sum")
            return None, None
        
        # Asumir que todos tienen el mismo eje temporal
        time_axis = valid_modules[0].time_axis
        summed_waveform = np.zeros_like(time_axis, dtype=float)
        
        for module in valid_modules:
            if len(module.waveform) == len(summed_waveform):
                summed_waveform += module.waveform
            else:
                print(f"⚠️ Module {module.module_id} has different waveform length, skipping")
        
        print(f"✅ Summed waveforms from {len(valid_modules)} modules")
        return time_axis, summed_waveform
    
    def get_modules_with_signal(self, min_photons: int = 1) -> List[FROSTModule]:
        """
        Devuelve lista de módulos que detectaron al menos min_photons fotones.
        
        Args:
            min_photons: umbral mínimo de fotones
            
        Returns:
            lista de FROSTModule
        """
        return [m for m in self.modules if m.n_photons_detected >= min_photons]
