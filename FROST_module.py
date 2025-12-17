import numpy as np
from typing import Optional, Tuple

class FROSTModule:
    """
    Clase que representa un módulo FROST individual en el detector.
    
    Contiene:
    - Posición y orientación del módulo
    - Dimensiones físicas
    - Número de fotones que le llegan
    - Distribución temporal de fotones
    - Waveform procesada por electrónica
    """
    
    def __init__(self, 
                 position: np.ndarray,
                 normal: np.ndarray,
                 width: float = 0.5,      # m
                 height: float = 0.5,     # m
                 efficiency: float = 1,
                 module_id: int = 0):
        """
        Constructor de un módulo FROST.
        
        Args:
            position: np.array([x, y, z]) - posición del centro del módulo (m)
            normal: np.array([nx, ny, nz]) - vector normal apuntando al interior del detector
            width: ancho del módulo (m)
            height: alto del módulo (m)
            efficiency: eficiencia de detección (0-1)
            module_id: identificador único del módulo
        """
        self.position = np.array(position)
        self.normal = np.array(normal) / np.linalg.norm(normal)  # normalizar
        self.width = width
        self.height = height
        self.efficiency = efficiency
        self.module_id = module_id
        
        # Atributos que se llenarán durante la simulación
        self.n_photons_arriving = 0.0        # fotones que llegan (antes de eficiencia)
        self.n_photons_detected = 0          # fotones detectados (después de eficiencia)
        self.photon_times = None             # array de tiempos de llegada de fotones
        self.temporal_histogram = None       # histograma temporal (bins, counts)
        self.waveform = None                 # señal procesada por electrónica
        self.time_axis = None                # eje temporal para la waveform
        
    def get_distance_to_point(self, point: np.ndarray) -> float:
        """
        Calcula la distancia del módulo a un punto.
        
        Args:
            point: np.array([x, y, z])
            
        Returns:
            distancia en metros
        """
        return np.linalg.norm(self.position - point)
    
    def set_photon_yield(self, n_photons: float):
        """
        Establece el número de fotones que llegan al módulo.
        
        Args:
            n_photons: número de fotones (puede ser float por cálculo de ángulo sólido)
        """
        self.n_photons_arriving = n_photons
        # Aplicar eficiencia de detección
        self.n_photons_detected = int(n_photons * self.efficiency)
        
    def set_temporal_distribution(self, photon_times: np.ndarray):
        """
        Establece la distribución temporal de fotones detectados.
        
        Args:
            photon_times: array de tiempos de llegada de fotones individuales (ns)
        """
        self.photon_times = photon_times
        
    def create_temporal_histogram(self, 
                                   time_range: Tuple[float, float] = (0, 100),
                                   n_bins: int = 100):
        """
        Crea un histograma temporal a partir de los tiempos de fotones.
        
        Args:
            time_range: (t_min, t_max) en ns
            n_bins: número de bins
        """
        if self.photon_times is None or len(self.photon_times) == 0:
            # Sin fotones: histograma vacío
            self.temporal_histogram = (
                np.linspace(time_range[0], time_range[1], n_bins + 1),  # bin edges
                np.zeros(n_bins)                                         # counts
            )
        else:
            counts, bin_edges = np.histogram(
                self.photon_times, 
                bins=n_bins, 
                range=time_range
            )
            self.temporal_histogram = (bin_edges, counts)
            
    def apply_electronics(self, 
                         sipm_tau_rise: float,
                         sipm_tau_decay: float,
                         spe_amplitude: float,
                         adc_sampling_time: float,
                         time_range: Tuple[float, float] = (0, 100),
                         baseline_noise_sigma: float = 0.0):
        """
        Aplica la respuesta electrónica del SiPM al histograma temporal.
        
        Args:
            sipm_tau_rise: tiempo de subida del SiPM (ns)
            sipm_tau_decay: tiempo de caída del SiPM (ns)
            spe_amplitude: amplitud del single photoelectron
            adc_sampling_time: periodo de muestreo del ADC (ns)
            time_range: (t_min, t_max) ventana temporal (ns)
            baseline_noise_sigma: desviación estándar del ruido de línea base
        """
        # Crear eje temporal
        t_min, t_max = time_range
        self.time_axis = np.arange(t_min, t_max, adc_sampling_time)
        
        # Si no hay histograma temporal, crearlo primero
        if self.temporal_histogram is None:
            n_bins = int((t_max - t_min) / adc_sampling_time)
            self.create_temporal_histogram(time_range, n_bins)
        
        # Extraer counts del histograma
        _, histogram_counts = self.temporal_histogram
        
        # Crear forma de pulso del SiPM
        pulse_time = np.arange(0, t_max - t_min, adc_sampling_time)
        A = (sipm_tau_decay + sipm_tau_rise) / (sipm_tau_decay**2) * spe_amplitude
        sipm_pulse = A * (1 - np.exp(-pulse_time / sipm_tau_rise)) * np.exp(-pulse_time / sipm_tau_decay)
        
        # Convolución del histograma con el pulso del SiPM
        self.waveform = np.convolve(histogram_counts, sipm_pulse, mode='full')[:len(self.time_axis)]
        
        # Añadir ruido de línea base
        if baseline_noise_sigma > 0:
            baseline_noise = np.random.normal(0, baseline_noise_sigma, len(self.waveform))
            self.waveform += baseline_noise
            
    def get_corner_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula las posiciones de las 4 esquinas del módulo rectangular.
        
        Returns:
            (r1, r2, r3, r4): tupla de 4 arrays numpy con posiciones 3D de las esquinas
                              en sentido antihorario
        """
        # Construir sistema de coordenadas local
        n = self.normal
        
        # Vector horizontal
        if abs(n[0]) < 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0])
        else:
            arbitrary = np.array([0.0, 1.0, 0.0])
        
        u = np.cross(n, arbitrary)
        u = u / np.linalg.norm(u)
        
        # Vector vertical
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        
        # Calcular esquinas
        half_width = self.width / 2.0
        half_height = self.height / 2.0
        
        r1 = self.position + (-half_width * u) + (-half_height * v)
        r2 = self.position + (+half_width * u) + (-half_height * v)
        r3 = self.position + (+half_width * u) + (+half_height * v)
        r4 = self.position + (-half_width * u) + (+half_height * v)
        
        return r1, r2, r3, r4
    
    def __repr__(self):
        return (f"FROSTModule(id={self.module_id}, "
                f"pos={self.position}, "
                f"n_photons={self.n_photons_detected})")
