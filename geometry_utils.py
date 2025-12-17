import numpy as np
from typing import Tuple

def calculate_solid_angle(point: np.ndarray, 
                         r1: np.ndarray, 
                         r2: np.ndarray, 
                         r3: np.ndarray, 
                         r4: np.ndarray) -> float:
    """
    Calcula el ángulo sólido subtendido por un rectángulo arbitrario desde un punto P.
    
    Usa la fórmula general para un cuadrilátero:
    
    Ω = Σ(i=1 to 4) arctan( v_i · (v_{i+1} × v_{i+2}) / 
                             (||v_i|| ||v_{i+1}|| ||v_{i+2}|| + v_i·v_{i+1} ||v_{i+2}|| 
                              + v_{i+1}·v_{i+2} ||v_i|| + v_{i+2}·v_i ||v_{i+1}||) )
    
    Args:
        point: np.array([x, y, z]) - punto desde donde se calcula el ángulo sólido
        r1, r2, r3, r4: np.array([x, y, z]) - esquinas del rectángulo en orden antihorario
    
    Returns:
        Ω: ángulo sólido en estereorradianes
    """
    # Vectores desde P a cada esquina
    v = [r1 - point, r2 - point, r3 - point, r4 - point]
    
    # Normas de los vectores
    norms = [np.linalg.norm(vi) for vi in v]
    
    # Evitar división por cero si el punto está muy cerca de una esquina
    if min(norms) < 1e-10:
        return 0.0
    
    # Suma sobre las 4 esquinas
    omega = 0.0
    for i in range(4):
        # Índices cíclicos
        i_next = (i + 1) % 4
        i_next2 = (i + 2) % 4
        
        # Producto mixto: v_i · (v_{i+1} × v_{i+2})
        cross_product = np.cross(v[i_next], v[i_next2])
        numerator = np.dot(v[i], cross_product)
        
        # Denominador
        denominator = (
            norms[i] * norms[i_next] * norms[i_next2]
            + np.dot(v[i], v[i_next]) * norms[i_next2]
            + np.dot(v[i_next], v[i_next2]) * norms[i]
            + np.dot(v[i_next2], v[i]) * norms[i_next]
        )
        
        # Evitar división por cero
        if abs(denominator) > 1e-12:
            omega += np.arctan(numerator / denominator)
    
    return abs(omega)


def calculate_photon_yield_for_module(module, 
                                       generation_point: np.ndarray,
                                       kinetic_energy: float,
                                       n_photons_per_mev: float = 25000) -> float:
    """
    Calcula el número de fotones que llegan a un módulo FROST desde un punto.
    
    IMPORTANTE: Esta implementación es coherente con FROST_simu.py, que NO calcula
    el ángulo crítico ni la transmisión de Fresnel LAr->PMMA. Solo usa una 
    eficiencia global (frost_efficiency) que se aplica posteriormente.
    
    Args:
        module: objeto FROSTModule
        generation_point: np.array([x, y, z]) - punto de generación de fotones
        kinetic_energy: energía depositada (MeV)
        n_photons_per_mev: fotones de centelleo por MeV
        
    Returns:
        número de fotones que llegan al módulo (float)
    """
    # Distancia desde el módulo al punto de generación
    distance = module.get_distance_to_point(generation_point)
    
    if distance < 1e-6:  # Evitar división por cero
        return 0.0
    
    # Dirección del módulo al punto de generación
    direction = (module.position - generation_point) / distance
    
    # Calcular esquinas del módulo
    r1, r2, r3, r4 = module.get_corner_positions()
    
    # Calcular ángulo sólido subtendido por el módulo
    omega = calculate_solid_angle(generation_point, r1, r2, r3, r4)
    
    # Fotones totales emitidos
    total_photons = kinetic_energy * n_photons_per_mev
    
    # Fotones que llegan según ángulo sólido
    # (NO se aplica transmisión de Fresnel aquí, solo geometría)
    photons_arriving = total_photons * omega / (4 * np.pi)
    
    return photons_arriving
