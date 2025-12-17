import numpy as np
import matplotlib.pyplot as plt
from detector_simulation import DetectorSimulation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== PARÁMETROS DE CONFIGURACIÓN ====================
# Ajusta estos valores antes de ejecutar la simulación

# Geometría del detector
detector_dimensions = (12.0, 12.0, 55.0)  # (X, Y, Z) en metros
module_dimensions = (0.25, 0.25)          # (ancho, alto) en metros

# Distribución de módulos
n_modules_x = 6  # Número de módulos a lo largo de X
n_modules_y = 6   # Número de módulos a lo largo de Y (altura)
n_modules_z = 27  # Número de módulos a lo largo de Z (eje del haz)

# Parámetros físicos de la simulación
max_distance_light_yield = 10.0  # metros - Distancia máxima para calcular light yield

# Energías de las partículas
kaon_energy = 105   # MeV
muon_energy = 152   # MeV
k_decay_time = 20.0 # ns

# Archivos de datos
wls_histogram_file = "FROST_252x252_sipms_42_2boards_temporal_histogram.txt"
use_wls_smearing = True

# ======================================================================

def simulate_event(detector, generation_point, kaon_energy, muon_energy, 
                   k_decay_time, generate_muon=True, wls_file=None, use_wls_smearing=False):
    """
    Simula un evento completo (kaón o kaón+muón).
    
    Returns:
        detector con módulos actualizados
    """
    # Limpiar estado anterior de los módulos
    for module in detector.modules:
        module.n_photons_arriving = 0.0
        module.n_photons_detected = 0
        module.photon_times = None
        module.temporal_histogram = None
        module.waveform = None
    
    # ========== CALCULAR LIGHT YIELD ==========
    detector.calculate_light_yield_for_nearby_modules(
        generation_point=generation_point,
        kinetic_energy=kaon_energy,
        max_distance=max_distance_light_yield  # Usar el parámetro global
    )
    
    # ========== PROPAGAR FOTONES DEL KAÓN ==========
    detector.propagate_photons_temporally(
        generation_time=0.0,
        generation_point=generation_point,
        use_wls_smearing=use_wls_smearing,
        wls_histogram_file=wls_file if use_wls_smearing else None
    )
    
    # Guardar tiempos de fotones del kaón
    kaon_photon_times = {}
    for module in detector.modules:
        if module.photon_times is not None and len(module.photon_times) > 0:
            kaon_photon_times[module.module_id] = module.photon_times.copy()
    
    # ========== SI HAY MUÓN, AÑADIR SUS FOTONES ==========
    if generate_muon:
        detector.calculate_light_yield_for_nearby_modules(
            generation_point=generation_point,
            kinetic_energy=muon_energy,
            max_distance=max_distance_light_yield  # Usar el parámetro global
        )
        
        detector.propagate_photons_temporally(
            generation_time=k_decay_time,  # El muón se genera después del kaón
            generation_point=generation_point,
            use_wls_smearing=use_wls_smearing,
            wls_histogram_file=wls_file if use_wls_smearing else None
        )
        
        # Combinar fotones de kaón y muón
        for module in detector.modules:
            if module.module_id in kaon_photon_times:
                if module.photon_times is not None and len(module.photon_times) > 0:
                    # Combinar ambos arrays
                    combined = np.concatenate([
                        kaon_photon_times[module.module_id],
                        module.photon_times
                    ])
                    module.photon_times = combined
                    module.n_photons_detected = len(combined)
                else:
                    # Solo había fotones del kaón
                    module.photon_times = kaon_photon_times[module.module_id]
                    module.n_photons_detected = len(module.photon_times)
    
    # ========== APLICAR ELECTRÓNICA ==========
    detector.apply_electronics_to_all_modules(
        sipm_tau_rise=1.5,
        sipm_tau_decay=3,
        spe_amplitude=10.0,
        adc_sampling_time=4,
        time_range=(0, 100),
        baseline_noise_sigma=0.01
    )
    
    return detector


def plot_module_comparison(module, kaon_only_signal, kaon_only_histogram,
                           kaon_muon_signal, kaon_muon_histogram, 
                           time_axis, row, col, fig, title, ncols):
    """
    Añade un subplot mostrando señal de kaón solo vs kaón+muón.
    """
    # Si no hay señal, solo mostrar el título
    if kaon_muon_signal is None or len(kaon_muon_signal) == 0:
        # Añadir texto indicando que no hay señal
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
        # Añadir histograma temporal de fotones (kaón+muón, barras azules)
        fig.add_trace(go.Bar(
            x=time_axis,
            y=kaon_muon_histogram,
            name='Photon histogram',
            marker_color='lightblue',
            showlegend=(row==1 and col==1)
        ), row=row, col=col)
        
        # Añadir señal de kaón+muón (línea roja)
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=kaon_muon_signal,
            mode='lines+markers',
            name='Kaon + Muon',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            showlegend=(row==1 and col==1)
        ), row=row, col=col)
        
        # Añadir señal de kaón solo (línea azul)
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=kaon_only_signal,
            mode='lines+markers',
            name='Kaon only',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            showlegend=(row==1 and col==1)
        ), row=row, col=col)
    
    # NO ACTUALIZAR ANOTACIONES AQUÍ - se hará después de crear todas las trazas


def main():
    """
    Simulación estilo FROST_simu.py con múltiples módulos.
    """
    print("="*80)
    print("SIMULACIÓN DE DETECTOR FROST")
    print("="*80)
    
    # ========== CONFIGURACIÓN DEL DETECTOR ==========
    detector = DetectorSimulation(
        detector_dimensions=detector_dimensions,
        module_dimensions=module_dimensions
    )
    
    detector.create_uniform_module_distribution(
        n_x=n_modules_x,
        n_y=n_modules_y,
        n_z=n_modules_z
    )
    
    # ========== INFORMACIÓN DETALLADA DE GEOMETRÍA ==========
    print("\n" + "="*80)
    print("📐 GEOMETRÍA DEL DETECTOR")
    print("="*80)
    
    print(f"\n🔷 Dimensiones totales del volumen activo:")
    print(f"   • Eje X (horizontal): {detector.detector_length:.2f} m")
    print(f"   • Eje Y (altura vertical): {detector.detector_height:.2f} m")
    print(f"   • Eje Z (eje del haz): {detector.detector_depth:.2f} m")
    print(f"   • Volumen total: {detector.detector_length * detector.detector_height * detector.detector_depth:.2f} m³")
    
    print(f"\n📦 Dimensiones de cada módulo FROST:")
    print(f"   • Ancho: {detector.module_width:.3f} m ({detector.module_width*100:.1f} cm)")
    print(f"   • Alto: {detector.module_height:.3f} m ({detector.module_height*100:.1f} cm)")
    print(f"   • Área activa por módulo: {detector.module_width * detector.module_height:.4f} m² ({detector.module_width * detector.module_height * 10000:.1f} cm²)")
    
    # Calcular espaciados entre módulos
    space_x = detector.detector_length / n_modules_x if n_modules_x > 0 else 0
    space_y = detector.detector_height / n_modules_y if n_modules_y > 0 else 0
    space_z = detector.detector_depth / n_modules_z if n_modules_z > 0 else 0
    
    # Calcular rangos ocupados (centrados)
    total_range_x = (n_modules_x - 1) * space_x if n_modules_x > 1 else 0
    total_range_y = (n_modules_y - 1) * space_y if n_modules_y > 1 else 0
    total_range_z = (n_modules_z - 1) * space_z if n_modules_z > 1 else 0
    
    print(f"\n📊 Distribución de módulos por eje:")
    print(f"   • Eje X: {n_modules_x} módulos")
    print(f"     - Espaciado entre centros: {space_x:.3f} m ({space_x*100:.1f} cm)")
    print(f"     - Rango total ocupado: {total_range_x:.3f} m (del primer al último módulo)")
    print(f"     - Posiciones de módulos: desde x={-total_range_x/2:.2f}m hasta x={+total_range_x/2:.2f}m")
    
    print(f"   • Eje Y: {n_modules_y} módulos")
    print(f"     - Espaciado entre centros: {space_y:.3f} m ({space_y*100:.1f} cm)")
    print(f"     - Rango total ocupado: {total_range_y:.3f} m (del primer al último módulo)")
    print(f"     - Posiciones de módulos: desde y={-total_range_y/2:.2f}m hasta y={+total_range_y/2:.2f}m")
    
    print(f"   • Eje Z: {n_modules_z} módulos")
    print(f"     - Espaciado entre centros: {space_z:.3f} m ({space_z*100:.1f} cm)")
    print(f"     - Rango total ocupado: {total_range_z:.3f} m (del primer al último módulo)")
    print(f"     - Posiciones de módulos: desde z={-total_range_z/2:.2f}m hasta z={+total_range_z/2:.2f}m")
    
    print(f"\n🔢 Distribución de módulos por pared:")
    n_modules_x_walls = 2 * n_modules_y * n_modules_z
    n_modules_z_walls = 2 * n_modules_x * n_modules_y
    total_modules = n_modules_x_walls + n_modules_z_walls
    
    print(f"   • Pared X+ (derecha): {n_modules_y}×{n_modules_z} = {n_modules_y * n_modules_z} módulos")
    print(f"   • Pared X- (izquierda): {n_modules_y}×{n_modules_z} = {n_modules_y * n_modules_z} módulos")
    print(f"   • Pared Z+ (trasera): {n_modules_x}×{n_modules_y} = {n_modules_x * n_modules_y} módulos")
    print(f"   • Pared Z- (frontal): {n_modules_x}×{n_modules_y} = {n_modules_x * n_modules_y} módulos")
    print(f"   • Paredes Y± (arriba/abajo): 0 módulos")
    print(f"   • TOTAL: {total_modules} módulos")
    
    # Calcular área total cubierta
    area_per_wall_x = n_modules_y * n_modules_z * detector.module_width * detector.module_height
    area_per_wall_z = n_modules_x * n_modules_y * detector.module_width * detector.module_height
    total_active_area = 2 * (area_per_wall_x + area_per_wall_z)
    
    # Calcular área total disponible en las paredes
    area_wall_x = detector.detector_height * detector.detector_depth
    area_wall_z = detector.detector_length * detector.detector_height
    total_wall_area = 2 * (area_wall_x + area_wall_z)
    
    coverage_percent = (total_active_area / total_wall_area) * 100
    
    print(f"\n📏 Cobertura de área:")
    print(f"   • Área activa total (SiPMs): {total_active_area:.3f} m² ({total_active_area*10000:.1f} cm²)")
    print(f"   • Área total de paredes: {total_wall_area:.3f} m²")
    print(f"   • Cobertura: {coverage_percent:.2f}%")
    
    print(f"\n⚙️ Parámetros de simulación:")
    print(f"   • Eficiencia de detección (frost_efficiency): {detector.frost_efficiency*100:.1f}%")
    print(f"   • Distancia máxima para light yield: {max_distance_light_yield:.1f} m")
    print(f"   • Índice de refracción LAr: {detector.n_lar}")
    print(f"   • Longitud de Rayleigh: {detector.rayleigh_length} m")
    
    # Punto de generación aleatorio - CORREGIDO
    generation_point = np.array([
        np.random.uniform(-detector.detector_length/2, detector.detector_length/2),
        np.random.uniform(-detector.detector_height/2, detector.detector_height/2),
        np.random.uniform(-detector.detector_depth/2, detector.detector_depth/2)
    ])
    
    print(f"\n📍 Punto de generación del evento:")
    print(f"   • Coordenadas: ({generation_point[0]:.2f}, {generation_point[1]:.2f}, {generation_point[2]:.2f}) m")
    print(f"   • Distancia al centro del detector: {np.linalg.norm(generation_point):.2f} m")
    
    # Calcular distancias a cada pared (PERPENDICULARES)
    dist_to_x_plus = abs(generation_point[0] - detector.detector_length/2)
    dist_to_x_minus = abs(generation_point[0] + detector.detector_length/2)
    dist_to_y_plus = abs(generation_point[1] - detector.detector_height/2)
    dist_to_y_minus = abs(generation_point[1] + detector.detector_height/2)
    dist_to_z_plus = abs(generation_point[2] - detector.detector_depth/2)
    dist_to_z_minus = abs(generation_point[2] + detector.detector_depth/2)
    
    print(f"   • Distancias perpendiculares a cada pared:")
    print(f"     - Pared X+: {dist_to_x_plus:.2f} m")
    print(f"     - Pared X-: {dist_to_x_minus:.2f} m")
    print(f"     - Pared Y+ (techo): {dist_to_y_plus:.2f} m")
    print(f"     - Pared Y- (suelo): {dist_to_y_minus:.2f} m")
    print(f"     - Pared Z+: {dist_to_z_plus:.2f} m")
    print(f"     - Pared Z-: {dist_to_z_minus:.2f} m")
    
    min_dist_to_walls = min(dist_to_x_plus, dist_to_x_minus, dist_to_y_plus, 
                            dist_to_y_minus, dist_to_z_plus, dist_to_z_minus)
    print(f"     - Pared más cercana (distancia perpendicular): {min_dist_to_walls:.2f} m")
    
    print(f"\n⚛️ Energías de las partículas:")
    print(f"   • Kaón: {kaon_energy} MeV")
    print(f"   • Muón: {muon_energy} MeV")
    print(f"   • Tiempo de decay K+: {k_decay_time} ns")
    
    # ========== SIMULACIÓN 1: KAÓN SOLO ==========
    print("\n" + "="*80)
    print("SIMULANDO: Kaón solo (sin decay)")
    print("="*80)
    
    # CORREGIDO: Usar energía total (kaón + muón) para kaón solo
    total_energy = kaon_energy + muon_energy
    
    detector_kaon_only = simulate_event(
        detector, generation_point, 
        total_energy,  # <-- Cambio: antes era kaon_energy
        muon_energy,   # Este parámetro no se usa cuando generate_muon=False
        k_decay_time, 
        generate_muon=False, 
        wls_file=wls_histogram_file, 
        use_wls_smearing=use_wls_smearing
    )
    
    # Guardar resultados del kaón solo
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
    
    # ========== SIMULACIÓN 2: KAÓN + MUÓN ==========
    print("\n" + "="*80)
    print("SIMULANDO: Kaón + Muón (con decay)")
    print("="*80)
    
    detector_kaon_muon = simulate_event(
        detector, generation_point, kaon_energy, muon_energy,
        k_decay_time, generate_muon=True,
        wls_file=wls_histogram_file, use_wls_smearing=use_wls_smearing
    )
    
    # ========== SELECCIONAR MÓDULOS PARA GRAFICAR ==========
    # Calcular módulos con señal
    modules_with_signal = sorted(
        [m for m in detector_kaon_muon.modules if m.n_photons_detected > 0],
        key=lambda m: m.n_photons_detected,
        reverse=True
    )
    
    # CAMBIO: Ahora graficamos TODOS los módulos, ordenados por ID
    all_modules_sorted = sorted(detector_kaon_muon.modules, key=lambda m: m.module_id)
    modules_to_plot = all_modules_sorted  # Mostrar todos los módulos
    
    n_modules_to_plot = len(modules_to_plot)
    n_modules_with_signal = len(modules_with_signal)
    
    # ========== INFORMACIÓN MEJORADA EN TERMINAL ==========
    print("\n" + "="*80)
    print("📊 RESUMEN DE SEÑALES DETECTADAS")
    print("="*80)
    
    total_photons = sum(m.n_photons_detected for m in detector_kaon_muon.modules)
    
    print(f"\n🔢 Estadísticas globales:")
    print(f"   • Módulos totales instalados: {len(detector.modules)}")
    print(f"   • Módulos con señal: {n_modules_with_signal}/{len(detector.modules)} ({100*n_modules_with_signal/len(detector.modules):.1f}%)")
    print(f"   • Fotones totales detectados: {total_photons}")
    
    if n_modules_with_signal == 0:
        print("\n⚠️ Ningún módulo recibió señal")
        print("   Posibles causas:")
        print("   - Punto de generación muy lejos de las paredes")
        print("   - max_distance muy pequeño en calculate_light_yield")
        print("   - frost_efficiency muy bajo")
        return
    
    # Distribución de señal por pared
    print(f"\n📍 Distribución de señal por pared:")
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
    
    for wall in ['X+', 'X-', 'Z+', 'Z-']:  # Solo paredes laterales
        if walls[wall] > 0:
            avg_photons = photons_per_wall[wall] / walls[wall]
            print(f"   • Pared {wall}: {walls[wall]} módulos, {photons_per_wall[wall]} fotones (promedio: {avg_photons:.1f} ph/módulo)")
    
    # TODOS los módulos con señal (no solo top 10)
    print(f"\n🏆 Todos los módulos con señal ({n_modules_with_signal} módulos):")
    for i, m in enumerate(modules_with_signal, 1):
        d = np.linalg.norm(m.position - generation_point)
        pos = m.position
        
        # Determinar pared
        wall = "?"
        if abs(pos[0] - detector.detector_length/2) < tolerance:
            wall = "X+"
        elif abs(pos[0] + detector.detector_length/2) < tolerance:
            wall = "X-"
        elif abs(pos[2] - detector.detector_depth/2) < tolerance:
            wall = "Z+"
        elif abs(pos[2] + detector.detector_depth/2) < tolerance:
            wall = "Z-"
        
        print(f"   {i:2d}. ID={m.module_id:3d} | Pared {wall} | Pos=({pos[0]:5.1f}, {pos[1]:5.1f}, {pos[2]:6.1f})m | d={d:5.2f}m | {m.n_photons_detected:4d} ph")
    
    # ========== ANÁLISIS MEJORADO DE MÓDULOS ADYACENTES SIN SEÑAL ==========
    if n_modules_with_signal > 0:
        print(f"\n🔍 Análisis COMPLETO de módulos adyacentes sin señal:")
        print(f"   (Analizando TODOS los módulos con señal y sus vecinos)")
        
        # Analizar módulos adyacentes para CADA módulo con señal
        for idx, signal_module in enumerate(modules_with_signal[:5], 1):  # Analizar top 5 con señal
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
            print(f"   Módulo #{idx} con señal:")
            print(f"   • ID={signal_module.module_id}, Pared {sig_wall}, Pos=({sig_pos[0]:.1f}, {sig_pos[1]:.1f}, {sig_pos[2]:.1f})m")
            print(f"   • Distancia al punto: {np.linalg.norm(sig_pos - generation_point):.2f}m")
            print(f"   • Fotones detectados: {signal_module.n_photons_detected}")
            
            # Buscar módulos adyacentes
            adjacent_modules = []
            for m in detector.modules:
                if m.module_id == signal_module.module_id:
                    continue
                
                # Mismo eje perpendicular a la pared (misma pared)
                if abs(m.position[0] - sig_pos[0]) < tolerance or \
                   abs(m.position[2] - sig_pos[2]) < tolerance:
                    # Calcular distancia 2D en el plano de la pared
                    if sig_wall in ['X+', 'X-']:
                        dist_in_plane = np.sqrt((m.position[1] - sig_pos[1])**2 + 
                                               (m.position[2] - sig_pos[2])**2)
                    else:  # Z+ o Z-
                        dist_in_plane = np.sqrt((m.position[0] - sig_pos[0])**2 + 
                                               (m.position[1] - sig_pos[1])**2)
                    
                    if 0 < dist_in_plane < 3*max(space_y, space_z):  # Vecinos cercanos
                        dist_to_gen = np.linalg.norm(m.position - generation_point)
                        adjacent_modules.append({
                            'module': m,
                            'dist_in_plane': dist_in_plane,
                            'dist_to_gen': dist_to_gen
                        })
            
            # Ordenar por distancia en el plano
            adjacent_modules.sort(key=lambda x: x['dist_in_plane'])
            
            # Contar cuántos tienen y no tienen señal
            n_adjacent_with_signal = sum(1 for adj in adjacent_modules if adj['module'].n_photons_detected > 0)
            n_adjacent_without_signal = len(adjacent_modules) - n_adjacent_with_signal
            
            print(f"\n   Módulos adyacentes encontrados: {len(adjacent_modules)}")
            print(f"   • CON señal: {n_adjacent_with_signal}")
            print(f"   • SIN señal: {n_adjacent_without_signal}")
            
            if n_adjacent_without_signal > 0:
                print(f"\n   Módulos adyacentes SIN señal (ordenados por cercanía):")
                
                without_signal_count = 0
                for adj in adjacent_modules:
                    m = adj['module']
                    if m.n_photons_detected == 0:
                        without_signal_count += 1
                        
                        print(f"\n   {without_signal_count}. ID={m.module_id:3d} | ✗ SIN señal")
                        print(f"      Pos=({m.position[0]:5.1f}, {m.position[1]:5.1f}, {m.position[2]:6.1f})m")
                        print(f"      Dist. al módulo con señal: {adj['dist_in_plane']:.2f}m (en plano)")
                        print(f"      Dist. al punto gen.: {adj['dist_to_gen']:.2f}m")
                        
                        # Diagnóstico detallado
                        if adj['dist_to_gen'] > max_distance_light_yield:
                            print(f"      ⚠️  CAUSA: Distancia > max_distance ({max_distance_light_yield:.1f}m)")
                            print(f"          → No se calculó light yield para este módulo")
                        else:
                            # Calcular ángulo sólido aproximado
                            solid_angle = (detector.module_width * detector.module_height) / (adj['dist_to_gen']**2)
                            print(f"      ⚠️  CAUSA PROBABLE: Ángulo sólido muy pequeño")
                            print(f"          → Ángulo sólido ≈ {solid_angle:.6f} sr")
                            print(f"          → Pocos fotones generados hacia este módulo")
                            print(f"          → Fotones no llegaron o fueron absorbidos/reflejados")
            else:
                print(f"\n   ✅ TODOS los módulos adyacentes tienen señal")
        
        # Estadísticas globales de adyacencia
        print(f"\n   {'='*70}")
        print(f"\n   📊 Estadísticas globales de módulos sin señal:")
        
        n_without_signal = len(detector.modules) - n_modules_with_signal
        
        if n_without_signal == 0:
            print(f"   • ✅ TODOS los módulos ({len(detector.modules)}) tienen señal!")
            print(f"   • ⚠️ Esto es inusual. Probablemente max_distance está configurado demasiado alto.")
        else:
            n_too_far = sum(1 for m in detector.modules 
                           if m.n_photons_detected == 0 and 
                           np.linalg.norm(m.position - generation_point) > max_distance_light_yield)
            n_within_range = n_without_signal - n_too_far
            
            print(f"   • Total sin señal: {n_without_signal}")
            print(f"   • Fuera de max_distance: {n_too_far} ({100*n_too_far/n_without_signal:.1f}%)")
            print(f"   • Dentro de max_distance pero sin señal: {n_within_range} ({100*n_within_range/n_without_signal:.1f}%)")
    
    # ========== CALCULAR LAYOUT DE SUBPLOTS ==========
    ncols = 5
    nrows = int(np.ceil(n_modules_to_plot / ncols))
    
    print(f"\n📈 Generando visualización:")
    print(f"   • Layout: {nrows} filas × {ncols} columnas = {n_modules_to_plot} módulos")
    
    # Espaciado optimizado - AJUSTADO DINÁMICAMENTE
    # Máximo espaciado permitido es 1/(nrows-1) según Plotly
    max_vertical_spacing = 1.0 / (nrows - 1) if nrows > 1 else 0.0
    max_horizontal_spacing = 1.0 / (ncols - 1) if ncols > 1 else 0.0
    
    # Usar el menor entre el deseado y el máximo permitido
    vertical_spacing = min(0.03, max_vertical_spacing * 0.8) if nrows > 1 else 0.0
    horizontal_spacing = min(0.03, max_horizontal_spacing * 0.8) if ncols > 1 else 0.0
    
    print(f"   • Espaciado vertical: {vertical_spacing:.4f}")
    print(f"   • Espaciado horizontal: {horizontal_spacing:.4f}")
    
    # ========== CREAR FIGURA CON SUBPLOTS DINÁMICOS ==========
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
        
        # Obtener información del módulo - DISTANCIA CORRECTA
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
        
        # Crear el título ANTES de llamar a plot_module_comparison
        title_text = f"<b>ID: {module.module_id}</b> | ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m | d={distance:.2f}m | {module.n_photons_detected} ph"
        
        # Graficar comparación en el subplot correspondiente
        plot_module_comparison(
            module, kaon_only_signal, kaon_only_histogram,
            kaon_muon_signal, kaon_muon_histogram,
            time_axis, row, col, fig, title_text, ncols
        )
        
        # Añadir anotación usando coordenadas de PAPEL - CORREGIDO
        subplot_height = (1.0 - (nrows - 1) * vertical_spacing) / nrows
        subplot_width = (1.0 - (ncols - 1) * horizontal_spacing) / ncols
        
        x_paper = (col - 1) * (subplot_width + horizontal_spacing) + subplot_width / 2
        y_paper = 1.0 - ((row - 1) * (subplot_height + vertical_spacing))
        
        # Ajustar offset: mayor para la primera fila (para evitar colisión con título principal)
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
    
    # ========== CONFIGURAR LAYOUT ==========
    for i in range(1, nrows + 1):
        for j in range(1, ncols + 1):
            if i == nrows:
                fig.update_xaxes(title_text="Time (ns)", row=i, col=j, title_font=dict(size=9))
            else:
                fig.update_xaxes(title_text="", row=i, col=j)
            
            if j == 1:
                fig.update_yaxes(title_text="Amp", row=i, col=j, title_font=dict(size=9))
            else:
                fig.update_yaxes(title_text="", row=i, col=j)
    
    plot_height = max(225 * nrows, 500)
    
    # ========== TÍTULO MEJORADO CON TODA LA INFO DE DISTANCIAS ==========
    # Calcular distancias mínimas a cada pared desde el punto de generación
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
    
    # Calcular distancias para los módulos
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
            f"<b>FROST Detector - {n_modules_to_plot} módulos ({n_modules_with_signal} con señal)</b><br>"
            f"<sub>Detector: {detector.detector_length:.0f}×{detector.detector_height:.0f}×{detector.detector_depth:.0f}m | "
            f"Módulos: {detector.module_width*100:.0f}×{detector.module_height*100:.0f}cm | "
            f"Eficiencia: {detector.frost_efficiency*100:.0f}% | "
            f"max_dist: {max_distance_light_yield:.1f}m</sub><br>"
            f"<sub>K+ decay: {k_decay_time}ns | "
            f"Pto gen: ({generation_point[0]:.1f}, {generation_point[1]:.1f}, {generation_point[2]:.1f})m → "
            f"Pared {closest_wall_name} ({closest_wall_dist:.2f}m) | "
            f"Módulo más cercano: {min_mod_dist:.2f}m | "
            f"Señal hasta: {max_signal_dist:.2f}m</sub>"
        ),
        title_font=dict(size=11),  # Reducido un poco más
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
        margin=dict(l=40, r=20, t=200, b=40)  # AUMENTADO de 120 a 160
    )
    
    # Títulos de subplots más pequeños
    for annotation in fig.layout.annotations:
        annotation.font.size = 8  # Aumentado de 7 a 8 para mejor legibilidad
    
    # ========== GUARDAR PRIMER PLOT (TODOS LOS MÓDULOS) ==========
    fig.write_html('frost_detector_all_modules.html')
    print(f"   • Archivo guardado: 'frost_detector_all_modules.html'")
    
    # ========== CREAR SEGUNDO PLOT SOLO CON MÓDULOS CON SEÑAL ==========
    print(f"\n📈 Generando visualización solo de módulos con señal:")
    
    modules_with_signal_to_plot = modules_with_signal
    n_signal_modules = len(modules_with_signal_to_plot)
    
    if n_signal_modules > 0:
        ncols_signal = 5
        nrows_signal = int(np.ceil(n_signal_modules / ncols_signal))
        
        print(f"   • Layout: {nrows_signal} filas × {ncols_signal} columnas = {n_signal_modules} módulos con señal")
        
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
                    fig_signal.update_xaxes(title_text="Time (ns)", row=i, col=j, title_font=dict(size=9))
                else:
                    fig_signal.update_xaxes(title_text="", row=i, col=j)
                
                if j == 1:
                    fig_signal.update_yaxes(title_text="Amp", row=i, col=j, title_font=dict(size=9))
                else:
                    fig_signal.update_yaxes(title_text="", row=i, col=j)
        
        plot_height_signal = max(220 * nrows_signal, 500)
        
        fig_signal.update_layout(
            title_text=(
                f"<b>FROST Detector - {n_signal_modules} módulos CON SEÑAL de {len(detector.modules)} módulos totales</b><br>"
                f"<sub>Detector: {detector.detector_length:.0f}×{detector.detector_height:.0f}×{detector.detector_depth:.0f}m | "
                f"Módulos: {detector.module_width*100:.0f}×{detector.module_height*100:.0f}cm | "
                f"Eficiencia: {detector.frost_efficiency*100:.0f}% | "
                f"max_dist: {max_distance_light_yield:.1f}m</sub><br>"
                f"<sub>K+ decay: {k_decay_time}ns | "
                f"Pto gen: ({generation_point[0]:.1f}, {generation_point[1]:.1f}, {generation_point[2]:.1f})m → "
                f"Pared {closest_wall_name} ({closest_wall_dist:.2f}m) | "
                f"Módulo más cercano: {min_mod_dist:.2f}m | "
                f"Señal hasta: {max_signal_dist:.2f}m</sub>"
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
        print(f"   • Archivo guardado: 'frost_detector_with_signal.html'")
        fig_signal.show()
    
    # Mostrar el primer plot
    fig.show()
    
    print("\n✅ Simulación completada exitosamente!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
