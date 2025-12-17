import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import resample


file ='FROST_252x252_sipms_42_2boards_temporal_histogram.txt'

tbins=100

#general parameters
c_light = 0.3  # m/ns

# generation parameters
kaon_energy=105  # MeV
muon_energy=152  # MeV
k_decay_time = 20  # ns
distance_to_frost=3 # distance from kaon proton decay to FROST module (m)
generate_muon = True

#LAr parameters
n_lar=1.23
rayleigh_length = 8  #m
rayleigh_power=1.5
#tau_fast=6     # ns
#tau_fast=3     # ns
tau_fast=0.1     # ns
#tau_slow=1500  # ns
#tau_slow=100 # ns
tau_slow=3000 # ns

# scintillation parameters
fast_contribution = 0.23
n_photons_mev=25000

# FROST parameters
frost_height = 0.252 #m
frost_width  = 0.252 #m
frost_eff=0.05
tau_wls=0*1.2   # ns
tau_ptp=0*1.45  # ns
simulate_prop_wls_smearing=True



# FROST distribution
frost_time_alignment_error=0  # alignment between several from modules (ns)
n_frost=1  # number of frost modules
frost_time_alignment = np.random.normal(0, frost_time_alignment_error, n_frost)

# Electronics parameters
sipm_tau_rise  = 1.5   # ns
sipm_tau_decay = 3.0  # ns
adc_sampling_time = 1    # ns   
baseline_error = 0
spe_amplitude = 10
amp_simulation = False
sampling_rate_hz = 1e9


global cdf_histo_prop_wls 
global bin_edges_histo_prop_wls

ncols=1
nrows=1

###############################################
def main():

    global cdf_histo_prop_wls, bin_edges_histo_prop_wls 
    
    histo_prop_wls, bin_edges_histo_prop_wls = read_file(file)

    # Step 2a: Compute cumulative distribution (CDF)
    bin_widths= np.diff(bin_edges_histo_prop_wls)
    cdf = np.cumsum(histo_prop_wls * bin_widths)
    cdf_histo_prop_wls = cdf / cdf[-1]  # ensure last value is exactly 1

    
    # ========== BUCLE SOBRE k_decay_time ==========
    #k_decay_times = np.arange(4, 21, 1)  # De 4 a 20 ns en pasos de 1 ns
    k_decay_times =[20]
    
    for k_decay_val in k_decay_times:
        global k_decay_time
        k_decay_time = k_decay_val
        
        print(f"Procesando k_decay_time = {k_decay_time} ns...")
        
        # Create subplots
        fig = make_subplots(rows=nrows, cols=ncols, 
                          subplot_titles=[f'K+ decay time = {k_decay_time} ns'])

        global sipm_tau_rise, sipm_tau_decay, rayleigh_length

        sipm_tau_rise  = 1.5    # ns
        sipm_tau_decay  = 15    # ns
        
        # Generar señales
        histo_kaon_muon, signal_kaon_muon = create_kaon_muon_signals(fig, 1, 1, generate_muon_flag=True)
        histo_kaon_only, signal_kaon_only = create_kaon_muon_signals(fig, 1, 1, generate_muon_flag=False)
        
        # Guardar histogramas en archivos de texto
        save_histogram_to_file(histo_kaon_muon, k_decay_time, "kaon_muon")
        save_histogram_to_file(histo_kaon_only, k_decay_time, "kaon_only")
        
        # Crear plot
        create_kaon_muon_plot_with_data(fig, 1, 1, 
                                       histo_kaon_muon, signal_kaon_muon,
                                       histo_kaon_only, signal_kaon_only,
                                       f'τ_r = {sipm_tau_rise} ns, τ_d = {sipm_tau_decay} ns')
        
        # Configuración de ejes
        fig.update_xaxes(title_text="time (ns)")
        fig.update_yaxes(title_text="entries")

        fig.update_layout(
            title_text= f"FROST toy simulation: K decay time = {k_decay_time} ns, "
            + f"Distance = {distance_to_frost} m, "
            + f"FROST size = {frost_width}x{frost_height} m²",
            template='plotly_white',
            showlegend=True,
            width=1200,   # Ancho en píxeles
            height=800    # Alto en píxeles
        )

        # Guardar como PNG (más rápido y directo)
        png_filename = f'FROST_k_decay_{k_decay_time}ns.png'
        fig.write_image(png_filename, scale=2)  # scale=2 para mayor resolución
        print(f"  - Plot guardado: {png_filename}")
        
        # Opcionalmente también guardar HTML (comentar si no lo necesitas)
        # html_filename = f'FROST_k_decay_{k_decay_time}ns.html'
        # fig.write_html(html_filename)
        # print(f"  - HTML guardado: {html_filename}")

    print("\n✅ Todos los plots y archivos de datos generados exitosamente!")
    return    

###############################################
def create_kaon_muon_signals(fig, row, col, generate_muon_flag):
    """
    Genera las señales sin añadirlas al plot (para poder guardarlas antes).
    
    Returns:
        histo_times_at_sipms: histograma de tiempos de llegada
        signal: señal después de la convolución electrónica
    """
    global generate_muon
    generate_muon = generate_muon_flag
    
    histo_times_at_sipms = np.array([])
    signal = np.array([])
    
    # loop over n_frost modules
    for ifrost in range(0, n_frost):
        histo_times_at_sipms_i, signal_i = frost_signal(ifrost, distance_to_frost)
        if ifrost == 0:
            histo_times_at_sipms = histo_times_at_sipms_i
            signal = signal_i
        else:
            histo_times_at_sipms += histo_times_at_sipms_i
            signal += signal_i
    
    return histo_times_at_sipms, signal

###############################################
def create_kaon_muon_plot_with_data(fig, row, col, 
                                    histo_kaon_muon, signal_kaon_muon,
                                    histo_kaon_only, signal_kaon_only,
                                    name):
    """
    Añade los datos previamente calculados al plot.
    """
    t_max = tbins
    t = np.arange(0, t_max, adc_sampling_time)
    
    # Añadir histograma de kaon+muon (barras azules)
    fig.add_trace(go.Bar(
        x=t, y=histo_kaon_muon, name='Photon histogram (K+μ)', 
        marker_color='lightblue'
    ), row=row, col=col)
    
    # Añadir señal de kaon+muon (línea roja)
    fig.add_trace(go.Scatter(
        x=t, y=signal_kaon_muon, mode='lines+markers', name='K+μ signal',
        line=dict(color='red', width=2), marker=dict(size=6)
    ), row=row, col=col)
    
    # Añadir señal de kaon only (línea azul)
    fig.add_trace(go.Scatter(
        x=t, y=signal_kaon_only, mode='lines+markers', name='K only signal',
        line=dict(color='blue', width=2), marker=dict(size=6)
    ), row=row, col=col)

###############################################
def save_histogram_to_file(histogram, k_decay_value, label):
    """
    Guarda el histograma en un archivo de texto.
    
    Args:
        histogram: array con los valores del histograma
        k_decay_value: valor de k_decay_time usado
        label: etiqueta para el nombre del archivo ("kaon_muon" o "kaon_only")
    """
    t_max = tbins
    t = np.arange(0, t_max, adc_sampling_time)
    
    filename = f'histogram_{label}_k_decay_{k_decay_value}ns.txt'
    
    with open(filename, 'w') as f:
        f.write(f"# FROST histogram - {label}\n")
        f.write(f"# k_decay_time = {k_decay_value} ns\n")
        f.write(f"# distance_to_frost = {distance_to_frost} m\n")
        f.write(f"# Column 1: time (ns)\n")
        f.write(f"# Column 2: number of photons\n")
        f.write("#" + "="*50 + "\n")
        
        for time_val, photon_count in zip(t, histogram):
            f.write(f"{time_val:.1f}\t{photon_count:.6f}\n")
    
    print(f"  - Archivo guardado: {filename}")

###############################################
def create_kaon_muon_plot(fig:go.Figure, row,col,name):
    # Esta función ya no se usa, pero se mantiene por compatibilidad
    pass

###############################################
def add_conf_plot(fig:go.Figure, row,col,
                  name, color='red', add_histo=False):
    # Esta función ya no se usa, pero se mantiene por compatibilidad
    pass


###############################################
def frost_signal(ifrost: int, dist: float):

    
    #-------- Light generation and propagation to FROST
    
    # number of kaon/muon photons arriving to frost
    npe_k  = light_yield_at_frost(kaon_energy,dist)
    npe_mu = light_yield_at_frost(muon_energy,dist)

    if not generate_muon:
        npe_k+=npe_mu
    
    #-------- FROST optical/geometrical response
    
    # generate the vector with the time of arrival to SiPMs
    times_at_sipms = compute_times_at_sipms(0,npe_k,dist)
    if generate_muon:
        times_at_sipms_mu = compute_times_at_sipms(k_decay_time,npe_mu,dist)
        times_at_sipms = np.append(times_at_sipms, times_at_sipms_mu)


    #-------- FROST electronics response        

    # introduce time misaligment between frost modules    
    times_at_sipms += frost_time_alignment[ifrost]
    
    # Create photon arrival histogram
    t_max = tbins    # ns
    t = np.arange(0, t_max, adc_sampling_time)
    histo_times_at_sipms, bin_edges = np.histogram(times_at_sipms, bins=t, density=False)
    
    # electronics shaping
    signal = electronics_convolution(histo_times_at_sipms, t_max)

    # amplifier simulation
    if amp_simulation:
        signal = apply_custom_transfer(signal, sampling_rate_hz)
    
    # baseline noise
    baseline_noise = np.random.normal(0, baseline_error, len(signal))
    signal += baseline_noise
    
    return histo_times_at_sipms, signal

###############################################
def compute_times_at_sipms(t0: float, npe: int, dist: float):

    times_at_sipms=[]
    for i in range(0,int(frost_eff*npe*(1-fast_contribution))):
        t = t0 \
        + prop_wls() \
        + slow_dt() \
        + ptp_dt() \
        + wls_dt() \
        + prop_lar(dist)

        if (t<tbins):
            times_at_sipms = np.append(times_at_sipms,t)
    for i in range(0,int(frost_eff*npe*fast_contribution)):
        t = t0 \
        + prop_wls() \
        + fast_dt() \
        + ptp_dt() \
        + wls_dt() \
        + prop_lar(dist)

        if (t<tbins):
            times_at_sipms = np.append(times_at_sipms,t)

    return times_at_sipms

###############################################
def light_yield_at_frost(kinetic_energy: float, distance_to_frost:float):
    return kinetic_energy*n_photons_mev*frost_height*frost_width/(4*3.1416*distance_to_frost**2)

###############################################
def fast_dt():

    return np.random.exponential(scale=tau_fast, size=1)

###############################################
def slow_dt():

    return np.random.exponential(scale=tau_slow, size=1)

###############################################
def ptp_dt():

    return np.random.exponential(scale=tau_ptp, size=1)

###############################################
def wls_dt():

    return np.random.exponential(scale=tau_wls, size=1)

###############################################
def prop_lar(dist: float):

    delta_t = pow(dist,rayleigh_power)*n_lar/(c_light*rayleigh_length)
    
    return np.random.normal(loc=0, scale=delta_t)

###############################################
def frost_t_align():

    return np.random.normal(loc=0, scale=frost_time_alignment_error)

###############################################
def prop_wls():


    if not simulate_prop_wls_smearing:
        return 0
        
    global cdf_histo_prop_wls, bin_edges_histo_prop_wls 
    
    # Step 2b: Generate uniform random numbers
    u = np.random.rand(1) 

    # Step 2c: Invert the CDF to get samples
    delta_t = np.interp(u, cdf_histo_prop_wls, bin_edges_histo_prop_wls[1:])

    return delta_t

###############################################
def sipm_pulse(t):

    # SiPM parameters
    A = (sipm_tau_decay + sipm_tau_rise) / (sipm_tau_decay**2)*spe_amplitude           # amplitude
    
    # Pulse shape
    pulse = A * (1 - np.exp(-t/sipm_tau_rise)) * np.exp(-t/sipm_tau_decay)
    
    return pulse

###############################################
def electronics_convolution(histo_times_at_sipms: np.histogram, t_max: float):

    # Simulation parameters
    t = np.arange(0, t_max, adc_sampling_time)
    
    # Create SiPM pulse vector for convolution
    pulse_time = np.arange(0, t_max, adc_sampling_time)   # pulse duration 100 ns
    pulse_vector = sipm_pulse(pulse_time)
    
    # Convolve histogram with SiPM pulse
    signal = np.convolve(histo_times_at_sipms, pulse_vector, mode='full')[:len(t)]
    
    return signal


###############################################
def apply_custom_transfer(signal, sampling_rate_hz):

    # Función para aplicar función de transferencia H(s) = G * (a s²)/(1 + b s + c s²)

    G = 10e-2        # amp gain in open lace
    Rq = 2.2e6       # quenching R
    Cd = 50e-15      # parasitic C
    N = 6000         # cells
    M = 1            # number of SiPMs in parallel           
    Ri = 1           # input R
    Rf = 390         # feedback R
    Tb = 200e-9      # output bandwidth limit
    Lo = 50e-6       # output inductance 
    Co = 100e-9      # output C
    Ro = 50          # output R

    # FFT
    Npts = len(signal)
    dt = 1 / sampling_rate_hz
    freqs = fftfreq(Npts, d=dt)
    omega = 2 * np.pi * freqs
    epsilon = 1e-20      
    s = 1j * omega
    s[np.abs(s) < epsilon] = epsilon  # evita dividir por cero en DC
    # Z(s)
    Zs = (1 + s * Rq * Cd) / (s * N * Cd)

    # Bloques de la función de transferencia
    H1 = (Zs / M) / ((Zs / M) + 2 * Ri)
    H2 = 2 * Rf / (1 + s * Tb)
    H3 = (s**2 * Lo * Co) / (1 + s * ((Lo / Ro) + Ro * Co) + 2 * s**2 * Lo * Co)

    # Función de transferencia total
    H = G * H1 * H2 * H3

    # Aplicar FFT e IFFT
    signal_fft = fft(signal)
    output_fft = signal_fft * H
    output_time = ifft(output_fft)

    return np.real(output_time)

###############################################
def read_file(file):

    # Read data from file
    values = []
    counts = []
    
    with open(file, "r") as f:
        for line in f:
            if line[0]=='#':
                continue
            if line.strip():  # skip empty lines
                v, c = line.split()
                values.append(float(v))
                counts.append(int(c))



    hist, bin_edges = np.histogram(values,bins=len(values),weights=counts)

    # Example: combine every 5 bins into 1
    hist , bin_edges = rebin_histogram(hist, bin_edges, factor=20)

    hist  = normalize_hist(hist, bin_edges)
    
#    plt.figure(figsize=(7,4))
#    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
#    plt.xlabel("Value")
#    plt.ylabel("Count")
#    plt.title("Histogram from bin centers + counts")
#    plt.tight_layout()
#    plt.show()

    return hist, bin_edges
    
###############################################
def rebin_histogram(hist, bin_edges, factor):
    # factor: how many old bins to combine into one new bin
    import numpy as np
    
    if len(hist) % factor != 0:
        raise ValueError("Number of bins must be divisible by the rebin factor")
    
    new_hist = hist.reshape(-1, factor).sum(axis=1)
    new_bin_edges = bin_edges[::factor]
    
    # make sure the last edge is included
    if len(new_bin_edges) != len(new_hist)+1:
        new_bin_edges = np.append(new_bin_edges, bin_edges[-1])
    
    return new_hist, new_bin_edges

###############################################
def normalize_hist(hist, bin_edges):
    # Compute bin widths
    bin_widths = np.diff(bin_edges)
    # Normalize so that sum(hist * bin_widths) = 1
    hist_normalized = hist / np.sum(hist * bin_widths)
    return hist_normalized

    
############
main()
############