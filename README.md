# actividad3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

# --- Funciones útiles ---

def crear_senal(t):
    # Señal compuesta: suma de 3 senos de distintas frecuencias + ruido blanco
    f1, f2, f3 = 5, 50, 120  # Frecuencias en Hz
    señal = (np.sin(2*np.pi*f1*t) + 
             0.5*np.sin(2*np.pi*f2*t) + 
             0.3*np.sin(2*np.pi*f3*t))
    ruido = 0.4 * np.random.normal(size=len(t))
    return señal + ruido

def diseno_filtro(tipo, orden, fc, fs, banda=None):
    nyq = 0.5 * fs
    if tipo == 'band':
        low = banda[0] / nyq
        high = banda[1] / nyq
        b, a = butter(orden, [low, high], btype='band')
    else:
        normal_cutoff = fc / nyq
        b, a = butter(orden, normal_cutoff, btype=tipo)
    return b, a

def aplicar_filtro(b, a, data):
    return lfilter(b, a, data)

def graficar_senal(t, señal, título):
    plt.figure(figsize=(10,4))
    plt.plot(t, señal, label='Señal')
    plt.title(título)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()

def graficar_respuesta_frecuencia(b, a, fs, título):
    w, h = freqz(b, a, worN=8000)
    plt.figure(figsize=(10,4))
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.title(título)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Ganancia')
    plt.grid(True)
    plt.show()

# --- Parámetros ---
fs = 500.0  # Frecuencia de muestreo (Hz)
t = np.arange(0, 2.0, 1/fs)  # Vector tiempo 2 segundos

# --- Generar señal ---
senal_original = crear_senal(t)
graficar_senal(t, senal_original, 'Señal Original con Ruido')

# --- Diseñar filtro pasa bajos ---
orden_lp = 6
fc_lp = 30.0  # Frecuencia de corte 30 Hz
b_lp, a_lp = diseno_filtro('low', orden_lp, fc_lp, fs)

graficar_respuesta_frecuencia(b_lp, a_lp, fs, 'Respuesta en Frecuencia - Filtro Pasa Bajos')

senal_filtrada_lp = aplicar_filtro(b_lp, a_lp, senal_original)
graficar_senal(t, senal_filtrada_lp, 'Señal después del Filtro Pasa Bajos')

# --- Diseñar filtro pasa altos ---
orden_hp = 6
fc_hp = 40.0  # Frecuencia de corte 40 Hz
b_hp, a_hp = diseno_filtro('high', orden_hp, fc_hp, fs)

graficar_respuesta_frecuencia(b_hp, a_hp, fs, 'Respuesta en Frecuencia - Filtro Pasa Altos')

senal_filtrada_hp = aplicar_filtro(b_hp, a_hp, senal_original)
graficar_senal(t, senal_filtrada_hp, 'Señal después del Filtro Pasa Altos')

# --- Diseñar filtro pasa banda ---
orden_bp = 4
banda_bp = [40, 100]  # Banda 40-100 Hz
b_bp, a_bp = diseno_filtro('band', orden_bp, None, fs, banda=banda_bp)

graficar_respuesta_frecuencia(b_bp, a_bp, fs, 'Respuesta en Frecuencia - Filtro Pasa Banda')

senal_filtrada_bp = aplicar_filtro(b_bp, a_bp, senal_original)
graficar_senal(t, senal_filtrada_bp, 'Señal después del Filtro Pasa Banda')
