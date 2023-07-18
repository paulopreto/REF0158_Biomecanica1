# -*- coding: utf-8 -*-
"""
EMG Signal Processing
Author: Prof. PAULO ROBERTO PEREIRA SANTIAGO
Institution: University of São Paulo
             School of Physical Education and Sport of Ribeirão Preto
             Biomechanics and Motor Control Laboratory
Date: 07/03/2023

This Python script was created to perform electromyographic (EMG) signal processing.
Initially, the EMG signal file is loaded into the program, followed by an analysis that is
performed between two points in time, defined by the user. Subsequently, the signal is filtered
with a 15-450 Hz band-pass filter. After this step, an RMS (Root Mean Square) calculation is
performed using 1 second windows and 1/2 second overlap. The script also calculates the
median frequency for each window of the filtered EMG signal. The processing results in four
plots: raw EMG signal, filtered EMG signal, RMS EMG, and median frequency over time.

How to use:
The script is run via the command line and accepts four arguments:
python3 emg_labiocom.py [emg_file] [fs] [start_time] [end_time] [--no-plot]

Where:
emg_file: is the path to the EMG signal file you wish to process.
fs: is the sampling rate of the EMG signal in Hz.
start_time: is the start time for the signal analysis in seconds.
end_time: is the end time for the signal analysis in seconds.
--no-plot: is an option that, if specified, disables the plotting of the EMG signal plots.

Example:
python3 emg_labiocom.py emg_data.txt 2000 2 7 --no-plot

Output:
The script will produce four plots representing:
1. Raw EMG signal.
2. Filtered EMG signal.
3. RMS EMG.
4. Median frequency over time.
If the --no-plot option is specified, the plots will not be displayed.
Moreover, a text file with the calculated EMG results will be saved as {filename}_results_emg_labiocom.txt.

Prerequisites:
This script requires Python 3 and the following Python libraries: numpy, scipy, matplotlib, os, and argparse.
These can be installed via pip:
pip install numpy scipy matplotlib os argparse

Note:
Proper use of this script requires a solid understanding of signal processing concepts,
particularly as they pertain to EMG signal processing. Improper use can result in inaccurate
or misleading analyses. Therefore, it is recommended that users have adequate knowledge or
seek guidance from a specialist when using this script for EMG signal analysis.
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, detrend
from scipy.signal.windows import hann


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# Funcoes para o filtro de bandpass
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# Funcoes para retificar
def full_wave_rectification(emg_signal):
    emg_detrend = detrend(emg_signal)
    emg_abs = np.abs(emg_detrend)
    return emg_abs


# Funcoes para envelope linear
def linear_envelope(emg_abs, cutoff, fs):
    emg_envelope = butter_lowpass_filter(emg_abs, cutoff, fs)
    time = np.arange(len(emg_abs)) / fs
    signal_integ = np.trapz(emg_envelope, time)
    return emg_envelope, signal_integ


# Funcao para calculo do RMS
def calculate_rms(semg, window_length, overlap):
    start = 0
    rms_values = []

    while start + window_length < len(semg):
        window = semg[start:start+window_length]
        rms_value = np.sqrt(np.mean(window**2))
        rms_values.append(rms_value)
        start += overlap
    return rms_values


# Funcoes para calculo da frequencia mediana
def calculate_median_frequency(semg, fs, window_length, overlap):
    start = 0
    median_freq_values = []

    while start + window_length < len(semg):
        window = semg[start:start+window_length]
        median_freq_value = calculate_median_frequency_for_window(window, fs)
        median_freq_values.append(median_freq_value)
        start += overlap

    return median_freq_values


def calculate_median_frequency_for_window(window, fs):
    nperseg = len(window)
    noverlap = int(nperseg / 2)  # Escolha do overlap
    nfft = 1024  # Escolha do numero de pontos FFT
    freqs, psd = welch(window, fs, window=hann(nperseg), nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    median_freq_idx = np.where(np.cumsum(psd) >= np.sum(psd) / 2)

    if median_freq_idx[0].size > 0:
        median_freq = freqs[median_freq_idx[0][0]]
    else:
        median_freq = np.nan  # ou algum outro valor que indique que a frequencia mediana nao pode ser calculada

    return median_freq


#  Funcao para calculo da curva de grau 2 para verificar tendencias nos resultados
def polynomial_fit(x, y, poly_deg=2):
    poly_coeff = np.polyfit(x, y, poly_deg)
    poly_vals = np.polyval(poly_coeff, x)

    return poly_vals


if __name__ == '__main__':
    # Argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Processa um sinal EMG.')
    parser.add_argument('emg_file', help='O arquivo de sinal EMG para processar.')
    parser.add_argument('fs', type=int, help='A taxa de amostragem do sinal EMG em Hz.')
    parser.add_argument('start_time', type=float, nargs='?', default=0.0, help='O tempo de início para a análise do sinal em segundos.')
    parser.add_argument('end_time', type=float, nargs='?', default=None, help='O tempo final para a análise do sinal em segundos.')
    parser.add_argument('--no-plot', action='store_true', help='Se especificado, não plota os gráficos do sinal EMG.')
    args = parser.parse_args()

    # Carregar sinal
    emg_file = args.emg_file
    fs = args.fs

    # Passando de volts para micro volts µV.s
    emg_signal = np.loadtxt(emg_file)
    emg_signal[:, 1] = emg_signal[:, 1] * 1000000
    time_full = np.arange(len(emg_signal)) / fs

    # Definir início e fim da análise do sinal
    start_time = args.start_time

    # se end_time for None, calcular o fim do sinal com base no seu tamanho
    if args.end_time is None:
        end_time = len(emg_signal) / fs
    else:
        end_time = args.end_time

    start_index = int(start_time * fs)
    end_index = int(end_time * fs)
    emg_signal_cut = emg_signal[start_index:end_index, 1]

    # Filtrar o sinal bandpass
    lowcut = 10.0  # low frequency
    highcut = 450.0  # high frequency
    window_time = 0.25

    emg_filtered = butter_bandpass_filter(emg_signal_cut, lowcut, highcut, fs, order=4)

    # Retificacao completa da onda
    emg_detrend = detrend(emg_filtered)
    emg_abs = np.abs(emg_detrend)
    emg_envelope, signal_integ = linear_envelope(emg_abs, cutoff=10, fs=fs)

    # Calcular RMS
    window_length = int(fs * window_time)  # para uma janela de 250ms
    overlap = int(window_length / 2)  # para uma sobreposição de 125ms
    rms_values = calculate_rms(emg_filtered, window_length, overlap)

    # Calcular a frequencia mediana (MDF)
    window_length = int(fs * window_time)  # para uma janela de window_time ms
    overlap = int(window_length / 2)  # para uma sobreposição de metade window_time ms
    median_freq_values = calculate_median_frequency(emg_filtered, fs, window_length, overlap)

    # Criar vetores de tempo
    time = np.arange(start_index, end_index) / fs
    time_rms = np.arange(start_index, start_index+len(rms_values)*overlap, overlap) / fs

    # Ajustar polinômio de grau 2 ao RMS e plotar
    poly2_rms = polynomial_fit(time_rms, rms_values, 2)

    # Ajustar polinômio de grau 2 ao MDF e plotar
    poly2_mdf = polynomial_fit(time_rms, median_freq_values, 2)

    # Salvando o arquivo de resultado
    # Adicione o sufixo "_results_labiocom" ao nome do arquivo
    base = os.path.basename(args.emg_file)
    filename = os.path.splitext(base)[0]
    output_file = f"{filename}_results_emg_labiocom.txt"

    # Certifique-se de que rms_values e median_freq_values têm o mesmo tamanho
    assert len(rms_values) == len(median_freq_values), \
        "rms_values and median_freq_values must have the same length"

    # Converta as listas em arrays numpy para facilitar a manipulação
    time_array = np.array(time_rms)
    rms_array = np.array(rms_values)
    freq_array = np.array(median_freq_values)

    # Combine os arrays em uma única matriz 2D
    data_matrix = np.vstack((time_array, rms_array, freq_array)).T

    # Gravar dados no arquivo
    with open(output_file, 'w') as f:
        f.write(f"Linear envelope (µVolts.s): {signal_integ}\n")
        f.write("Time(s),RMS(µVolts),Median_Frequency(Hz)\n")
        np.savetxt(f, data_matrix, fmt='%f', delimiter=',')  # Agora a matriz será salva com separação por vírgulas

    print(f"Results written to {output_file}\n Have good studies!")

    if not args.no_plot:
        # Criar Graficos
        # Grafico do sinal bruto e filtrado
        # Cria os subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        # Grafico do sinal bruto
        axs[0].plot(time_full, emg_signal, label='Raw EMG', color='blue', linewidth=1)
        axs[0].set_title('Raw EMG Full-Signal')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('sEMG (µ Volts)')
        axs[0].axis('tight')
        axs[0].grid(True)

        # Grafico do sinal filtrado
        axs[1].plot(time, emg_signal_cut, label='Raw EMG', color='blue', linewidth=3)
        axs[1].plot(time, emg_filtered, label='Filtered EMG', color='red', linewidth=1)
        axs[1].set_title('Cut and Filtered EMG Signal')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('sEMG (µ Volts)')
        axs[1].axis('tight')
        axs[1].legend()
        axs[1].grid(True)

        # Ajusta o layout
        plt.tight_layout()

        # Grafico para Full-Wave Rectified EMG
        plt.figure(2)
        plt.plot(time, emg_abs)
        # Plotando o envelope linear
        plt.plot(time[:len(emg_envelope)], emg_envelope, color='r', linewidth=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Rectified EMG (µ Volts)')
        plt.title(f'FULL-WAVE & LINEAR ENVELOPE = {signal_integ:.1f} µVolts.s')
        plt.grid(True)
        plt.axis('tight')

        # Plotando o RMS
        plt.figure(3)
        plt.plot(time_rms, rms_values)
        plt.plot(time_rms, poly2_rms, color='red', linestyle='--')
        plt.title('EMG - RMS')
        plt.xlabel('Time (s)')
        plt.ylabel('RMS (µ Volts)')
        plt.grid(True)

        # Plotando o MDF
        plt.figure(4)
        plt.plot(time_rms, median_freq_values)
        plt.plot(time_rms, poly2_mdf, color='red', linestyle='--')
        plt.title('EMG - Median Frequency')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.axis('tight')
        plt.grid(True)

        # Plotando o MDF
        plt.figure(5)
        freqs, psd = welch(emg_filtered, fs)

        # Find the index of the maximum y value
        index_max = np.argmax(psd)

        # Find the corresponding x value
        freq_max = freqs[index_max]

        # Plotting the MDF
        plt.plot(freqs, psd)
        plt.title('EMG - PWelch')
        plt.ylabel('PSD (dB/Hz)')
        plt.xlabel('Frequency (Hz)')
        plt.axis('tight')
        plt.grid(True)

        # Add a red dot for the max
        plt.plot(freq_max, psd[index_max], 'ro')

        # Annotate the max value with an arrow
        plt.annotate('Max: {:.2f}, {:.2f}'.format(freq_max, psd[index_max]),
                    xy=(freq_max, psd[index_max]), xycoords='data',
                    xytext=(+10, +30), textcoords='offset points', fontsize=12,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

        plt.show()
