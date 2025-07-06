"""
Practical Communication Systems Project
======================================

This comprehensive project implements real-world communication systems:
- WiFi (IEEE 802.11) System
- LTE/5G Cellular Systems
- Satellite Communication Systems
- Complete protocol stack simulations

Features:
- Real-world system parameters
- Complete frame structures
- Channel models and link budgets
- Performance analysis
- Comparative studies
- Educational explanations and documentation

Author: Communication Systems Learning Project
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import warnings
warnings.filterwarnings('ignore')

class PracticalSystemsProject:
    """
    Practical Communication Systems Implementation
    
    This class provides comprehensive implementations of real-world
    communication systems including WiFi, LTE/5G, and satellite systems.
    """
    
    def __init__(self):
        """Initialize the Practical Systems Project"""
        print("🌐 Practical Communication Systems Project")
        print("==========================================")
        print("\nThis project demonstrates:")
        print("• WiFi (IEEE 802.11) system implementation")
        print("• LTE/5G cellular system simulation")
        print("• Satellite communication systems")
        print("• Real-world performance analysis")
        
    def run_all_demonstrations(self):
        """Run all practical systems demonstrations"""
        demonstrations = [
            ("WiFi (IEEE 802.11) System", self.wifi_system_project),
            ("LTE/5G Cellular System", self.lte_5g_system_project),
            ("Satellite Communication", self.satellite_communication_project)
        ]
        
        for name, demo_func in demonstrations:
            print(f"\n{'='*60}")
            print(f"DEMONSTRATION: {name}")
            print(f"{'='*60}")
            try:
                demo_func()
            except Exception as e:
                print(f"Error in {name}: {e}")
                continue
    
    def wifi_system_project(self):
        """WiFi (IEEE 802.11) system simulation"""
        print("\n📶 WiFi (IEEE 802.11) System Project")
        print("This demonstration shows how WiFi systems work end-to-end")
        
        # WiFi 802.11n/ac parameters
        wifi_params = {
            'center_freq': 2.4e9,  # 2.4 GHz
            'bandwidth': 20e6,     # 20 MHz
            'subcarriers': 52,     # OFDM subcarriers
            'guard_interval': 0.8e-6,  # 0.8 μs
            'symbol_duration': 4e-6,   # 4 μs
            'mimo_streams': 2,     # 2x2 MIMO
            'modulation': '64-QAM'
        }
        
        def generate_wifi_frame():
            """Generate WiFi OFDM frame structure"""
            # Simplified WiFi frame structure
            frame = {
                'preamble': np.ones(16),  # Short preamble
                'header': np.random.randint(0, 2, 48),  # PLCP header
                'payload': np.random.randint(0, 2, 1536)  # Data payload
            }
            return frame
        
        def qam64_modulation(bits):
            """64-QAM modulation"""
            # Convert to proper integer array
            bits = np.array(bits, dtype=int)
            
            if len(bits) % 6 != 0:
                # Pad bits to make divisible by 6
                bits = np.pad(bits, (0, 6 - len(bits) % 6), mode='constant')
            
            # 64-QAM constellation (8x8)
            constellation = []
            for i in range(8):
                for j in range(8):
                    I = 2*i - 7
                    Q = 2*j - 7
                    constellation.append(I + 1j*Q)
            
            constellation = np.array(constellation)
            
            # Map bits to symbols
            symbols = []
            for i in range(0, len(bits), 6):
                bit_group = bits[i:i+6]
                symbol_index = int(''.join(map(str, bit_group)), 2)
                symbols.append(constellation[symbol_index])
            
            return np.array(symbols)
        
        def wifi_ofdm_modulation(data_symbols, n_subcarriers=52):
            """WiFi OFDM modulation"""
            # Pad or truncate to fit subcarriers
            if len(data_symbols) < n_subcarriers:
                data_symbols = np.pad(data_symbols, (0, n_subcarriers - len(data_symbols)), mode='constant')
            else:
                data_symbols = data_symbols[:n_subcarriers]
            
            # Add pilot subcarriers (simplified)
            ofdm_symbol = np.zeros(64, dtype=complex)
            data_indices = np.arange(6, 32)  # Data subcarriers
            pilot_indices = [11, 25, 39, 53]  # Pilot subcarriers
            
            ofdm_symbol[data_indices] = data_symbols[:len(data_indices)]
            for idx in pilot_indices:
                if idx < len(ofdm_symbol):
                    ofdm_symbol[idx] = 1  # Pilot symbols
            
            # IFFT
            time_signal = np.fft.ifft(ofdm_symbol, 64)
            
            # Add cyclic prefix
            cp_length = 16
            cyclic_prefix = time_signal[-cp_length:]
            ofdm_with_cp = np.concatenate([cyclic_prefix, time_signal])
            
            return ofdm_with_cp
        
        def wifi_channel_model(signal, multipath_delay=50e-9):
            """WiFi indoor channel model"""
            # Simplified indoor multipath channel
            # Based on typical WiFi environments
            
            delays = np.array([0, 10e-9, 30e-9, 50e-9, 150e-9]) * 1e6  # Convert to samples
            gains = np.array([1.0, 0.7, 0.5, 0.3, 0.1])
            phases = np.random.uniform(0, 2*np.pi, len(gains))
            
            # Apply multipath
            channel_output = np.zeros_like(signal)
            for delay, gain, phase in zip(delays, gains, phases):
                delay_samples = int(delay)
                if delay_samples < len(signal):
                    shifted_signal = np.zeros_like(signal)
                    shifted_signal[delay_samples:] = signal[:-delay_samples] if delay_samples > 0 else signal
                    channel_output += gain * np.exp(1j * phase) * shifted_signal
            
            return channel_output
        
        print(f"WiFi System Parameters:")
        print(f"  Standard: IEEE 802.11n")
        print(f"  Center frequency: {wifi_params['center_freq']/1e9:.1f} GHz")
        print(f"  Bandwidth: {wifi_params['bandwidth']/1e6:.0f} MHz")
        print(f"  Subcarriers: {wifi_params['subcarriers']}")
        print(f"  Modulation: {wifi_params['modulation']}")
        
        # Generate WiFi frame
        frame = generate_wifi_frame()
        all_bits = np.concatenate([frame['preamble'], frame['header'], frame['payload']])
        
        # 64-QAM modulation
        qam_symbols = qam64_modulation(all_bits)
        print(f"  Generated {len(qam_symbols)} 64-QAM symbols from {len(all_bits)} bits")
        
        # OFDM modulation
        n_ofdm_symbols = len(qam_symbols) // 26  # 26 data subcarriers per OFDM symbol
        ofdm_signal = []
        
        for i in range(n_ofdm_symbols):
            start_idx = i * 26
            end_idx = start_idx + 26
            if end_idx <= len(qam_symbols):
                data_chunk = qam_symbols[start_idx:end_idx]
                ofdm_sym = wifi_ofdm_modulation(data_chunk)
                ofdm_signal.extend(ofdm_sym)
        
        ofdm_signal = np.array(ofdm_signal)
        
        # Channel effects
        received_signal = wifi_channel_model(ofdm_signal)
        
        # Add noise
        snr_db = 25  # Typical WiFi SNR
        snr_linear = 10**(snr_db/10)
        noise_std = np.sqrt(1/(2*snr_linear))
        noise = noise_std * (np.random.randn(len(received_signal)) + 1j*np.random.randn(len(received_signal)))
        received_signal += noise
        
        # Calculate throughput
        symbol_rate = 1 / (4e-6 + 0.8e-6)  # Symbol duration + GI
        bits_per_symbol = 6 * 26  # 64-QAM * 26 data subcarriers
        throughput_mbps = symbol_rate * bits_per_symbol / 1e6
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.scatter(qam_symbols.real, qam_symbols.imag, alpha=0.7, s=30)
        plt.title('64-QAM Constellation')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.subplot(2, 3, 2)
        plt.plot(np.real(ofdm_signal[:320]))  # First 4 OFDM symbols
        plt.title('OFDM Time Domain Signal')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        freq_spectrum = np.fft.fftshift(np.fft.fft(ofdm_signal, 1024))
        frequencies = np.fft.fftshift(np.fft.fftfreq(1024, 1/20e6))
        plt.plot(frequencies/1e6, 20*np.log10(np.abs(freq_spectrum)))
        plt.title('WiFi OFDM Spectrum')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 4)
        plt.scatter(received_signal.real, received_signal.imag, alpha=0.3, s=10)
        plt.title('Received Signal Constellation')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.subplot(2, 3, 5)
        # Frame structure visualization
        frame_structure = ['Preamble', 'Header', 'Payload']
        frame_lengths = [len(frame['preamble']), len(frame['header']), len(frame['payload'])]
        plt.bar(frame_structure, frame_lengths)
        plt.title('WiFi Frame Structure')
        plt.ylabel('Bits')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 6)
        # WiFi standards comparison
        wifi_standards = ['802.11a', '802.11g', '802.11n', '802.11ac', '802.11ax']
        max_rates = [54, 54, 600, 6933, 9608]  # Mbps
        plt.bar(wifi_standards, max_rates)
        plt.title('WiFi Standard Data Rates')
        plt.ylabel('Max Rate (Mbps)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\\nPerformance Results:")
        print(f"  Theoretical throughput: {throughput_mbps:.1f} Mbps")
        print(f"  OFDM symbols generated: {n_ofdm_symbols}")
        print(f"  Frame efficiency: {len(frame['payload'])/len(all_bits)*100:.1f}%")
        print(f"  SNR: {snr_db} dB")
        
        return {
            'throughput_mbps': throughput_mbps,
            'frame_efficiency': len(frame['payload'])/len(all_bits)*100,
            'n_ofdm_symbols': n_ofdm_symbols,
            'snr_db': snr_db
        }
    
    def lte_5g_system_project(self):
        """LTE/5G cellular system simulation"""
        print("\n📱 LTE/5G Cellular System Project")
        print("This demonstration shows how modern cellular systems work")
        
        # LTE parameters
        lte_params = {
            'center_freq': 1.9e9,  # 1.9 GHz (Band 2)
            'bandwidth': 20e6,     # 20 MHz
            'subcarriers': 1200,   # 12 subcarriers per RB * 100 RBs
            'cp_length': 4.69e-6,  # Normal CP
            'symbol_duration': 66.7e-6,  # OFDM symbol duration
            'mimo_layers': 4,      # 4x4 MIMO
            'modulation': '256-QAM'
        }
        
        # 5G NR parameters
        nr_params = {
            'center_freq': 3.5e9,  # 3.5 GHz (n78)
            'bandwidth': 100e6,    # 100 MHz
            'subcarrier_spacing': 30e3,  # 30 kHz
            'mimo_layers': 8,      # 8x8 MIMO
            'modulation': '256-QAM',
            'beamforming': True
        }
        
        def generate_lte_resource_grid():
            """Generate LTE resource grid"""
            # LTE resource grid: 12 subcarriers x 14 OFDM symbols
            n_subcarriers = 12
            n_symbols = 14
            
            # Generate resource elements
            resource_grid = np.zeros((n_subcarriers, n_symbols), dtype=complex)
            
            # Reference signals (pilots)
            pilot_positions = [(0, 0), (0, 4), (0, 7), (0, 11)]  # Simplified
            for pos in pilot_positions:
                resource_grid[pos] = 1 + 1j
            
            # Data symbols (QPSK for control, 64-QAM for data)
            data_positions = [(i, j) for i in range(n_subcarriers) for j in range(n_symbols) 
                             if (i, j) not in pilot_positions]
            
            for pos in data_positions:
                # Simulate 64-QAM symbol
                I = np.random.choice([-7, -5, -3, -1, 1, 3, 5, 7])
                Q = np.random.choice([-7, -5, -3, -1, 1, 3, 5, 7])
                resource_grid[pos] = I + 1j*Q
            
            return resource_grid
        
        def lte_ofdma_modulation(resource_grid):
            """LTE OFDMA modulation"""
            n_subcarriers, n_symbols = resource_grid.shape
            
            # Extend to full FFT size (e.g., 2048 for 20 MHz)
            fft_size = 2048
            ofdm_symbols = []
            
            for symbol_idx in range(n_symbols):
                # Get frequency domain symbol
                freq_symbol = np.zeros(fft_size, dtype=complex)
                
                # Place resource grid in center
                start_idx = (fft_size - n_subcarriers) // 2
                freq_symbol[start_idx:start_idx+n_subcarriers] = resource_grid[:, symbol_idx]
                
                # IFFT
                time_symbol = np.fft.ifft(freq_symbol)
                
                # Add cyclic prefix
                cp_length = fft_size // 4  # Simplified
                cyclic_prefix = time_symbol[-cp_length:]
                symbol_with_cp = np.concatenate([cyclic_prefix, time_symbol])
                
                ofdm_symbols.append(symbol_with_cp)
            
            return np.concatenate(ofdm_symbols)
        
        def lte_mimo_precoding(signal, n_layers=4):
            """LTE MIMO precoding"""
            # Simplified precoding matrix
            precoding_matrix = np.random.randn(n_layers, n_layers) + 1j*np.random.randn(n_layers, n_layers)
            precoding_matrix = precoding_matrix / np.sqrt(n_layers)
            
            # Apply precoding (simplified)
            signal_matrix = signal.reshape(-1, 1)
            precoded_signals = []
            
            for layer in range(n_layers):
                precoded_signal = signal_matrix.flatten() * precoding_matrix[layer, 0]
                precoded_signals.append(precoded_signal)
            
            return np.array(precoded_signals)
        
        def cellular_channel_model(signal, scenario='urban'):
            """Cellular channel model"""
            if scenario == 'urban':
                # Urban multipath model
                delays = np.array([0, 0.1, 0.3, 0.5, 1.0, 2.0]) * 1e-6  # microseconds
                gains = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
            elif scenario == 'rural':
                # Rural model (fewer paths)
                delays = np.array([0, 0.5, 1.5]) * 1e-6
                gains = np.array([1.0, 0.5, 0.2])
            else:
                # Default model
                delays = np.array([0, 0.2, 0.8]) * 1e-6
                gains = np.array([1.0, 0.6, 0.3])
            
            # Apply multipath
            channel_output = np.zeros_like(signal)
            for delay, gain in zip(delays, gains):
                delay_samples = int(delay * 1e6)  # Assuming 1 MHz sampling
                if delay_samples < len(signal):
                    shifted_signal = np.zeros_like(signal)
                    shifted_signal[delay_samples:] = signal[:-delay_samples] if delay_samples > 0 else signal
                    channel_output += gain * shifted_signal
            
            return channel_output
        
        print(f"LTE System Parameters:")
        print(f"  Center frequency: {lte_params['center_freq']/1e9:.1f} GHz")
        print(f"  Bandwidth: {lte_params['bandwidth']/1e6:.0f} MHz")
        print(f"  Subcarriers: {lte_params['subcarriers']}")
        print(f"  MIMO layers: {lte_params['mimo_layers']}")
        
        print(f"\\n5G NR System Parameters:")
        print(f"  Center frequency: {nr_params['center_freq']/1e9:.1f} GHz")
        print(f"  Bandwidth: {nr_params['bandwidth']/1e6:.0f} MHz")
        print(f"  Subcarrier spacing: {nr_params['subcarrier_spacing']/1e3:.0f} kHz")
        print(f"  MIMO layers: {nr_params['mimo_layers']}")
        print(f"  Beamforming: {nr_params['beamforming']}")
        
        # Generate LTE resource grid
        resource_grid = generate_lte_resource_grid()
        
        # OFDMA modulation
        lte_signal = lte_ofdma_modulation(resource_grid)
        
        # MIMO precoding
        mimo_signals = lte_mimo_precoding(lte_signal, lte_params['mimo_layers'])
        
        # Channel effects (use first MIMO layer for visualization)
        received_signal = cellular_channel_model(mimo_signals[0], 'urban')
        
        # Add noise
        snr_db = 20  # Typical cellular SNR
        snr_linear = 10**(snr_db/10)
        noise_std = np.sqrt(1/(2*snr_linear))
        noise = noise_std * (np.random.randn(len(received_signal)) + 1j*np.random.randn(len(received_signal)))
        received_signal += noise
        
        # Calculate spectral efficiency
        bits_per_symbol = 8  # 256-QAM
        subcarrier_spacing = 15e3  # 15 kHz
        spectral_efficiency = bits_per_symbol * subcarrier_spacing * lte_params['subcarriers'] / lte_params['bandwidth']
        
        # Plot results
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 3, 1)
        plt.imshow(np.abs(resource_grid), cmap='viridis', aspect='auto')
        plt.title('LTE Resource Grid')
        plt.xlabel('OFDM Symbol')
        plt.ylabel('Subcarrier')
        plt.colorbar(label='Magnitude')
        
        plt.subplot(3, 3, 2)
        plt.plot(np.real(lte_signal[:1000]))
        plt.title('LTE OFDMA Signal')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 3)
        # LTE vs 5G comparison
        technologies = ['LTE', '5G NR']
        max_rates = [1000, 10000]  # Mbps (theoretical)
        plt.bar(technologies, max_rates)
        plt.title('LTE vs 5G Peak Rates')
        plt.ylabel('Peak Rate (Mbps)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 4)
        freq_spectrum = np.fft.fftshift(np.fft.fft(lte_signal, 4096))
        frequencies = np.fft.fftshift(np.fft.fftfreq(4096, 1/20e6))
        plt.plot(frequencies/1e6, 20*np.log10(np.abs(freq_spectrum)))
        plt.title('LTE OFDMA Spectrum')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 5)
        plt.scatter(received_signal.real, received_signal.imag, alpha=0.3, s=10)
        plt.title('Received Constellation')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.subplot(3, 3, 6)
        # MIMO layer visualization
        mimo_layers = np.arange(1, 9)
        mimo_capacity = mimo_layers * np.log2(1 + 10**(snr_db/10))
        plt.plot(mimo_layers, mimo_capacity, 'o-')
        plt.title('MIMO Capacity vs Layers')
        plt.xlabel('Number of Layers')
        plt.ylabel('Capacity (bits/s/Hz)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 7)
        # 5G use cases
        use_cases = ['eMBB', 'URLLC', 'mMTC']
        latency = [10, 1, 100]  # ms
        plt.bar(use_cases, latency)
        plt.title('5G Use Case Latency')
        plt.ylabel('Latency (ms)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 8)
        # Frequency bands
        bands = ['Sub-1GHz', '1-6GHz', 'mmWave']
        coverage = [100, 10, 1]  # km
        plt.bar(bands, coverage)
        plt.title('5G Coverage by Band')
        plt.ylabel('Coverage (km)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 9)
        # Beamforming pattern (simplified)
        angles = np.linspace(-90, 90, 181)
        beam_pattern = np.sinc(2 * np.sin(np.deg2rad(angles)))**2
        plt.plot(angles, 20*np.log10(beam_pattern))
        plt.title('5G Beamforming Pattern')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Gain (dB)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\\nPerformance Results:")
        print(f"  LTE spectral efficiency: {spectral_efficiency:.2f} bits/s/Hz")
        print(f"  Resource grid size: {resource_grid.shape}")
        print(f"  SNR: {snr_db} dB")
        
        print(f"\\nKey Differences LTE vs 5G:")
        print(f"  Latency: LTE ~10ms, 5G ~1ms")
        print(f"  Peak rate: LTE ~1Gbps, 5G ~10Gbps")
        print(f"  Efficiency: 5G ~3x better than LTE")
        print(f"  Beamforming: 5G has advanced beamforming")
        
        return {
            'spectral_efficiency': spectral_efficiency,
            'resource_grid_shape': resource_grid.shape,
            'snr_db': snr_db,
            'mimo_layers': lte_params['mimo_layers']
        }
    
    def satellite_communication_project(self):
        """Satellite communication system simulation"""
        print("\n🛰️ Satellite Communication System Project")
        print("This demonstration shows how satellite communication systems work")
        
        # Satellite parameters
        satellite_params = {
            'orbit_altitude': 35786e3,  # GEO altitude (km)
            'frequency_uplink': 14e9,   # 14 GHz (Ku-band)
            'frequency_downlink': 12e9, # 12 GHz (Ku-band)
            'dish_diameter': 1.2,       # 1.2m dish
            'modulation': 'QPSK',
            'coding_rate': 3/4
        }
        
        def calculate_link_budget():
            """Calculate satellite link budget"""
            # Satellite link parameters
            satellite_power = 100  # W
            satellite_gain = 40    # dBi
            dish_gain = 42         # dBi (1.2m dish)
            frequency = satellite_params['frequency_downlink']
            distance = satellite_params['orbit_altitude']
            
            # Free space path loss
            fspl = 20 * np.log10(distance) + 20 * np.log10(frequency) - 147.55
            
            # Received power
            eirp = 10 * np.log10(satellite_power) + satellite_gain
            received_power = eirp + dish_gain - fspl
            
            # Noise calculation
            noise_temp = 150  # K (system noise temperature)
            bandwidth = 36e6  # 36 MHz
            noise_power = -228.6 + 10*np.log10(noise_temp) + 10*np.log10(bandwidth)
            
            # C/N ratio
            cn_ratio = received_power - noise_power
            
            return {
                'fspl': fspl,
                'received_power': received_power,
                'noise_power': noise_power,
                'cn_ratio': cn_ratio,
                'eirp': eirp
            }
        
        def satellite_channel_model(signal, elevation_angle=45):
            """Satellite channel model including atmospheric effects"""
            # Atmospheric attenuation
            if elevation_angle < 10:
                attenuation_db = 2.5  # High attenuation at low elevation
            elif elevation_angle < 30:
                attenuation_db = 1.0  # Medium attenuation
            else:
                attenuation_db = 0.3  # Low attenuation
            
            attenuation_linear = 10**(-attenuation_db/10)
            
            # Doppler shift (simplified)
            doppler_freq = 100  # Hz (typical for GEO)
            doppler_phase = 2 * np.pi * doppler_freq * np.arange(len(signal)) / 1e6
            doppler_effect = np.exp(1j * doppler_phase)
            
            # Apply channel effects
            channel_output = signal * attenuation_linear * doppler_effect
            
            return channel_output, attenuation_db
        
        def dvb_s2_framing():
            """DVB-S2 frame structure simulation"""
            # DVB-S2 frame parameters
            frame_length = 64800  # bits (normal frame)
            payload_length = 48600  # bits (after coding)
            
            # Generate frame structure
            frame = {
                'sync_pattern': np.ones(90),  # SOF + PLSC
                'pilot_symbols': np.ones(36),  # Pilot symbols
                'payload_data': np.random.randint(0, 2, payload_length),
                'fec_parity': np.random.randint(0, 2, frame_length - payload_length - 90 - 36)
            }
            
            return frame
        
        def satellite_qpsk_modulation(bits):
            """Satellite QPSK modulation with pulse shaping"""
            # Group bits into pairs
            if len(bits) % 2 != 0:
                bits = np.append(bits, 0)
            
            symbols = []
            for i in range(0, len(bits), 2):
                bit_pair = bits[i:i+2]
                if np.array_equal(bit_pair, [0, 0]):
                    symbols.append(1+1j)
                elif np.array_equal(bit_pair, [0, 1]):
                    symbols.append(-1+1j)
                elif np.array_equal(bit_pair, [1, 0]):
                    symbols.append(1-1j)
                else:
                    symbols.append(-1-1j)
            
            symbols = np.array(symbols)
            
            # Root-raised cosine pulse shaping (simplified)
            samples_per_symbol = 4
            upsampled = np.zeros(len(symbols) * samples_per_symbol, dtype=complex)
            upsampled[::samples_per_symbol] = symbols
            
            # Simple pulse shaping filter
            pulse_length = 16
            pulse = np.sinc(np.arange(-pulse_length//2, pulse_length//2))
            shaped_signal = np.convolve(upsampled, pulse, mode='same')
            
            return shaped_signal, symbols
        
        print(f"Satellite System Parameters:")
        print(f"  Orbit: Geostationary (GEO)")
        print(f"  Altitude: {satellite_params['orbit_altitude']/1e3:.0f} km")
        print(f"  Uplink frequency: {satellite_params['frequency_uplink']/1e9:.1f} GHz")
        print(f"  Downlink frequency: {satellite_params['frequency_downlink']/1e9:.1f} GHz")
        print(f"  Dish diameter: {satellite_params['dish_diameter']} m")
        print(f"  Modulation: {satellite_params['modulation']}")
        
        # Calculate link budget
        link_budget = calculate_link_budget()
        
        # Generate DVB-S2 frame
        frame = dvb_s2_framing()
        all_bits = np.concatenate([frame['sync_pattern'], frame['pilot_symbols'], 
                                  frame['payload_data'], frame['fec_parity']])
        
        # QPSK modulation
        modulated_signal, qpsk_symbols = satellite_qpsk_modulation(all_bits)
        
        # Satellite channel
        received_signal, attenuation = satellite_channel_model(modulated_signal, elevation_angle=30)
        
        # Add noise based on link budget
        snr_db = link_budget['cn_ratio'] - 10*np.log10(satellite_params['coding_rate'])  # Account for coding gain
        snr_linear = 10**(snr_db/10)
        noise_std = np.sqrt(1/(2*snr_linear))
        noise = noise_std * (np.random.randn(len(received_signal)) + 1j*np.random.randn(len(received_signal)))
        received_signal += noise
        
        # Calculate orbital parameters
        orbital_period = 2 * np.pi * np.sqrt(satellite_params['orbit_altitude']**3 / (3.986e14))  # seconds
        orbital_velocity = 2 * np.pi * satellite_params['orbit_altitude'] / orbital_period  # m/s
        
        # Calculate coverage
        earth_radius = 6371e3  # km
        coverage_angle = np.arccos(earth_radius / (earth_radius + satellite_params['orbit_altitude']))
        coverage_radius = earth_radius * np.sin(coverage_angle)
        
        # Plot results
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 3, 1)
        plt.scatter(qpsk_symbols.real, qpsk_symbols.imag, alpha=0.7, s=30)
        plt.title('Satellite QPSK Constellation')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.subplot(3, 3, 2)
        plt.plot(np.real(modulated_signal[:1000]))
        plt.title('Pulse-Shaped QPSK Signal')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 3)
        # Link budget visualization
        budget_components = ['EIRP', 'FSPL', 'Antenna Gain', 'Noise']
        values = [link_budget['eirp'], -link_budget['fspl'], 42, link_budget['noise_power']]
        colors = ['green', 'red', 'blue', 'orange']
        plt.bar(budget_components, values, color=colors)
        plt.title('Satellite Link Budget')
        plt.ylabel('Power (dBm)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 4)
        plt.scatter(received_signal.real, received_signal.imag, alpha=0.3, s=10)
        plt.title('Received Constellation')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.subplot(3, 3, 5)
        # Satellite orbits comparison
        orbits = ['LEO', 'MEO', 'GEO']
        altitudes = [500, 10000, 35786]
        plt.bar(orbits, altitudes)
        plt.title('Satellite Orbit Altitudes')
        plt.ylabel('Altitude (km)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 6)
        # Frequency bands
        bands = ['C-band', 'Ku-band', 'Ka-band']
        frequencies = [6, 14, 30]  # GHz
        plt.bar(bands, frequencies)
        plt.title('Satellite Frequency Bands')
        plt.ylabel('Frequency (GHz)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 7)
        # Elevation angle vs signal quality
        elevations = np.arange(5, 91, 5)
        signal_quality = []
        for elev in elevations:
            if elev < 10:
                quality = 50
            elif elev < 30:
                quality = 75
            else:
                quality = 90
            signal_quality.append(quality)
        
        plt.plot(elevations, signal_quality, 'o-')
        plt.title('Signal Quality vs Elevation')
        plt.xlabel('Elevation Angle (degrees)')
        plt.ylabel('Signal Quality (%)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 8)
        # Frame structure
        frame_parts = ['Sync', 'Pilot', 'Payload', 'FEC']
        frame_sizes = [len(frame['sync_pattern']), len(frame['pilot_symbols']), 
                      len(frame['payload_data']), len(frame['fec_parity'])]
        plt.pie(frame_sizes, labels=frame_parts, autopct='%1.1f%%')
        plt.title('DVB-S2 Frame Structure')
        
        plt.subplot(3, 3, 9)
        # Coverage area visualization
        theta = np.linspace(0, 2*np.pi, 100)
        earth_circle = earth_radius/1e3 * np.exp(1j * theta)
        coverage_circle = coverage_radius/1e3 * np.exp(1j * theta)
        
        plt.plot(earth_circle.real, earth_circle.imag, 'b-', label='Earth')
        plt.plot(coverage_circle.real, coverage_circle.imag, 'r--', label='Coverage')
        plt.plot(0, satellite_params['orbit_altitude']/1e3, 'rs', markersize=10, label='Satellite')
        plt.title('GEO Satellite Coverage')
        plt.xlabel('Distance (1000 km)')
        plt.ylabel('Distance (1000 km)')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\\nOrbital Parameters:")
        print(f"  Orbital period: {orbital_period/3600:.1f} hours")
        print(f"  Orbital velocity: {orbital_velocity/1000:.2f} km/s")
        print(f"  Coverage radius: {coverage_radius/1e3:.0f} km")
        
        print(f"\\nLink Budget Results:")
        print(f"  EIRP: {link_budget['eirp']:.1f} dBW")
        print(f"  Free space path loss: {link_budget['fspl']:.1f} dB")
        print(f"  Received power: {link_budget['received_power']:.1f} dBm")
        print(f"  C/N ratio: {link_budget['cn_ratio']:.1f} dB")
        print(f"  Atmospheric attenuation: {attenuation:.1f} dB")
        
        print(f"\\nDVB-S2 Frame Analysis:")
        print(f"  Total frame length: {len(all_bits)} bits")
        print(f"  Payload efficiency: {len(frame['payload_data'])/len(all_bits)*100:.1f}%")
        print(f"  Coding rate: {satellite_params['coding_rate']:.2f}")
        
        return {
            'link_budget': link_budget,
            'orbital_period_hours': orbital_period/3600,
            'coverage_radius_km': coverage_radius/1e3,
            'frame_efficiency': len(frame['payload_data'])/len(all_bits)*100,
            'attenuation_db': attenuation
        }

def interactive_menu():
    """Interactive menu for running demonstrations"""
    project = PracticalSystemsProject()
    
    while True:
        print("\\n" + "="*50)
        print("🌐 PRACTICAL SYSTEMS PROJECT MENU")
        print("="*50)
        print("1. WiFi (IEEE 802.11) System")
        print("2. LTE/5G Cellular System")
        print("3. Satellite Communication")
        print("4. Run All Demonstrations")
        print("5. Exit")
        
        try:
            choice = input("\\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                project.wifi_system_project()
            elif choice == '2':
                project.lte_5g_system_project()
            elif choice == '3':
                project.satellite_communication_project()
            elif choice == '4':
                project.run_all_demonstrations()
            elif choice == '5':
                print("\\nExiting Practical Systems Project. Goodbye!")
                break
            else:
                print("\\nInvalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\\n\\nExiting Practical Systems Project. Goodbye!")
            break
        except Exception as e:
            print(f"\\nError: {e}")
            print("Please try again.")

if __name__ == "__main__":
    print(__doc__)
    interactive_menu()
