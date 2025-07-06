"""
Channel Effects Simulation Project
=================================

This comprehensive project simulates various channel effects in communication systems:
- Multipath Fading (Rayleigh and Rician)
- Doppler Effects and Frequency Shifts
- Frequency Selective Fading
- Impact on Digital Modulation

Features:
- Realistic channel models
- Fading coefficient generation
- Visual analysis of channel effects
- Performance impact assessment
- Educational explanations and documentation

Author: Communication Systems Learning Project
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import warnings
warnings.filterwarnings('ignore')

class ChannelEffectsProject:
    """
    Channel Effects Simulation Implementation
    
    This class provides comprehensive simulations of various channel effects
    that occur in wireless communication systems.
    """
    
    def __init__(self):
        """Initialize the Channel Effects Project"""
        print("🌊 Channel Effects Simulation Project")
        print("====================================")
        print("\nThis project demonstrates:")
        print("• Multipath fading (Rayleigh and Rician)")
        print("• Doppler effects and frequency shifts")
        print("• Frequency selective fading")
        print("• Impact on digital communication")
        
    def run_all_demonstrations(self):
        """Run all channel effects demonstrations"""
        demonstrations = [
            ("Multipath Fading Analysis", self.multipath_fading_project),
            ("Doppler Effects Simulation", self.doppler_effects_project),
            ("Frequency Selective Fading", self.frequency_selective_fading_project)
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
    
    def multipath_fading_project(self):
        """Rayleigh and Rician fading simulation"""
        print("\n🌟 Multipath Fading Project")
        print("This demonstration shows how multipath propagation affects signals")
        
        def rayleigh_fading(N, fd, fs):
            """Generate Rayleigh fading coefficients"""
            # Jakes' model implementation
            N0 = 4 * fd / fs  # Normalized Doppler frequency
            
            # Generate complex Gaussian random variables
            x = np.random.randn(N) + 1j * np.random.randn(N)
            
            # Apply Doppler filter (simplified)
            if N0 > 0:
                # Simple first-order filter
                alpha = np.exp(-2 * np.pi * N0)
                h = np.zeros(N, dtype=complex)
                h[0] = x[0]
                
                for i in range(1, N):
                    h[i] = alpha * h[i-1] + np.sqrt(1 - alpha**2) * x[i]
            else:
                h = x
            
            return h / np.sqrt(np.mean(np.abs(h)**2))  # Normalize
        
        def rician_fading(N, K_factor, fd, fs):
            """Generate Rician fading coefficients"""
            # K-factor in linear scale
            K_linear = 10**(K_factor/10)
            
            # Generate Rayleigh component
            rayleigh_comp = rayleigh_fading(N, fd, fs)
            
            # Add LOS component
            los_comp = np.sqrt(K_linear / (K_linear + 1))
            scattered_comp = np.sqrt(1 / (K_linear + 1)) * rayleigh_comp
            
            rician_comp = los_comp + scattered_comp
            
            return rician_comp
        
        # Parameters
        N = 10000  # Number of samples
        fd = 100   # Doppler frequency (Hz)
        fs = 1000  # Sampling frequency (Hz)
        
        print(f"Simulation Parameters:")
        print(f"  Samples: {N}")
        print(f"  Doppler frequency: {fd} Hz")
        print(f"  Sampling frequency: {fs} Hz")
        
        # Generate fading coefficients
        rayleigh_h = rayleigh_fading(N, fd, fs)
        rician_h = rician_fading(N, 10, fd, fs)  # K = 10 dB
        
        # Generate QPSK symbols
        data_bits = np.random.randint(0, 2, N*2)
        qpsk_symbols = []
        for i in range(0, len(data_bits), 2):
            bit_pair = data_bits[i:i+2]
            if np.array_equal(bit_pair, [0, 0]):
                qpsk_symbols.append(1+1j)
            elif np.array_equal(bit_pair, [0, 1]):
                qpsk_symbols.append(-1+1j)
            elif np.array_equal(bit_pair, [1, 0]):
                qpsk_symbols.append(1-1j)
            else:
                qpsk_symbols.append(-1-1j)
        
        qpsk_symbols = np.array(qpsk_symbols)
        
        # Apply fading
        rayleigh_received = rayleigh_h * qpsk_symbols
        rician_received = rician_h * qpsk_symbols
        
        # Add AWGN
        snr_db = 15
        snr_linear = 10**(snr_db/10)
        noise_std = np.sqrt(1/(2*snr_linear))
        
        noise_ray = noise_std * (np.random.randn(N) + 1j*np.random.randn(N))
        noise_ric = noise_std * (np.random.randn(N) + 1j*np.random.randn(N))
        
        rayleigh_received += noise_ray
        rician_received += noise_ric
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Fading coefficient magnitude
        plt.subplot(2, 3, 1)
        plt.plot(20*np.log10(np.abs(rayleigh_h[:500])), label='Rayleigh')
        plt.plot(20*np.log10(np.abs(rician_h[:500])), label='Rician (K=10dB)')
        plt.xlabel('Sample')
        plt.ylabel('Magnitude (dB)')
        plt.title('Fading Coefficient Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Fading coefficient phase
        plt.subplot(2, 3, 2)
        plt.plot(np.angle(rayleigh_h[:500]), label='Rayleigh')
        plt.plot(np.angle(rician_h[:500]), label='Rician')
        plt.xlabel('Sample')
        plt.ylabel('Phase (radians)')
        plt.title('Fading Coefficient Phase')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Magnitude distribution
        plt.subplot(2, 3, 3)
        plt.hist(np.abs(rayleigh_h), bins=50, alpha=0.7, density=True, label='Rayleigh')
        plt.hist(np.abs(rician_h), bins=50, alpha=0.7, density=True, label='Rician')
        plt.xlabel('Magnitude')
        plt.ylabel('Probability Density')
        plt.title('Magnitude Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Constellation - AWGN only
        plt.subplot(2, 3, 4)
        awgn_only = qpsk_symbols + noise_ray
        plt.scatter(awgn_only.real, awgn_only.imag, alpha=0.3, s=10)
        plt.title('AWGN Channel')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Constellation - Rayleigh fading
        plt.subplot(2, 3, 5)
        plt.scatter(rayleigh_received.real, rayleigh_received.imag, alpha=0.3, s=10)
        plt.title('Rayleigh Fading Channel')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Constellation - Rician fading
        plt.subplot(2, 3, 6)
        plt.scatter(rician_received.real, rician_received.imag, alpha=0.3, s=10)
        plt.title('Rician Fading Channel')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate statistics
        coherence_time = 1 / (2 * fd)
        
        print(f"\\nFading Statistics:")
        print(f"  Rayleigh - Mean magnitude: {np.mean(np.abs(rayleigh_h)):.3f}")
        print(f"  Rician - Mean magnitude: {np.mean(np.abs(rician_h)):.3f}")
        print(f"  Coherence time: {coherence_time*1000:.1f} ms")
        print(f"  Fade duration: {1/fd*1000:.1f} ms")
        
        return {
            'rayleigh_mean': np.mean(np.abs(rayleigh_h)),
            'rician_mean': np.mean(np.abs(rician_h)),
            'coherence_time': coherence_time,
            'doppler_freq': fd
        }
    
    def doppler_effects_project(self):
        """Doppler effects simulation"""
        print("\n🚗 Doppler Effects Project")
        print("This demonstration shows how mobile velocity affects signal frequency")
        
        def doppler_shift(signal, fd, fs):
            """Apply Doppler shift to signal"""
            N = len(signal)
            t = np.arange(N) / fs
            
            # Doppler shift as frequency offset
            doppler_factor = np.exp(1j * 2 * np.pi * fd * t)
            
            return signal * doppler_factor
        
        # Parameters
        fc = 2.4e9  # Carrier frequency (2.4 GHz)
        fs = 1e6    # Sampling frequency (1 MHz)
        v_max = 120 # Maximum velocity (km/h)
        
        # Convert velocity to Doppler frequency
        fd_max = v_max * 1000 / 3600 * fc / 3e8  # Maximum Doppler frequency
        
        print(f"System Parameters:")
        print(f"  Carrier frequency: {fc/1e9:.1f} GHz")
        print(f"  Maximum velocity: {v_max} km/h")
        print(f"  Maximum Doppler frequency: {fd_max:.1f} Hz")
        
        # Generate test signal (OFDM-like)
        N = 1000
        test_signal = np.random.randn(N) + 1j * np.random.randn(N)
        
        # Apply different Doppler shifts
        velocities = [0, 30, 60, 120]  # km/h
        
        plt.figure(figsize=(15, 10))
        
        doppler_results = []
        for i, v in enumerate(velocities):
            fd = v * 1000 / 3600 * fc / 3e8
            
            # Apply Doppler shift
            doppler_signal = doppler_shift(test_signal, fd, fs)
            
            # Calculate frequency spectrum
            freq_spectrum = np.fft.fftshift(np.fft.fft(doppler_signal, 2048))
            frequencies = np.fft.fftshift(np.fft.fftfreq(2048, 1/fs))
            
            plt.subplot(2, 2, i+1)
            plt.plot(frequencies/1000, 20*np.log10(np.abs(freq_spectrum)))
            plt.title(f'Doppler Shift: {v} km/h (fd = {fd:.1f} Hz)')
            plt.xlabel('Frequency (kHz)')
            plt.ylabel('Magnitude (dB)')
            plt.grid(True, alpha=0.3)
            plt.xlim([-100, 100])
            
            doppler_results.append({
                'velocity': v,
                'doppler_freq': fd,
                'freq_spectrum': freq_spectrum,
                'frequencies': frequencies
            })
        
        plt.tight_layout()
        plt.show()
        
        # Simulate Doppler spread effect on OFDM
        print("\\nDoppler spread effect on OFDM:")
        
        # OFDM parameters
        N_fft = 64
        subcarrier_spacing = 15e3  # 15 kHz (LTE-like)
        symbol_duration = 1 / subcarrier_spacing
        
        # Calculate coherence time
        coherence_time = 1 / (2 * fd_max)
        
        print(f"  OFDM symbol duration: {symbol_duration*1000:.2f} ms")
        print(f"  Coherence time: {coherence_time*1000:.2f} ms")
        
        if symbol_duration < coherence_time:
            print("  ✓ OFDM symbol duration < coherence time (Good)")
            impact = "Minimal Inter-Carrier Interference (ICI)"
        else:
            print("  ✗ OFDM symbol duration > coherence time (ICI expected)")
            impact = "Significant Inter-Carrier Interference (ICI)"
        
        print(f"  Impact: {impact}")
        
        return {
            'max_doppler': fd_max,
            'coherence_time': coherence_time,
            'symbol_duration': symbol_duration,
            'ici_impact': impact,
            'results': doppler_results
        }
    
    def frequency_selective_fading_project(self):
        """Frequency selective fading simulation"""
        print("\n📡 Frequency Selective Fading Project")
        print("This demonstration shows how multipath delay spread affects signals")
        
        def generate_multipath_channel(delays, gains, N):
            """Generate multipath channel impulse response"""
            # Convert delays to sample indices
            max_delay = max(delays)
            h = np.zeros(N, dtype=complex)
            
            for delay, gain in zip(delays, gains):
                delay_samples = int(delay * 1e6)  # Assume 1 MHz sampling
                if delay_samples < N:
                    # Add complex gain with random phase
                    phase = np.random.uniform(0, 2*np.pi)
                    h[delay_samples] = gain * np.exp(1j * phase)
            
            return h
        
        # Define multipath profiles
        channels = {
            'LOS': {
                'delays': [0],  # microseconds
                'gains': [1.0],
                'name': 'Line of Sight',
                'environment': 'Indoor/Rural'
            },
            'Urban': {
                'delays': [0, 0.5, 1.0, 2.0, 5.0],  # microseconds
                'gains': [1.0, 0.7, 0.5, 0.3, 0.1],
                'name': 'Urban Environment',
                'environment': 'Dense urban with tall buildings'
            },
            'Suburban': {
                'delays': [0, 1.0, 3.0, 7.0],  # microseconds
                'gains': [1.0, 0.5, 0.3, 0.1],
                'name': 'Suburban Environment',
                'environment': 'Residential areas'
            }
        }
        
        # Parameters
        N_channel = 100  # Channel impulse response length
        N_fft = 64       # FFT size for frequency response
        
        plt.figure(figsize=(15, 12))
        
        channel_results = []
        for idx, (channel_type, params) in enumerate(channels.items()):
            print(f"\\nAnalyzing {params['name']} ({params['environment']}):")
            
            # Generate channel impulse response
            h = generate_multipath_channel(params['delays'], params['gains'], N_channel)
            
            # Calculate frequency response
            H = np.fft.fft(h, N_fft)
            frequencies = np.arange(N_fft) / N_fft
            
            # Plot impulse response
            plt.subplot(3, 3, idx*3 + 1)
            plt.stem(np.arange(N_channel), np.abs(h), basefmt='b-')
            plt.title(f'{params["name"]}\\nImpulse Response')
            plt.xlabel('Sample')
            plt.ylabel('Magnitude')
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 20])
            
            # Plot frequency response magnitude
            plt.subplot(3, 3, idx*3 + 2)
            plt.plot(frequencies, 20*np.log10(np.abs(H)))
            plt.title('Frequency Response (Magnitude)')
            plt.xlabel('Normalized Frequency')
            plt.ylabel('Magnitude (dB)')
            plt.grid(True, alpha=0.3)
            
            # Plot frequency response phase
            plt.subplot(3, 3, idx*3 + 3)
            plt.plot(frequencies, np.angle(H))
            plt.title('Frequency Response (Phase)')
            plt.xlabel('Normalized Frequency')
            plt.ylabel('Phase (radians)')
            plt.grid(True, alpha=0.3)
            
            # Calculate RMS delay spread
            if len(params['delays']) > 1:
                mean_delay = np.average(params['delays'], weights=params['gains'])
                rms_delay_spread = np.sqrt(np.average((np.array(params['delays']) - mean_delay)**2, 
                                                     weights=params['gains']))
                coherence_bandwidth = 1 / (2 * np.pi * rms_delay_spread * 1e-6)
                
                print(f"  Mean delay: {mean_delay:.2f} μs")
                print(f"  RMS delay spread: {rms_delay_spread:.2f} μs")
                print(f"  Coherence bandwidth: {coherence_bandwidth/1e3:.1f} kHz")
                
                channel_results.append({
                    'name': params['name'],
                    'mean_delay': mean_delay,
                    'rms_delay_spread': rms_delay_spread,
                    'coherence_bandwidth': coherence_bandwidth,
                    'frequency_response': H
                })
            else:
                print(f"  No multipath - flat fading")
                channel_results.append({
                    'name': params['name'],
                    'mean_delay': 0,
                    'rms_delay_spread': 0,
                    'coherence_bandwidth': float('inf'),
                    'frequency_response': H
                })
        
        plt.tight_layout()
        plt.show()
        
        # Simulate effect on OFDM
        print("\\nFrequency selective fading effect on OFDM:")
        
        # OFDM signal
        N_ofdm = 64
        ofdm_data = np.random.randn(N_ofdm) + 1j * np.random.randn(N_ofdm)
        ofdm_signal = np.fft.ifft(ofdm_data)
        
        # Apply urban channel
        h_urban = generate_multipath_channel(channels['Urban']['delays'], 
                                           channels['Urban']['gains'], 
                                           N_channel)
        
        # Convolve with channel
        received_signal = np.convolve(ofdm_signal, h_urban, mode='same')
        
        # Demodulate
        received_data = np.fft.fft(received_signal)
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(ofdm_data.real, ofdm_data.imag, alpha=0.7, s=50, label='Transmitted')
        plt.title('Transmitted OFDM Subcarriers')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        plt.subplot(1, 2, 2)
        plt.scatter(received_data.real, received_data.imag, alpha=0.7, s=50, label='Received')
        plt.title('Received OFDM Subcarriers\\n(Frequency Selective Fading)')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        # Analysis of impact
        print("\\nImpact Analysis:")
        for result in channel_results:
            if result['rms_delay_spread'] > 0:
                # Compare with typical OFDM parameters
                ofdm_subcarrier_spacing = 15e3  # 15 kHz
                symbol_duration = 1 / ofdm_subcarrier_spacing
                
                if result['coherence_bandwidth'] > ofdm_subcarrier_spacing:
                    print(f"  {result['name']}: Flat fading per subcarrier (Good)")
                else:
                    print(f"  {result['name']}: Frequency selective per subcarrier (Needs equalization)")
        
        return {
            'channel_results': channel_results,
            'ofdm_impact': 'Frequency selective fading causes inter-symbol interference'
        }

def interactive_menu():
    """Interactive menu for running demonstrations"""
    project = ChannelEffectsProject()
    
    while True:
        print("\\n" + "="*50)
        print("🌊 CHANNEL EFFECTS PROJECT MENU")
        print("="*50)
        print("1. Multipath Fading Analysis")
        print("2. Doppler Effects Simulation")
        print("3. Frequency Selective Fading")
        print("4. Run All Demonstrations")
        print("5. Exit")
        
        try:
            choice = input("\\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                project.multipath_fading_project()
            elif choice == '2':
                project.doppler_effects_project()
            elif choice == '3':
                project.frequency_selective_fading_project()
            elif choice == '4':
                project.run_all_demonstrations()
            elif choice == '5':
                print("\\nExiting Channel Effects Project. Goodbye!")
                break
            else:
                print("\\nInvalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\\n\\nExiting Channel Effects Project. Goodbye!")
            break
        except Exception as e:
            print(f"\\nError: {e}")
            print("Please try again.")

if __name__ == "__main__":
    print(__doc__)
    interactive_menu()
