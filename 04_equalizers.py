"""
Equalizer Systems Project
========================

This comprehensive project implements various equalization techniques for communication systems:
- Zero-Forcing (ZF) Equalizer
- Minimum Mean Square Error (MMSE) Equalizer
- Adaptive Equalizers (LMS Algorithm)
- Performance Analysis and Comparison

Features:
- Multiple equalizer implementations
- Frequency and time domain processing
- Adaptive algorithms with convergence analysis
- BER performance comparisons
- Visual constellation analysis
- Educational explanations and documentation

Author: Communication Systems Learning Project
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, lfilter
import warnings
warnings.filterwarnings('ignore')

class EqualizerProject:
    """
    Equalizer Systems Implementation
    
    This class provides comprehensive implementations of various equalization
    techniques used to combat intersymbol interference (ISI) in communication systems.
    """
    
    def __init__(self):
        """Initialize the Equalizer Project"""
        print("⚖️ Equalizer Systems Project")
        print("===========================")
        print("\nThis project demonstrates:")
        print("• Zero-Forcing (ZF) equalization")
        print("• Minimum Mean Square Error (MMSE) equalization")
        print("• Adaptive equalization with LMS algorithm")
        print("• Performance analysis and comparison")
        
    def run_all_demonstrations(self):
        """Run all equalizer demonstrations"""
        demonstrations = [
            ("Zero-Forcing Equalizer", self.zero_forcing_equalizer_project),
            ("MMSE Equalizer", self.mmse_equalizer_project),
            ("Adaptive Equalizer (LMS)", self.adaptive_equalizer_project)
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
    
    def zero_forcing_equalizer_project(self):
        """Zero-forcing equalizer implementation"""
        print("\n🎯 Zero-Forcing Equalizer Project")
        print("This demonstration shows how ZF equalizers combat ISI")
        
        def zero_forcing_equalizer(H):
            """Zero-forcing equalizer"""
            # For SISO case: W = 1/H
            # For MIMO case: W = (H^H * H)^(-1) * H^H
            
            if H.ndim == 1:
                # SISO case
                W = 1 / H
            else:
                # MIMO case
                W = np.linalg.pinv(H)
            
            return W
        
        # Generate multipath channel
        h = np.array([1.0, 0.5, 0.3])  # Channel impulse response
        N_symbols = 1000
        
        print(f"System Parameters:")
        print(f"  Channel impulse response: {h}")
        print(f"  Number of symbols: {N_symbols}")
        
        # Generate QPSK symbols
        data_bits = np.random.randint(0, 2, N_symbols * 2)
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
        
        # Apply channel (convolution)
        received_signal = np.convolve(qpsk_symbols, h, mode='same')
        
        # Add noise
        snr_db = 20
        snr_linear = 10**(snr_db/10)
        noise_std = np.sqrt(1/(2*snr_linear))
        noise = noise_std * (np.random.randn(N_symbols) + 1j*np.random.randn(N_symbols))
        received_signal += noise
        
        # Frequency domain equalization
        N_fft = 64
        
        # Convert to frequency domain
        H_freq = np.fft.fft(h, N_fft)
        
        # Zero-forcing equalizer
        W_zf = zero_forcing_equalizer(H_freq)
        
        # Apply equalization in blocks
        equalized_signal = np.zeros_like(received_signal)
        
        for i in range(0, N_symbols, N_fft):
            block_end = min(i + N_fft, N_symbols)
            block = received_signal[i:block_end]
            
            # Pad to FFT size
            if len(block) < N_fft:
                block = np.pad(block, (0, N_fft - len(block)), mode='constant')
            
            # FFT
            block_freq = np.fft.fft(block)
            
            # Apply equalizer
            equalized_freq = block_freq * W_zf
            
            # IFFT
            equalized_block = np.fft.ifft(equalized_freq)
            
            # Store result
            equalized_signal[i:block_end] = equalized_block[:block_end - i]
        
        # Demodulate
        def qpsk_demod(symbols):
            bits = []
            for symbol in symbols:
                if symbol.real > 0 and symbol.imag > 0:
                    bits.extend([0, 0])
                elif symbol.real < 0 and symbol.imag > 0:
                    bits.extend([0, 1])
                elif symbol.real > 0 and symbol.imag < 0:
                    bits.extend([1, 0])
                else:
                    bits.extend([1, 1])
            return np.array(bits)
        
        # Calculate BER
        demod_bits_no_eq = qpsk_demod(received_signal)
        demod_bits_with_eq = qpsk_demod(equalized_signal)
        
        ber_no_eq = np.mean(data_bits != demod_bits_no_eq[:len(data_bits)])
        ber_with_eq = np.mean(data_bits != demod_bits_with_eq[:len(data_bits)])
        
        print(f"\\nResults:")
        print(f"  BER without equalization: {ber_no_eq:.6f}")
        print(f"  BER with ZF equalization: {ber_with_eq:.6f}")
        print(f"  Improvement factor: {ber_no_eq/max(ber_with_eq, 1e-6):.2f}")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.stem(np.arange(len(h)), np.abs(h), basefmt='b-')
        plt.title('Channel Impulse Response')
        plt.xlabel('Sample')
        plt.ylabel('Magnitude')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(20*np.log10(np.abs(H_freq)))
        plt.title('Channel Frequency Response')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.plot(20*np.log10(np.abs(W_zf)))
        plt.title('Zero-Forcing Equalizer')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 4)
        plt.scatter(qpsk_symbols.real, qpsk_symbols.imag, alpha=0.7, s=50, label='Transmitted')
        plt.title('Transmitted Constellation')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        plt.subplot(2, 3, 5)
        plt.scatter(received_signal.real, received_signal.imag, alpha=0.3, s=10, label='Received')
        plt.title('Received Constellation (No Equalization)')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        plt.subplot(2, 3, 6)
        plt.scatter(equalized_signal.real, equalized_signal.imag, alpha=0.3, s=10, label='Equalized')
        plt.title('Equalized Constellation (ZF)')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'ber_no_eq': ber_no_eq,
            'ber_with_eq': ber_with_eq,
            'improvement_factor': ber_no_eq/max(ber_with_eq, 1e-6),
            'channel_response': h,
            'equalizer_response': W_zf
        }
    
    def mmse_equalizer_project(self):
        """MMSE (Minimum Mean Square Error) equalizer implementation"""
        print("\n🎯 MMSE Equalizer Project")
        print("This demonstration compares ZF and MMSE equalization performance")
        
        def mmse_equalizer(H, noise_variance):
            """MMSE equalizer"""
            # For SISO case: W = H* / (|H|^2 + noise_variance)
            # For MIMO case: W = (H^H * H + noise_variance * I)^(-1) * H^H
            
            if H.ndim == 1:
                # SISO case
                W = np.conj(H) / (np.abs(H)**2 + noise_variance)
            else:
                HH = np.conj(H.T) @ H
                W = np.linalg.inv(HH + noise_variance * np.eye(H.shape[1])) @ np.conj(H.T)
            
            return W
        
        # Test different SNR values
        snr_values = np.arange(0, 25, 2)
        
        # Channel
        h = np.array([1.0, 0.6, 0.4])
        N_fft = 64
        H_freq = np.fft.fft(h, N_fft)
        
        print(f"System Parameters:")
        print(f"  Channel impulse response: {h}")
        print(f"  SNR range: {snr_values[0]} to {snr_values[-1]} dB")
        
        ber_zf = []
        ber_mmse = []
        
        for snr_db in snr_values:
            print(f"  Processing SNR: {snr_db} dB")
            
            # Generate symbols
            N_symbols = 1000
            data_bits = np.random.randint(0, 2, N_symbols * 2)
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
            
            # Apply channel
            received_signal = np.convolve(qpsk_symbols, h, mode='same')
            
            # Add noise
            snr_linear = 10**(snr_db/10)
            noise_variance = 1/(2*snr_linear)
            noise_std = np.sqrt(noise_variance)
            noise = noise_std * (np.random.randn(N_symbols) + 1j*np.random.randn(N_symbols))
            received_signal += noise
            
            # Equalizers
            W_zf = 1 / H_freq
            W_mmse = mmse_equalizer(H_freq, noise_variance)
            
            # Apply equalization (simplified for single block)
            received_freq = np.fft.fft(received_signal, N_fft)
            
            equalized_zf = np.fft.ifft(received_freq * W_zf)[:N_symbols]
            equalized_mmse = np.fft.ifft(received_freq * W_mmse)[:N_symbols]
            
            # Demodulate
            def qpsk_demod(symbols):
                bits = []
                for symbol in symbols:
                    if symbol.real > 0 and symbol.imag > 0:
                        bits.extend([0, 0])
                    elif symbol.real < 0 and symbol.imag > 0:
                        bits.extend([0, 1])
                    elif symbol.real > 0 and symbol.imag < 0:
                        bits.extend([1, 0])
                    else:
                        bits.extend([1, 1])
                return np.array(bits)
            
            # Calculate BER
            demod_bits_zf = qpsk_demod(equalized_zf)
            demod_bits_mmse = qpsk_demod(equalized_mmse)
            
            ber_zf.append(np.mean(data_bits != demod_bits_zf[:len(data_bits)]))
            ber_mmse.append(np.mean(data_bits != demod_bits_mmse[:len(data_bits)]))
        
        # Plot BER comparison
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.semilogy(snr_values, ber_zf, 'o-', label='Zero-Forcing')
        plt.semilogy(snr_values, ber_mmse, 's-', label='MMSE')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Bit Error Rate')
        plt.title('BER Comparison: ZF vs MMSE')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(20*np.log10(np.abs(H_freq)), label='Channel')
        plt.title('Channel Frequency Response')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Compare equalizers at medium SNR
        test_snr = 15
        noise_var = 1/(2*10**(test_snr/10))
        W_zf_test = 1 / H_freq
        W_mmse_test = mmse_equalizer(H_freq, noise_var)
        
        plt.subplot(2, 2, 3)
        plt.plot(20*np.log10(np.abs(W_zf_test)), label='Zero-Forcing')
        plt.plot(20*np.log10(np.abs(W_mmse_test)), label='MMSE')
        plt.title(f'Equalizer Responses (SNR = {test_snr} dB)')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        combined_zf = H_freq * W_zf_test
        combined_mmse = H_freq * W_mmse_test
        plt.plot(20*np.log10(np.abs(combined_zf)), label='ZF Combined')
        plt.plot(20*np.log10(np.abs(combined_mmse)), label='MMSE Combined')
        plt.title('Combined Channel + Equalizer Response')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Performance analysis
        low_snr_advantage = ber_zf[0]/max(ber_mmse[0], 1e-6)
        high_snr_advantage = ber_zf[-1]/max(ber_mmse[-1], 1e-6)
        
        print(f"\\nPerformance Analysis:")
        print(f"  At low SNR ({snr_values[0]} dB): MMSE advantage = {low_snr_advantage:.2f}x")
        print(f"  At high SNR ({snr_values[-1]} dB): MMSE advantage = {high_snr_advantage:.2f}x")
        print(f"  MMSE shows greater advantage at low SNR due to noise regularization")
        
        return {
            'snr_values': snr_values,
            'ber_zf': ber_zf,
            'ber_mmse': ber_mmse,
            'low_snr_advantage': low_snr_advantage,
            'high_snr_advantage': high_snr_advantage
        }
    
    def adaptive_equalizer_project(self):
        """Adaptive equalizer using LMS algorithm"""
        print("\n🔄 Adaptive Equalizer Project")
        print("This demonstration shows how adaptive equalizers track time-varying channels")
        
        def lms_equalizer(received_signal, training_symbols, mu, N_taps):
            """LMS adaptive equalizer"""
            N_symbols = len(training_symbols)
            
            # Initialize equalizer taps
            w = np.zeros(N_taps, dtype=complex)
            w[N_taps//2] = 1.0  # Initialize with delta function
            
            # Delay line
            delay_line = np.zeros(N_taps, dtype=complex)
            
            # Output storage
            equalized_output = np.zeros(N_symbols, dtype=complex)
            error_history = []
            weight_history = []
            
            for n in range(N_symbols):
                # Update delay line
                delay_line[1:] = delay_line[:-1]
                delay_line[0] = received_signal[n]
                
                # Equalizer output
                y = np.dot(np.conj(w), delay_line)
                equalized_output[n] = y
                
                # Error calculation
                error = training_symbols[n] - y
                error_history.append(np.abs(error)**2)
                weight_history.append(w.copy())
                
                # LMS update
                w = w + mu * error * delay_line
            
            return equalized_output, w, error_history, weight_history
        
        # Generate training sequence
        N_train = 500
        training_bits = np.random.randint(0, 2, N_train * 2)
        training_symbols = []
        for i in range(0, len(training_bits), 2):
            bit_pair = training_bits[i:i+2]
            if np.array_equal(bit_pair, [0, 0]):
                training_symbols.append(1+1j)
            elif np.array_equal(bit_pair, [0, 1]):
                training_symbols.append(-1+1j)
            elif np.array_equal(bit_pair, [1, 0]):
                training_symbols.append(1-1j)
            else:
                training_symbols.append(-1-1j)
        
        training_symbols = np.array(training_symbols)
        
        print(f"Adaptive Equalizer Parameters:")
        print(f"  Training sequence length: {N_train} symbols")
        print(f"  Channel: Time-varying multipath")
        
        # Channel with time variation
        h_base = np.array([1.0, 0.5, 0.3])
        
        # Generate received signal with time-varying channel
        received_signal = np.zeros(N_train, dtype=complex)
        channel_history = []
        
        for n in range(N_train):
            # Slowly varying channel
            variation = 0.1 * np.sin(2 * np.pi * n / 100)
            h = h_base * (1 + variation)
            channel_history.append(h.copy())
            
            # Apply channel (simplified)
            if n >= len(h):
                signal_part = np.sum(training_symbols[n-len(h)+1:n+1] * h[::-1])
            else:
                signal_part = np.sum(training_symbols[:n+1] * h[-(n+1):])
            
            received_signal[n] = signal_part
        
        # Add noise
        snr_db = 15
        snr_linear = 10**(snr_db/10)
        noise_std = np.sqrt(1/(2*snr_linear))
        noise = noise_std * (np.random.randn(N_train) + 1j*np.random.randn(N_train))
        received_signal += noise
        
        # Test different step sizes
        mu_values = [0.01, 0.05, 0.1]
        N_taps = 5
        
        print(f"  Number of equalizer taps: {N_taps}")
        print(f"  Testing step sizes: {mu_values}")
        
        plt.figure(figsize=(15, 10))
        
        results = []
        for idx, mu in enumerate(mu_values):
            print(f"  Running LMS with μ = {mu}")
            
            # Run LMS equalizer
            equalized, final_taps, error_hist, weight_hist = lms_equalizer(
                received_signal, training_symbols, mu, N_taps)
            
            # Plot convergence
            plt.subplot(2, 3, idx+1)
            plt.plot(10*np.log10(error_hist))
            plt.title(f'LMS Convergence (μ = {mu})')
            plt.xlabel('Iteration')
            plt.ylabel('MSE (dB)')
            plt.grid(True, alpha=0.3)
            
            # Plot final equalizer taps
            plt.subplot(2, 3, idx+4)
            plt.stem(np.arange(N_taps), np.abs(final_taps), basefmt='b-')
            plt.title(f'Final Equalizer Taps (μ = {mu})')
            plt.xlabel('Tap Index')
            plt.ylabel('Magnitude')
            plt.grid(True, alpha=0.3)
            
            final_mse = error_hist[-1]
            convergence_time = np.argmin(np.array(error_hist) < 2 * final_mse) if len(error_hist) > 10 else len(error_hist)
            
            results.append({
                'mu': mu,
                'final_mse': final_mse,
                'convergence_time': convergence_time,
                'equalized_signal': equalized,
                'final_taps': final_taps
            })
            
            print(f"    Final MSE: {final_mse:.6f}")
            print(f"    Convergence time: {convergence_time} symbols")
        
        plt.tight_layout()
        plt.show()
        
        # Compare constellations
        plt.figure(figsize=(12, 6))
        
        # Best performing mu
        best_result = min(results, key=lambda x: x['final_mse'])
        best_mu = best_result['mu']
        equalized_best = best_result['equalized_signal']
        
        plt.subplot(1, 2, 1)
        plt.scatter(received_signal.real, received_signal.imag, alpha=0.3, s=10, label='Received')
        plt.title('Received Signal (No Equalization)')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        plt.subplot(1, 2, 2)
        plt.scatter(equalized_best.real, equalized_best.imag, alpha=0.3, s=10, label='Equalized')
        plt.scatter(training_symbols.real, training_symbols.imag, alpha=0.7, s=50, 
                   marker='x', c='red', label='Training')
        plt.title(f'Adaptive Equalized Signal (μ = {best_mu})')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\\nAdaptive Equalization Results:")
        print(f"  Best step size: μ = {best_mu}")
        print(f"  Final MSE: {best_result['final_mse']:.6f}")
        print(f"  Convergence time: {best_result['convergence_time']} symbols")
        
        return {
            'results': results,
            'best_mu': best_mu,
            'n_taps': N_taps,
            'training_length': N_train
        }

def interactive_menu():
    """Interactive menu for running demonstrations"""
    project = EqualizerProject()
    
    while True:
        print("\\n" + "="*50)
        print("⚖️ EQUALIZER SYSTEMS PROJECT MENU")
        print("="*50)
        print("1. Zero-Forcing Equalizer")
        print("2. MMSE Equalizer")
        print("3. Adaptive Equalizer (LMS)")
        print("4. Run All Demonstrations")
        print("5. Exit")
        
        try:
            choice = input("\\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                project.zero_forcing_equalizer_project()
            elif choice == '2':
                project.mmse_equalizer_project()
            elif choice == '3':
                project.adaptive_equalizer_project()
            elif choice == '4':
                project.run_all_demonstrations()
            elif choice == '5':
                print("\\nExiting Equalizer Systems Project. Goodbye!")
                break
            else:
                print("\\nInvalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\\n\\nExiting Equalizer Systems Project. Goodbye!")
            break
        except Exception as e:
            print(f"\\nError: {e}")
            print("Please try again.")

if __name__ == "__main__":
    print(__doc__)
    interactive_menu()
