"""
MIMO Systems Project
===================

This comprehensive project implements various MIMO (Multiple-Input Multiple-Output) techniques:
- Spatial Diversity (Alamouti Space-Time Block Coding)
- Spatial Multiplexing (BLAST)
- Digital Beamforming (Conventional and MVDR)
- Performance Analysis and Capacity Calculations

Features:
- Complete MIMO system implementations
- Diversity and multiplexing gain analysis
- Adaptive beamforming algorithms
- Channel capacity calculations
- Visual performance analysis
- Educational explanations and documentation

Author: Communication Systems Learning Project
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, inv
import warnings
warnings.filterwarnings('ignore')

class MIMOSystemsProject:
    """
    MIMO Systems Implementation
    
    This class provides comprehensive implementations of various MIMO
    techniques for improving communication system performance.
    """
    
    def __init__(self):
        """Initialize the MIMO Systems Project"""
        print("📡📡 MIMO Systems Project")
        print("========================")
        print("\nThis project demonstrates:")
        print("• Spatial diversity with Alamouti coding")
        print("• Spatial multiplexing with BLAST")
        print("• Digital beamforming techniques")
        print("• MIMO channel capacity analysis")
        
    def run_all_demonstrations(self):
        """Run all MIMO demonstrations"""
        demonstrations = [
            ("Spatial Diversity (Alamouti)", self.spatial_diversity_project),
            ("Spatial Multiplexing (BLAST)", self.spatial_multiplexing_project),
            ("Digital Beamforming", self.beamforming_project)
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
    
    def spatial_diversity_project(self):
        """Spatial diversity using Alamouti code"""
        print("\n🌟 Spatial Diversity Project (Alamouti Code)")
        print("This demonstration shows how spatial diversity improves reliability")
        
        def alamouti_encode(symbols):
            """Alamouti space-time block code encoding"""
            # Input: stream of symbols [s1, s2, s3, s4, ...]
            # Output: 2x2 matrix for each pair
            #         [s1  s2]  [s3  s4]
            #         [-s2* s1*] [-s4* s3*] ...
            
            if len(symbols) % 2 != 0:
                symbols = np.append(symbols, 0)  # Pad if odd length
            
            encoded_blocks = []
            for i in range(0, len(symbols), 2):
                s1, s2 = symbols[i], symbols[i+1]
                block = np.array([
                    [s1, s2],
                    [-np.conj(s2), np.conj(s1)]
                ])
                encoded_blocks.append(block)
            
            return encoded_blocks
        
        def alamouti_decode(received_blocks, channel_matrix):
            """Alamouti decoding with ML detection"""
            # Channel matrix H is 2x2: [[h11, h12], [h21, h22]]
            # where hij is channel from TX antenna j to RX antenna i
            
            decoded_symbols = []
            
            for r_block in received_blocks:
                # r_block is 2x2: [[r11, r12], [r21, r22]]
                r1, r2 = r_block[0], r_block[1]  # received at two time slots
                
                # Alamouti combining
                h11, h12 = channel_matrix[0, 0], channel_matrix[0, 1]
                h21, h22 = channel_matrix[1, 0], channel_matrix[1, 1]
                
                # Combiner output
                s1_hat = (np.conj(h11) * r1[0] + h21 * np.conj(r2[0]) + 
                         np.conj(h12) * r1[1] + h22 * np.conj(r2[1]))
                s2_hat = (np.conj(h12) * r1[0] - h22 * np.conj(r2[0]) - 
                         np.conj(h11) * r1[1] + h21 * np.conj(r2[1]))
                
                # Normalize by channel power
                channel_power = np.abs(h11)**2 + np.abs(h12)**2 + np.abs(h21)**2 + np.abs(h22)**2
                s1_hat /= channel_power
                s2_hat /= channel_power
                
                decoded_symbols.extend([s1_hat, s2_hat])
            
            return np.array(decoded_symbols)
        
        # Generate QPSK symbols
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
        
        print(f"Alamouti System Parameters:")
        print(f"  Number of symbols: {N_symbols}")
        print(f"  TX antennas: 2")
        print(f"  RX antennas: 2")
        print(f"  Modulation: QPSK")
        
        # Alamouti encoding
        encoded_blocks = alamouti_encode(qpsk_symbols)
        
        # 2x2 MIMO channel (2 TX, 2 RX)
        H = np.array([
            [1.0 + 0.2j, 0.8 - 0.3j],
            [0.6 + 0.4j, 1.1 + 0.1j]
        ])
        
        # Simulate transmission through MIMO channel
        received_blocks = []
        snr_db = 15
        snr_linear = 10**(snr_db/10)
        noise_std = np.sqrt(1/(2*snr_linear))
        
        for encoded_block in encoded_blocks:
            # MIMO channel: Y = H * X + N
            received_block = H @ encoded_block
            
            # Add noise
            noise = noise_std * (np.random.randn(2, 2) + 1j*np.random.randn(2, 2))
            received_block += noise
            
            received_blocks.append(received_block)
        
        # Alamouti decoding
        decoded_symbols = alamouti_decode(received_blocks, H)
        
        # Compare with SISO system
        # Simulate single antenna transmission
        siso_received = qpsk_symbols * H[0, 0]  # Use only one channel coefficient
        siso_noise = noise_std * (np.random.randn(len(qpsk_symbols)) + 1j*np.random.randn(len(qpsk_symbols)))
        siso_received += siso_noise
        siso_decoded = siso_received / H[0, 0]
        
        # Calculate SNR improvement
        alamouti_power = np.mean(np.abs(decoded_symbols - qpsk_symbols[:len(decoded_symbols)])**2)
        siso_power = np.mean(np.abs(siso_decoded - qpsk_symbols)**2)
        snr_improvement_db = 10 * np.log10(siso_power / alamouti_power)
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.scatter(qpsk_symbols.real, qpsk_symbols.imag, alpha=0.7, s=50)
        plt.title('Original QPSK Symbols')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.subplot(2, 3, 2)
        # Visualize channel matrix
        plt.imshow(np.abs(H), cmap='viridis')
        plt.colorbar(label='Channel Magnitude')
        plt.title('MIMO Channel Matrix |H|')
        plt.xlabel('TX Antenna')
        plt.ylabel('RX Antenna')
        
        plt.subplot(2, 3, 3)
        plt.scatter(siso_decoded.real, siso_decoded.imag, alpha=0.3, s=10, label='SISO')
        plt.title('SISO System Reception')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        plt.subplot(2, 3, 4)
        plt.scatter(decoded_symbols.real, decoded_symbols.imag, alpha=0.3, s=10, label='Alamouti')
        plt.title('Alamouti Code Reception')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        # BER vs SNR comparison
        snr_range = np.arange(0, 20, 2)
        ber_siso = []
        ber_alamouti = []
        
        for test_snr in snr_range:
            # Simplified BER calculation for comparison
            test_snr_linear = 10**(test_snr/10)
            
            # SISO BER (QPSK in Rayleigh fading)
            avg_snr_siso = test_snr_linear * np.abs(H[0,0])**2
            ber_siso.append(0.5 * (1 - np.sqrt(avg_snr_siso / (1 + avg_snr_siso))))
            
            # Alamouti BER (diversity order 2)
            avg_snr_alamouti = test_snr_linear * (np.abs(H[0,0])**2 + np.abs(H[0,1])**2 + 
                                                 np.abs(H[1,0])**2 + np.abs(H[1,1])**2) / 2
            ber_alamouti.append(0.25 * (1 - np.sqrt(avg_snr_alamouti / (1 + avg_snr_alamouti)))**2)
        
        plt.subplot(2, 3, 5)
        plt.semilogy(snr_range, ber_siso, 'b-o', label='SISO')
        plt.semilogy(snr_range, ber_alamouti, 'r-s', label='Alamouti (2x2)')
        plt.xlabel('SNR (dB)')
        plt.ylabel('BER')
        plt.title('BER Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 6)
        # Diversity gain visualization
        diversity_gain = np.array(ber_siso) / np.array(ber_alamouti)
        plt.plot(snr_range, 10*np.log10(diversity_gain), 'g-o')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Diversity Gain (dB)')
        plt.title('Diversity Gain vs SNR')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\\nPerformance Results:")
        print(f"  MIMO channel matrix H:")
        for i, row in enumerate(H):
            print(f"    Row {i}: {row}")
        print(f"  Number of decoded symbols: {len(decoded_symbols)}")
        print(f"  SNR improvement: {snr_improvement_db:.2f} dB")
        print(f"  Diversity order: 2 (2 TX × 2 RX)")
        print(f"  Code rate: 1 symbol/channel use")
        print(f"  Alamouti provides diversity gain without rate loss")
        
        return {
            'snr_improvement': snr_improvement_db,
            'diversity_order': 2,
            'code_rate': 1,
            'ber_siso': ber_siso,
            'ber_alamouti': ber_alamouti
        }
    
    def spatial_multiplexing_project(self):
        """Spatial multiplexing MIMO system"""
        print("\n🚀 Spatial Multiplexing Project (BLAST)")
        print("This demonstration shows how spatial multiplexing increases data rate")
        
        def blast_encode(symbols, n_tx):
            """BLAST encoding - distribute symbols across antennas"""
            # Reshape symbols into matrix: each column is for one TX antenna
            n_symbols = len(symbols)
            n_time_slots = int(np.ceil(n_symbols / n_tx))
            
            # Pad symbols if necessary
            padded_symbols = np.pad(symbols, (0, n_time_slots * n_tx - n_symbols), mode='constant')
            
            # Reshape: each row is a time slot, each column is a TX antenna
            symbol_matrix = padded_symbols.reshape(n_time_slots, n_tx)
            
            return symbol_matrix
        
        def zf_mimo_detector(received_matrix, channel_matrix):
            """Zero-forcing MIMO detector"""
            # received_matrix: n_rx × n_time_slots
            # channel_matrix: n_rx × n_tx
            
            # Zero-forcing: W = (H^H * H)^(-1) * H^H
            H_hermitian = np.conj(channel_matrix.T)
            W = np.linalg.inv(H_hermitian @ channel_matrix) @ H_hermitian
            
            # Apply detector
            detected_matrix = W @ received_matrix
            
            return detected_matrix
        
        def mmse_mimo_detector(received_matrix, channel_matrix, noise_variance):
            """MMSE MIMO detector"""
            # MMSE: W = (H^H * H + σ²I)^(-1) * H^H
            n_tx = channel_matrix.shape[1]
            H_hermitian = np.conj(channel_matrix.T)
            W = np.linalg.inv(H_hermitian @ channel_matrix + 
                            noise_variance * np.eye(n_tx)) @ H_hermitian
            
            # Apply detector
            detected_matrix = W @ received_matrix
            
            return detected_matrix
        
        # Parameters
        n_tx = 4  # Number of TX antennas
        n_rx = 4  # Number of RX antennas
        
        # Generate QPSK symbols
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
        
        print(f"BLAST System Parameters:")
        print(f"  Number of symbols: {N_symbols}")
        print(f"  TX antennas: {n_tx}")
        print(f"  RX antennas: {n_rx}")
        print(f"  Modulation: QPSK")
        
        # BLAST encoding
        symbol_matrix = blast_encode(qpsk_symbols, n_tx)
        n_time_slots = symbol_matrix.shape[0]
        
        # Random MIMO channel
        np.random.seed(42)  # For reproducibility
        H = (np.random.randn(n_rx, n_tx) + 1j*np.random.randn(n_rx, n_tx)) / np.sqrt(2)
        
        # Transmission through MIMO channel
        # Y = H * X + N
        transmitted_matrix = symbol_matrix.T  # n_tx × n_time_slots
        received_matrix = H @ transmitted_matrix
        
        # Add noise
        snr_db = 15
        snr_linear = 10**(snr_db/10)
        noise_variance = 1 / snr_linear
        noise_std = np.sqrt(noise_variance)
        noise = noise_std * (np.random.randn(n_rx, n_time_slots) + 
                           1j*np.random.randn(n_rx, n_time_slots))
        received_matrix += noise
        
        # MIMO detection
        detected_zf = zf_mimo_detector(received_matrix, H)
        detected_mmse = mmse_mimo_detector(received_matrix, H, noise_variance)
        
        # Reshape back to symbol streams
        detected_symbols_zf = detected_zf.T.flatten()[:len(qpsk_symbols)]
        detected_symbols_mmse = detected_mmse.T.flatten()[:len(qpsk_symbols)]
        
        # Calculate capacity
        # C = log2(det(I + SNR/n_tx * H * H^H))
        identity = np.eye(n_rx)
        channel_gram = H @ np.conj(H.T)
        capacity = np.log2(np.real(np.linalg.det(identity + snr_linear/n_tx * channel_gram)))
        
        # SVD analysis
        U, S, Vh = np.linalg.svd(H)
        
        # Plot results
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 3, 1)
        plt.imshow(np.abs(H), cmap='viridis', aspect='auto')
        plt.colorbar(label='Channel Magnitude')
        plt.title(f'MIMO Channel Matrix |H| ({n_rx}×{n_tx})')
        plt.xlabel('TX Antenna')
        plt.ylabel('RX Antenna')
        
        plt.subplot(3, 3, 2)
        plt.scatter(qpsk_symbols.real, qpsk_symbols.imag, alpha=0.7, s=30, label='Original')
        plt.title('Original QPSK Symbols')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        plt.subplot(3, 3, 3)
        plt.scatter(detected_symbols_zf.real, detected_symbols_zf.imag, 
                   alpha=0.3, s=10, label='ZF Detection')
        plt.title('Zero-Forcing Detection')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        plt.subplot(3, 3, 4)
        plt.scatter(detected_symbols_mmse.real, detected_symbols_mmse.imag, 
                   alpha=0.3, s=10, label='MMSE Detection')
        plt.title('MMSE Detection')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        plt.subplot(3, 3, 5)
        plt.bar(range(len(S)), S)
        plt.title('Channel Singular Values')
        plt.xlabel('Index')
        plt.ylabel('Singular Value')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 6)
        # Capacity vs SNR
        snr_range = np.arange(0, 30, 2)
        capacity_mimo = []
        capacity_siso = []
        
        for test_snr in snr_range:
            test_snr_linear = 10**(test_snr/10)
            
            # MIMO capacity
            cap_mimo = np.log2(np.real(np.linalg.det(
                np.eye(n_rx) + test_snr_linear/n_tx * channel_gram)))
            capacity_mimo.append(cap_mimo)
            
            # SISO capacity (for comparison)
            cap_siso = np.log2(1 + test_snr_linear * np.abs(H[0,0])**2)
            capacity_siso.append(cap_siso)
        
        plt.plot(snr_range, capacity_mimo, 'b-o', label=f'{n_tx}×{n_rx} MIMO')
        plt.plot(snr_range, capacity_siso, 'r--', label='SISO')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Capacity (bits/s/Hz)')
        plt.title('Channel Capacity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 7)
        # Simple BER estimation
        def calculate_ber(original, detected):
            # Hard decision for QPSK
            original_bits = []
            detected_bits = []
            
            for sym in original:
                if sym.real > 0 and sym.imag > 0:
                    original_bits.extend([0, 0])
                elif sym.real < 0 and sym.imag > 0:
                    original_bits.extend([0, 1])
                elif sym.real > 0 and sym.imag < 0:
                    original_bits.extend([1, 0])
                else:
                    original_bits.extend([1, 1])
            
            for sym in detected:
                if sym.real > 0 and sym.imag > 0:
                    detected_bits.extend([0, 0])
                elif sym.real < 0 and sym.imag > 0:
                    detected_bits.extend([0, 1])
                elif sym.real > 0 and sym.imag < 0:
                    detected_bits.extend([1, 0])
                else:
                    detected_bits.extend([1, 1])
            
            return np.mean(np.array(original_bits) != np.array(detected_bits[:len(original_bits)]))
        
        ber_zf = calculate_ber(qpsk_symbols, detected_symbols_zf)
        ber_mmse = calculate_ber(qpsk_symbols, detected_symbols_mmse)
        
        detection_methods = ['Zero-Forcing', 'MMSE']
        ber_values = [ber_zf, ber_mmse]
        
        plt.bar(detection_methods, ber_values)
        plt.title('BER Comparison')
        plt.ylabel('Bit Error Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 8)
        # Condition number analysis
        condition_number = np.max(S) / np.min(S)
        plt.text(0.5, 0.7, f'Condition Number: {condition_number:.2f}', 
                transform=plt.gca().transAxes, ha='center', fontsize=12)
        plt.text(0.5, 0.5, f'Channel Capacity: {capacity:.2f} bits/s/Hz', 
                transform=plt.gca().transAxes, ha='center', fontsize=12)
        plt.text(0.5, 0.3, f'Multiplexing Gain: {n_tx}x', 
                transform=plt.gca().transAxes, ha='center', fontsize=12)
        plt.title('MIMO System Metrics')
        plt.axis('off')
        
        plt.subplot(3, 3, 9)
        # Power distribution across antennas
        tx_power = np.mean(np.abs(symbol_matrix)**2, axis=0)
        plt.bar(range(n_tx), tx_power)
        plt.title('Power Distribution (TX Antennas)')
        plt.xlabel('TX Antenna')
        plt.ylabel('Average Power')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\\nPerformance Results:")
        print(f"  MIMO configuration: {n_tx}×{n_rx}")
        print(f"  Channel capacity: {capacity:.2f} bits/s/Hz")
        print(f"  Multiplexing gain: {n_tx}x")
        print(f"  Channel condition number: {condition_number:.2f}")
        print(f"  ZF BER: {ber_zf:.6f}")
        print(f"  MMSE BER: {ber_mmse:.6f}")
        print(f"  Channel singular values: {S}")
        print(f"  BLAST provides multiplexing gain for higher data rates")
        
        return {
            'capacity': capacity,
            'multiplexing_gain': n_tx,
            'condition_number': condition_number,
            'ber_zf': ber_zf,
            'ber_mmse': ber_mmse,
            'singular_values': S
        }
    
    def beamforming_project(self):
        """Digital beamforming for MIMO systems"""
        print("\n📡 Digital Beamforming Project")
        print("This demonstration shows how beamforming improves signal quality")
        
        def calculate_steering_vector(n_elements, element_spacing, angle_deg):
            """Calculate steering vector for uniform linear array"""
            angle_rad = np.deg2rad(angle_deg)
            k = 2 * np.pi / 1.0  # Assuming wavelength = 1
            
            steering_vector = np.zeros(n_elements, dtype=complex)
            for i in range(n_elements):
                phase = k * i * element_spacing * np.sin(angle_rad)
                steering_vector[i] = np.exp(1j * phase)
            
            return steering_vector
        
        def conventional_beamformer(steering_vector):
            """Conventional (delay-and-sum) beamformer"""
            return steering_vector / np.linalg.norm(steering_vector)
        
        def mvdr_beamformer(steering_vector, interference_covariance):
            """Minimum Variance Distortionless Response (MVDR) beamformer"""
            # MVDR: w = R^(-1) * a / (a^H * R^(-1) * a)
            R_inv = np.linalg.inv(interference_covariance)
            numerator = R_inv @ steering_vector
            denominator = np.conj(steering_vector.T) @ R_inv @ steering_vector
            
            return numerator / denominator
        
        def calculate_array_pattern(weights, n_elements, element_spacing, angles):
            """Calculate array radiation pattern"""
            pattern = np.zeros(len(angles), dtype=complex)
            
            for i, angle in enumerate(angles):
                steering_vec = calculate_steering_vector(n_elements, element_spacing, angle)
                pattern[i] = np.conj(weights.T) @ steering_vec
            
            return pattern
        
        # Array parameters
        n_elements = 8
        element_spacing = 0.5  # In wavelengths
        
        # Desired signal direction
        desired_angle = 30  # degrees
        desired_steering_vector = calculate_steering_vector(n_elements, element_spacing, desired_angle)
        
        # Interference directions
        interference_angles = [-20, 45, 60]  # degrees
        interference_power = [1.0, 0.8, 0.6]  # relative powers
        
        print(f"Beamforming System Parameters:")
        print(f"  Number of elements: {n_elements}")
        print(f"  Element spacing: {element_spacing} wavelengths")
        print(f"  Desired signal angle: {desired_angle}°")
        print(f"  Interference angles: {interference_angles}°")
        print(f"  Interference powers: {interference_power}")
        
        # Create interference covariance matrix
        noise_power = 0.1
        R_interference = noise_power * np.eye(n_elements, dtype=complex)  # Noise contribution
        
        for angle, power in zip(interference_angles, interference_power):
            interference_steering = calculate_steering_vector(n_elements, element_spacing, angle)
            R_interference += power * np.outer(interference_steering, np.conj(interference_steering))
        
        # Calculate beamforming weights
        conventional_weights = conventional_beamformer(desired_steering_vector)
        mvdr_weights = mvdr_beamformer(desired_steering_vector, R_interference)
        
        # Calculate array patterns
        angles = np.linspace(-90, 90, 361)
        conventional_pattern = calculate_array_pattern(conventional_weights, n_elements, 
                                                     element_spacing, angles)
        mvdr_pattern = calculate_array_pattern(mvdr_weights, n_elements, 
                                             element_spacing, angles)
        
        # Simulate received signals
        N_samples = 1000
        
        # Desired signal (QPSK)
        desired_bits = np.random.randint(0, 2, N_samples * 2)
        desired_symbols = []
        for i in range(0, len(desired_bits), 2):
            bit_pair = desired_bits[i:i+2]
            if np.array_equal(bit_pair, [0, 0]):
                desired_symbols.append(1+1j)
            elif np.array_equal(bit_pair, [0, 1]):
                desired_symbols.append(-1+1j)
            elif np.array_equal(bit_pair, [1, 0]):
                desired_symbols.append(1-1j)
            else:
                desired_symbols.append(-1-1j)
        
        desired_symbols = np.array(desired_symbols)
        
        # Received signal at array
        received_array = np.zeros((n_elements, N_samples), dtype=complex)
        
        # Desired signal contribution
        for i in range(n_elements):
            received_array[i, :] = desired_steering_vector[i] * desired_symbols
        
        # Interference contributions
        for angle, power in zip(interference_angles, interference_power):
            interference_steering = calculate_steering_vector(n_elements, element_spacing, angle)
            interference_signal = np.sqrt(power) * (np.random.randn(N_samples) + 1j*np.random.randn(N_samples))
            
            for i in range(n_elements):
                received_array[i, :] += interference_steering[i] * interference_signal
        
        # Add noise
        noise = np.sqrt(noise_power) * (np.random.randn(n_elements, N_samples) + 
                                       1j*np.random.randn(n_elements, N_samples))
        received_array += noise
        
        # Apply beamforming
        conventional_output = np.conj(conventional_weights.T) @ received_array
        mvdr_output = np.conj(mvdr_weights.T) @ received_array
        
        # Calculate SINR
        def calculate_sinr(beamformed_output, desired_symbols):
            signal_power = np.mean(np.abs(desired_symbols)**2)
            noise_interference_power = np.mean(np.abs(beamformed_output - desired_symbols)**2)
            return signal_power / noise_interference_power
        
        sinr_conventional = calculate_sinr(conventional_output, desired_symbols)
        sinr_mvdr = calculate_sinr(mvdr_output, desired_symbols)
        
        # Plot results
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 3, 1)
        # Array geometry
        element_positions = np.arange(n_elements) * element_spacing
        plt.plot(element_positions, np.zeros(n_elements), 'bo', markersize=10)
        plt.arrow(0, 0.5, np.sin(np.deg2rad(desired_angle)), np.cos(np.deg2rad(desired_angle)), 
                 head_width=0.1, head_length=0.1, fc='green', ec='green', label='Desired')
        for angle in interference_angles:
            plt.arrow(0, 0.5, np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 
                     head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
        plt.xlabel('Position (wavelengths)')
        plt.ylabel('Height')
        plt.title('Array Geometry')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(3, 3, 2)
        plt.plot(np.arange(n_elements), np.abs(conventional_weights), 'bo-', label='Conventional')
        plt.plot(np.arange(n_elements), np.abs(mvdr_weights), 'rs-', label='MVDR')
        plt.xlabel('Element Index')
        plt.ylabel('Weight Magnitude')
        plt.title('Beamforming Weights')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 3)
        plt.plot(np.arange(n_elements), np.angle(conventional_weights), 'bo-', label='Conventional')
        plt.plot(np.arange(n_elements), np.angle(mvdr_weights), 'rs-', label='MVDR')
        plt.xlabel('Element Index')
        plt.ylabel('Weight Phase (radians)')
        plt.title('Beamforming Weight Phases')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 4)
        plt.plot(angles, 20*np.log10(np.abs(conventional_pattern)), 'b-', label='Conventional')
        plt.axvline(x=desired_angle, color='green', linestyle='--', alpha=0.7, label='Desired')
        for angle in interference_angles:
            plt.axvline(x=angle, color='red', linestyle=':', alpha=0.7)
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Array Gain (dB)')
        plt.title('Conventional Beamforming Pattern')
        plt.ylim([-40, 20])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 5)
        plt.plot(angles, 20*np.log10(np.abs(mvdr_pattern)), 'r-', label='MVDR')
        plt.axvline(x=desired_angle, color='green', linestyle='--', alpha=0.7, label='Desired')
        for angle in interference_angles:
            plt.axvline(x=angle, color='red', linestyle=':', alpha=0.7)
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Array Gain (dB)')
        plt.title('MVDR Beamforming Pattern')
        plt.ylim([-40, 20])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 6)
        # SINR comparison
        methods = ['Conventional', 'MVDR']
        sinr_values = [10*np.log10(sinr_conventional), 10*np.log10(sinr_mvdr)]
        
        plt.bar(methods, sinr_values)
        plt.ylabel('SINR (dB)')
        plt.title('SINR Comparison')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 7)
        plt.scatter(desired_symbols.real, desired_symbols.imag, alpha=0.7, s=30, label='Desired')
        plt.title('Original Desired Signal')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        plt.subplot(3, 3, 8)
        plt.scatter(conventional_output.real, conventional_output.imag, alpha=0.3, s=10, 
                   label='Conventional')
        plt.title('Conventional Beamformer Output')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        plt.subplot(3, 3, 9)
        plt.scatter(mvdr_output.real, mvdr_output.imag, alpha=0.3, s=10, label='MVDR')
        plt.title('MVDR Beamformer Output')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate null depths
        null_depths = []
        for i, angle in enumerate(interference_angles):
            conv_gain = np.abs(conventional_pattern[np.argmin(np.abs(angles - angle))])
            mvdr_gain = np.abs(mvdr_pattern[np.argmin(np.abs(angles - angle))])
            null_depths.append({
                'angle': angle,
                'conventional_db': 20*np.log10(conv_gain),
                'mvdr_db': 20*np.log10(mvdr_gain)
            })
        
        print(f"\\nPerformance Results:")
        print(f"  Conventional beamformer SINR: {10*np.log10(sinr_conventional):.2f} dB")
        print(f"  MVDR beamformer SINR: {10*np.log10(sinr_mvdr):.2f} dB")
        print(f"  SINR improvement: {10*np.log10(sinr_mvdr/sinr_conventional):.2f} dB")
        
        # Null depth analysis
        print(f"\\nInterference Suppression:")
        for i, null_info in enumerate(null_depths):
            print(f"  Interference {i+1} ({null_info['angle']}°):")
            print(f"    Conventional: {null_info['conventional_db']:.1f} dB")
            print(f"    MVDR: {null_info['mvdr_db']:.1f} dB")
            print(f"    Improvement: {null_info['conventional_db'] - null_info['mvdr_db']:.1f} dB")
        
        return {
            'sinr_conventional': 10*np.log10(sinr_conventional),
            'sinr_mvdr': 10*np.log10(sinr_mvdr),
            'sinr_improvement': 10*np.log10(sinr_mvdr/sinr_conventional),
            'null_depths': null_depths
        }

def interactive_menu():
    """Interactive menu for running demonstrations"""
    project = MIMOSystemsProject()
    
    while True:
        print("\\n" + "="*50)
        print("📡📡 MIMO SYSTEMS PROJECT MENU")
        print("="*50)
        print("1. Spatial Diversity (Alamouti)")
        print("2. Spatial Multiplexing (BLAST)")
        print("3. Digital Beamforming")
        print("4. Run All Demonstrations")
        print("5. Exit")
        
        try:
            choice = input("\\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                project.spatial_diversity_project()
            elif choice == '2':
                project.spatial_multiplexing_project()
            elif choice == '3':
                project.beamforming_project()
            elif choice == '4':
                project.run_all_demonstrations()
            elif choice == '5':
                print("\\nExiting MIMO Systems Project. Goodbye!")
                break
            else:
                print("\\nInvalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\\n\\nExiting MIMO Systems Project. Goodbye!")
            break
        except Exception as e:
            print(f"\\nError: {e}")
            print("Please try again.")

if __name__ == "__main__":
    print(__doc__)
    interactive_menu()
