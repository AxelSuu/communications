#!/usr/bin/env python3
"""
MIMO Systems Demonstration Script
=================================

This script demonstrates the MIMO (Multiple-Input Multiple-Output) systems
implemented in the comsys.py file. It shows three key MIMO concepts:

1. Spatial Diversity (Alamouti Code)
2. Spatial Multiplexing (BLAST)
3. Digital Beamforming

Author: Advanced Communication Systems Project
"""

import numpy as np
import matplotlib.pyplot as plt
from comsys import CommunicationSystemsProjects

def main():
    """Main demonstration function"""
    print("🚀 MIMO Systems Demonstration")
    print("=" * 40)
    
    # Initialize the communication systems project
    proj = CommunicationSystemsProjects()
    
    # Show available MIMO projects
    print("\n📡 Available MIMO Projects:")
    print("1. Spatial Diversity (Alamouti Code)")
    print("2. Spatial Multiplexing (BLAST)")
    print("3. Digital Beamforming")
    print("\n" + "=" * 40)
    
    # Run individual projects with explanations
    print("\n🔸 Running Spatial Diversity Project...")
    print("This demonstrates how spatial diversity improves reliability")
    print("by using multiple antennas to combat fading effects.")
    proj.spatial_diversity_project()
    
    input("\nPress Enter to continue to Spatial Multiplexing...")
    
    print("\n🔸 Running Spatial Multiplexing Project...")
    print("This demonstrates how spatial multiplexing increases data rate")
    print("by transmitting multiple data streams simultaneously.")
    proj.spatial_multiplexing_project()
    
    input("\nPress Enter to continue to Digital Beamforming...")
    
    print("\n🔸 Running Digital Beamforming Project...")
    print("This demonstrates how beamforming can focus signal energy")
    print("in desired directions while suppressing interference.")
    proj.beamforming_project()
    
    print("\n✅ All MIMO demonstrations completed!")
    print("\nKey MIMO Concepts Demonstrated:")
    print("• Spatial Diversity: Improved reliability through redundancy")
    print("• Spatial Multiplexing: Increased data rate through parallel streams")
    print("• Digital Beamforming: Improved signal quality through directional processing")
    
    print("\n📊 MIMO Benefits:")
    print("• Higher data rates (multiplexing gain)")
    print("• Better reliability (diversity gain)")
    print("• Improved signal quality (beamforming gain)")
    print("• Efficient spectrum utilization")

def run_specific_project():
    """Run a specific MIMO project based on user choice"""
    proj = CommunicationSystemsProjects()
    
    print("\n🚀 MIMO Project Selection")
    print("=" * 30)
    print("1. Spatial Diversity (Alamouti)")
    print("2. Spatial Multiplexing (BLAST)")
    print("3. Digital Beamforming")
    print("4. All MIMO Projects")
    
    try:
        choice = int(input("\nEnter your choice (1-4): "))
        
        if choice == 1:
            proj.spatial_diversity_project()
        elif choice == 2:
            proj.spatial_multiplexing_project()
        elif choice == 3:
            proj.beamforming_project()
        elif choice == 4:
            proj.mimo_projects()
        else:
            print("Invalid choice! Please select 1-4.")
            
    except ValueError:
        print("Please enter a valid number.")

if __name__ == "__main__":
    print("MIMO Systems Demonstration")
    print("=" * 30)
    print("Choose demonstration mode:")
    print("1. Full demonstration (all projects)")
    print("2. Select specific project")
    
    try:
        mode = int(input("Enter choice (1-2): "))
        
        if mode == 1:
            main()
        elif mode == 2:
            run_specific_project()
        else:
            print("Invalid choice! Running full demonstration...")
            main()
            
    except ValueError:
        print("Invalid input! Running full demonstration...")
        main()
