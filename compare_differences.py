import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np
from main import process_ieee_file, create_pp_network

import pandas as pd
import numpy as np
from scipy.io import loadmat
import pandapower as pp

def compare_power_flow_results(pp_net, matlab_file_path, solution_idx=0):
    """
    Compare power flow results between pandapower network and MATLAB solutions.
    
    Parameters:
    -----------
    pp_net : pandapower.auxiliary.pandapowerNet
        The pandapower network with completed power flow calculation
    matlab_file_path : str
        Path to the MATLAB .mat file containing the solutions
    solution_idx : int
        Index of the solution to compare (0 to 471)
        
    Returns:
    --------
    pandas.DataFrame
        Comparison of results
    """
    # Load MATLAB results
    mat_data = loadmat(matlab_file_path)
    
    # Extract voltage magnitudes and angles from MATLAB data
    matlab_vmag = mat_data['Vact']['mag'][0][0][:, solution_idx]
    matlab_vang = mat_data['Vact']['ang'][0][0][:, solution_idx]
    
    # Get base MVA
    base_mva = float(mat_data['mpc'][0][0][1][0][0])  # Should be 100
    
    # Extract bus and generator data
    mpc_bus = mat_data['mpc'][0][0][2]  # 30x13 array with bus data
    mpc_gen = mat_data['mpc'][0][0][3]  # 6x21 array with generator data
    
    # Create comparison dataframe
    comparison = pd.DataFrame()
    
    # Add bus numbers (MATLAB uses 1-based indexing)
    comparison['bus_number'] = mpc_bus[:, 0].astype(int) - 1  # Convert to 0-based indexing
    
    # Add bus types based on MATPOWER convention
    # 1=PQ, 2=PV, 3=ref, 4=isolated
    bus_types = {1: 'PQ', 2: 'PV', 3: 'Slack', 4: 'Isolated'}
    comparison['bus_type'] = [bus_types[int(t)] for t in mpc_bus[:, 1]]
    
    # Add voltage magnitudes
    comparison['voltage_mag_pp'] = pp_net.res_bus.vm_pu.values
    comparison['voltage_mag_matlab'] = matlab_vmag
    comparison['voltage_mag_diff'] = comparison['voltage_mag_pp'] - comparison['voltage_mag_matlab']
    
    # Add voltage angles
    comparison['voltage_ang_pp'] = pp_net.res_bus.va_degree.values
    comparison['voltage_ang_matlab'] = np.degrees(matlab_vang)  # Convert radians to degrees
    comparison['voltage_ang_diff'] = comparison['voltage_ang_pp'] - comparison['voltage_ang_matlab']
    
    # Add power values from pandapower
    comparison['p_mw_pp'] = -pp_net.res_bus.p_mw.values / (base_mva)
    comparison['q_mvar_pp'] = -pp_net.res_bus.q_mvar.values / (base_mva)
    
    # Add load data from MATLAB (columns 2 and 3 are Pd and Qd)
    comparison['p_mw_matlab_load'] = -mpc_bus[:, 2] / (base_mva)  # Negative because loads are consumption
    comparison['q_mvar_matlab_load'] = -mpc_bus[:, 3] / (base_mva)
    
    # Initialize generator power columns
    comparison['p_mw_matlab_gen'] = 0.0
    comparison['q_mvar_matlab_gen'] = 0.0
    
    # Add generator power injections
    for gen_idx in range(mpc_gen.shape[0]):
        bus_idx = int(mpc_gen[gen_idx, 0]) - 1  # Convert to 0-based indexing
        # Find the row in comparison dataframe for this bus
        bus_mask = comparison['bus_number'] == bus_idx
        comparison.loc[bus_mask, 'p_mw_matlab_gen'] = mpc_gen[gen_idx, 1] / (base_mva)
        comparison.loc[bus_mask, 'q_mvar_matlab_gen'] = mpc_gen[gen_idx, 2] / (base_mva)
    
    # Calculate net power injections
    comparison['p_mw_matlab_net'] = comparison['p_mw_matlab_gen'] + comparison['p_mw_matlab_load']
    comparison['q_mvar_matlab_net'] = comparison['q_mvar_matlab_gen'] + comparison['q_mvar_matlab_load']

    # Calculate power differences
    comparison['p_mw_diff'] = comparison['p_mw_pp'] - comparison['p_mw_matlab_net']
    comparison['q_mvar_diff'] = comparison['q_mvar_pp'] - comparison['q_mvar_matlab_net']
    
    # Calculate statistics
    stats = pd.DataFrame({
        'max_voltage_diff': [comparison['voltage_mag_diff'].abs().max()],
        'max_angle_diff': [comparison['voltage_ang_diff'].abs().max()],
        'max_p_diff': [comparison['p_mw_diff'].abs().max()],
        'max_q_diff': [comparison['q_mvar_diff'].abs().max()],
        'solution_index': [solution_idx],
        'total_solutions': [mat_data['Vact']['mag'][0][0].shape[1]],
        'base_mva': [base_mva]
    })

    comparison = comparison[[
        'bus_number', 'bus_type', 'voltage_mag_pp', 'voltage_mag_matlab', 'voltage_mag_diff',
        'voltage_ang_pp', 'voltage_ang_matlab', 'voltage_ang_diff',
        'p_mw_pp', 'p_mw_matlab_net', 'p_mw_diff',
        'q_mvar_pp', 'q_mvar_matlab_net', 'q_mvar_diff'
    ]]
    
    return comparison, stats

def print_solution_summary(comparison, stats):
    """Print a summary of the solution comparison"""
    print("\nSolution Comparison Summary:")
    print(f"Using solution {stats['solution_index'].iloc[0]} of {stats['total_solutions'].iloc[0]}")
    print(f"Base MVA: {stats['base_mva'].iloc[0]}")
    
    print(f"\nMaximum Differences:")
    print(f"Voltage magnitude: {stats['max_voltage_diff'].iloc[0]:.6f} pu")
    print(f"Voltage angle: {stats['max_angle_diff'].iloc[0]:.6f} degrees")
    print(f"Active power: {stats['max_p_diff'].iloc[0]:.6f} MW")
    print(f"Reactive power: {stats['max_q_diff'].iloc[0]:.6f} MVAr")
    
    print("\nBus Type Summary:")
    print(comparison.groupby('bus_type').size().to_string())
    
    # Print details for special buses
    for bus_type in ['Slack', 'PV']:
        print(f"\n{bus_type} Bus Details:")
        bus_data = comparison[comparison.bus_type == bus_type]
        for _, bus in bus_data.iterrows():
            print(f"Bus {int(bus['bus_number'])}:")
            print(f"V = {bus['voltage_mag_pp']:.4f} ∠ {bus['voltage_ang_pp']:.2f}°")
            print(f"P = {bus['p_mw_pp']:.2f} MW")
            print(f"Q = {bus['q_mvar_pp']:.2f} MVAr")

    print("\nComparison Data:")
    print(comparison)

    comparison.to_csv("comparison.csv")


if __name__ == '__main__':
    bus_df, branch_df, mva_base = process_ieee_file('inputs/ieee30cdf.txt')
    net = create_pp_network(bus_df, branch_df, mva_base)
    #net = pn.case_ieee30()

    pp.runpp(net)

    min_i = 0
    min_max_voltage_diff = float('inf')

    for i in range(472):
        comparison, stats = compare_power_flow_results(net, "matlab_solutions/case30_solu/case30_100load.mat", i)

        diff = stats['max_voltage_diff'].iloc[0] + stats['max_angle_diff'].iloc[0] + stats['max_p_diff'].iloc[0] + stats['max_q_diff'].iloc[0]

        if diff < min_max_voltage_diff:
            min_i = i
            min_max_voltage_diff = diff
    
    comparison, stats = compare_power_flow_results(net, "matlab_solutions/case30_solu/case30_100load.mat", min_i)

    print_solution_summary(comparison, stats)