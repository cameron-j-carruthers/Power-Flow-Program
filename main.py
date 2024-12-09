import pandas as pd
import pandapower as pp
import numpy as np
import os
import pandapower.networks as pn

def parse_bus_data(lines):
    """
    Parse the bus data section from the IEEE Common Data Format file and convert
    to a dataframe

    Parameters
    ----------
    lines : list
        List of lines from the IEEE Common Data Format file
    """

    bus_data = []

    for line in lines:
        if line.strip().startswith('-999'): # this is the last line in the bus data section
            break
        if not line.strip() or 'BUS DATA FOLLOWS' in line: # skip empty lines and the header
            continue
        
        # set up the info based on the documentation
        data = {
            'bus_number': int(line[0:4].strip()),
            'name': line[5:17].strip(),
            'area': int(line[18:20].strip()),
            'zone': int(line[20:23].strip()) if line[20:23].strip() else 0,
            'type': int(line[24:26].strip()),
            'voltage_pu': float(line[27:33].strip()),
            'angle_degree': float(line[33:40].strip()),
            'load_mw': float(line[40:49].strip()),
            'load_mvar': float(line[49:59].strip()),
            'gen_mw': float(line[59:67].strip()),
            'gen_mvar': float(line[67:75].strip()),
            'base_kv': float(line[76:83].strip()) if line[76:83].strip() else 0.0,
            'desired_volts': float(line[84:90].strip()) if line[84:90].strip() else 1.0,
            'max_mvar': float(line[90:98].strip()) if line[90:98].strip() else 0.0,
            'min_mvar': float(line[98:106].strip()) if line[98:106].strip() else 0.0,
            'shunt_g': float(line[106:114].strip()) if line[106:114].strip() else 0.0,
            'shunt_b': float(line[114:122].strip()) if line[114:122].strip() else 0.0,
            'remote_crtl': int(line[123:127].strip()) if line[123:127].strip() else 0,
        }

        bus_data.append(data)
    
    return pd.DataFrame(bus_data)

def parse_branch_data(lines):
    """
    Parse the branch data section from the IEEE Common Data Format file and convert
    to a dataframe

    Parameters
    ----------
    lines : list
        List of lines from the IEEE Common Data Format file
    """

    branch_data = []
    for line in lines:
        if line.strip().startswith('-999'):
            break
        if not line.strip() or 'BRANCH DATA FOLLOWS' in line:
            continue
            
        # Extract data according to the format specification
        data = {
            'tap_bus': int(line[0:4].strip()),
            'z_bus': int(line[5:9].strip()),
            'area': int(line[10:12].strip()) if line[10:12].strip() else 0,
            'zone': int(line[12:15].strip()) if line[12:15].strip() else 0,
            'circuit': int(line[16:17].strip()) if line[16:17].strip() else 1,
            'type': int(line[18:19].strip()) if line[18:19].strip() else 0,
            'r_pu': float(line[19:29].strip()),
            'x_pu': float(line[29:40].strip()),
            'b_pu': float(line[40:51].strip()),
            'mva_rating1': float(line[51:56].strip()) if line[51:56].strip() else 0,
            'mva_rating2': float(line[56:63].strip()) if line[56:63].strip() else 0,
            'mva_rating3': float(line[63:67].strip()) if line[63:67].strip() else 0,
            'ctrl_bus': int(line[68:72].strip()) if line[68:72].strip() else 0,
            'side': int(line[73:74].strip()) if line[73:74].strip() else 0,
            'tap_ratio': float(line[76:82].strip()) if line[76:82].strip() else 1.0,
            'phase_shift': float(line[83:90].strip()) if line[83:90].strip() else 0.0,
            'min_tap': float(line[90:97].strip()) if line[90:97].strip() else 0.0,
            'max_tap': float(line[97:104].strip()) if line[97:104].strip() else 0.0,
            'step_size': float(line[105:111].strip()) if line[105:111].strip() else 0.0,
            'min_limit': float(line[112:118].strip()) if line[112:118].strip() else 0.0,
            'max_limit': float(line[118:126].strip()) if line[118:126].strip() else 0.0,
        }
        branch_data.append(data)
    
    return pd.DataFrame(branch_data)

def process_ieee_file(file_path):
    """
    Process IEEE Common Data Format file and return the input data for pandapower

    Parameters
    ----------
    file_path : str
        Path to the IEEE Common Data Format file
    """

    with open(file_path, 'r') as file:
        lines = file.readlines()

    mva_base = float(lines[0][31:37].strip())
    
    bus_start = next(i for i, line in enumerate(lines) if 'BUS DATA FOLLOWS' in line)
    branch_start = next(i for i, line in enumerate(lines) if 'BRANCH DATA FOLLOWS' in line)

    bus_df = parse_bus_data(lines[bus_start:branch_start])
    branch_df = parse_branch_data(lines[branch_start:])

    return bus_df, branch_df, mva_base

def create_pp_network(bus_df, branch_df, mva_base=100.0):
    """
    Create a PandaPower Network from bus and branch dataframes.

    Parameters
    ----------
    bus_df : pd.DataFrame
        Dataframe containing bus data with columns:
        bus_number, name, type, voltage_pu, angle_degree, load_mw, load_mvar,
        gen_mw, gen_mvar, base_kv, max_mvar, min_mvar, shunt_g, shunt_b
    branch_df : pd.DataFrame
        Dataframe containing branch data with columns:
        tap_bus, z_bus, type, r_pu, x_pu, b_pu, mva_rating1, ctrl_bus,
        side, tap_ratio, phase_shift, min_tap, max_tap, step_size, 
        min_limit, max_limit
    mva_base : float, optional
        Base MVA for the system, defaults to 100.0

    Returns
    -------
    pandapower.auxiliary.pandapowerNet
        The created PandaPower network
    """
    import pandapower as pp
    import numpy as np

    # Create empty network
    net = pp.create_empty_network(sn_mva=mva_base)
    
    # Dictionary to store bus indices
    bus_indices = {}

    # Process buses
    for _, bus in bus_df.iterrows():
    #     # Create the bus
    #     idx = pp.create_bus(
    #         net, 
    #         vn_kv=float(bus['base_kv']),
    #         name=str(bus['name']),
    #         index=int(bus['bus_number']),
    #         type={0: 'b', 1: 'b', 2: 'pv', 3: 'n'}.get(bus['type'], 'b'),
    #         zone=int(bus['zone']),
    #         in_service=True
    #     )

    #     if bus['load_mw'] > 0 or bus['load_mvar'] > 0:
    #         pp.create_load(
    #             net,
    #             bus=bus['bus_number'],
    #             p_mw=bus['load_mw'],
    #             q_mvar=bus['load_mvar'],
    #             in_service=True
    #         )

    #     if bus['gen_mw'] > 0 or bus['gen_mvar'] > 0:
    #         pp.create_sgen(
    #             net,
    #             bus=bus['bus_number'],
    #             p_mw=bus['gen_mw'],
    #             vm_pu=bus['voltage_pu'],
    #             min_q_mvar=bus['min_mvar'],
    #             max_q_mvar=bus['max_mvar'],
    #             in_service=True
    #         )
        
    #     if bus['type'] == 3:
    #         pp.create_ext_grid(
    #             net,
    #             bus=bus['bus_number'],
    #             vm_pu=bus['voltage_pu'],
    #             va_degree=bus['angle_degree'],
    #         )
    
    # for _, branch in branch_df.iterrows():
    #     from_bus = branch['tap_bus']
    #     to_bus = branch['z_bus']
    #     r_pu = branch['r_pu']
    #     x_pu = branch['x_pu']
    #     b_pu = branch['b_pu']
    #     max_i_ka = branch['mva_rating1'] / (branch['base_kv'] * 1.732) # TODO: why 1.732?
    #     name = f"Circuit {branch['circuit']}" if branch['circuit'] else "Branch"

    #     if branch['type'] == 0:
    #         pp.create_line_from_parameters(
    #             net,
    #             from_bus=from_bus,
    #             to_bus=to_bus,
    #             length_km=1.0, # TODO: Not sure what to use
    #             r_ohm_per_km=r_pu,
    #             x_ohm_per_km=x_pu,
    #             c_nf_per_km=b_pu,   
    #             max_i_ka=max_i_ka,
    #             name=name,
    #             in_service=True
    #         )
    #     elif branch['type'] == 1:
    #         pp.create_transformer_from_parameters(
    #             net,
    #             hv_bus=from_bus,
    #             lv_bus=to_bus,
    #             sn_mva=branch['mva_rating1'],
    #             vn_hv_kv=bus_df.loc[bus_df['bus_number'] == from_bus, 'base_kv'].iloc[0],
    #             vn_lv_kv=bus_df.loc[bus_df['bus_number'] == to_bus, 'base_kv'].iloc[0],
    #             vk_percent=x_pu * 100,
    #             vkr_percent=r_pu * 100,
    #             pfe_kw=0,
    #             i0_percent=0,
    #             shift_degree=branch['phase_shift'],
                
    #         )

        # Create bus - ensure proper voltage base
        idx = pp.create_bus(
            net,
            vn_kv=float(bus['base_kv']),  # Must be positive
            name=str(bus['name']),
        )
        bus_indices[bus['bus_number']] = idx

        # Add load if present
        if abs(bus['load_mw']) > 0 or abs(bus['load_mvar']) > 0:
            pp.create_load(
                net,
                bus=idx,
                p_mw=float(bus['load_mw']),  # Don't divide by mva_base
                q_mvar=float(bus['load_mvar']),  # Don't divide by mva_base
                name=f"Load_{bus['bus_number']}"
            )

        # Process generators based on bus type
        if bus['type'] == 3:  # Slack bus
            pp.create_ext_grid(
                net,
                bus=idx,
                vm_pu=float(bus['voltage_pu']),
                va_degree=float(bus['angle_degree']),  # Add angle
                name=f"Slack_{bus['bus_number']}"
            )
        elif bus['type'] == 2:  # PV bus
            pp.create_gen(
                net,
                bus=idx,
                p_mw=float(bus['gen_mw']),  # Don't divide by mva_base
                vm_pu=float(bus['voltage_pu']),
                name=f"Gen_{bus['bus_number']}",
                max_q_mvar=float(bus['max_mvar']) if bus['max_mvar'] != 0 else None,
                min_q_mvar=float(bus['min_mvar']) if bus['min_mvar'] != 0 else None
            )
        elif bus['type'] == 1 and (abs(bus['gen_mw']) > 0 or abs(bus['gen_mvar']) > 0):  # PQ bus with generation
            pp.create_sgen(
                net,
                bus=idx,
                p_mw=float(bus['gen_mw']),
                q_mvar=float(bus['gen_mvar']),
                name=f"SGen_{bus['bus_number']}"
            )

        # Add shunt if present
        if abs(bus['shunt_b']) > 0 or abs(bus['shunt_g']) > 0:
            pp.create_shunt(
                net,
                bus=idx,
                q_mvar=-float(bus['shunt_b']) * (bus['base_kv'] ** 2),  # Convert to actual MVAr
                p_mw=float(bus['shunt_g']) * (bus['base_kv'] ** 2)  # Convert to actual MW
            )

    #Process branches
    for _, branch in branch_df.iterrows():
        from_bus = bus_indices[branch['tap_bus']]
        to_bus = bus_indices[branch['z_bus']]

        if branch['type'] == 0:  # Line
            # Convert impedance from per-unit to physical units
            vbase = bus_df.loc[bus_df['bus_number'] == branch['tap_bus'], 'base_kv'].iloc[0]
            zbase = (vbase ** 2) / mva_base
            r_ohm = float(branch['r_pu']) * zbase
            x_ohm = float(branch['x_pu']) * zbase
            
            # Convert susceptance to nanosiemens
            b_pu = float(branch['b_pu'])
            c_nf = b_pu * 1e9 / (2 * np.pi * 50 * zbase)  # Assuming 50 Hz
            
            pp.create_line_from_parameters(
                net,
                from_bus=from_bus,
                to_bus=to_bus,
                length_km=1.0,
                r_ohm_per_km=r_ohm,
                x_ohm_per_km=x_ohm,
                c_nf_per_km=c_nf,
                max_i_ka=float(branch['mva_rating1']) / (vbase * np.sqrt(3)) if branch['mva_rating1'] > 0 else np.nan,
                name=f"Line_{branch['tap_bus']}_{branch['z_bus']}"
            )
        else:  # Transformer
            from_vbase = bus_df.loc[bus_df['bus_number'] == branch['tap_bus'], 'base_kv'].iloc[0]
            to_vbase = bus_df.loc[bus_df['bus_number'] == branch['z_bus'], 'base_kv'].iloc[0]
            
            # Calculate tap position
            tap_neutral = 1.0
            tap_step = float(branch['step_size']) if branch['step_size'] != 0 else 0.0
            if tap_step > 0:
                tap_pos = int(round((branch['tap_ratio'] - tap_neutral) / tap_step))
                tap_min = int(round((branch['min_tap'] - tap_neutral) / tap_step))
                tap_max = int(round((branch['max_tap'] - tap_neutral) / tap_step))
            else:
                tap_pos = 0
                tap_min = -10
                tap_max = 10

            pp.create_transformer_from_parameters(
                net,
                hv_bus=from_bus,
                lv_bus=to_bus,
                sn_mva=float(branch['mva_rating1']) if branch['mva_rating1'] > 0 else mva_base,
                vn_hv_kv=from_vbase,
                vn_lv_kv=to_vbase,
                vk_percent=float(branch['x_pu']) * 100,  # Convert to percentage
                vkr_percent=float(branch['r_pu']) * 100,  # Convert to percentage
                pfe_kw=0,
                i0_percent=0,
                shift_degree=float(branch['phase_shift']),
                tap_pos=tap_pos,
                tap_neutral=tap_neutral,
                tap_step_percent=tap_step * 100,
                tap_side="hv" if branch['side'] == 1 else "lv",
                tap_min=tap_min,
                tap_max=tap_max,
                name=f"Transformer_{branch['tap_bus']}_{branch['z_bus']}"
            )

            # Add transformer control if specified
            if branch['ctrl_bus'] != 0:
                control_mode = {2: 'v', 3: 'q', 4: 'p'}.get(branch['type'])
                if control_mode:
                    controlled_bus = bus_indices[branch['ctrl_bus']]
                    pp.create_transformer_control(
                        net,
                        tid=len(net.transformer) - 1,
                        side="hv" if branch['side'] == 1 else "lv",
                        controlled_bus=controlled_bus,
                        vm_set_pu=float(branch['max_limit']) if control_mode == 'v' else None,
                        vm_lower_pu=float(branch['min_limit']) if control_mode == 'v' else None,
                        vm_upper_pu=float(branch['max_limit']) if control_mode == 'v' else None,
                        q_set_mvar=float(branch['max_limit']) if control_mode == 'q' else None,
                        p_set_mw=float(branch['max_limit']) if control_mode == 'p' else None
                    )

    return net

def run_power_flow(bus_df, branch_df, mva_base, net=None):
    """
    Run the power flow on the network
    
    Parameters
    ----------
    bus_df : pd.DataFrame
        Dataframe containing the bus data
    
    branch_df : pd.DataFrame
        Dataframe containing the branch data

    mva_base : float
        Base MVA of the system
    """

    if net is None:
        net = create_pp_network(bus_df, branch_df, mva_base)

    pp.runpp(
        net,
        calculate_voltage_angles=True,
        init='dc',
        enforce_q_lims=False,
        max_iteration=10000,
        tolerance_mva=1e-6,
        algorithm='nr'
    )

    bus_results, branch_results, slack_results, pv_bus_results = extract_results(net)
    return bus_results, branch_results, slack_results, pv_bus_results

def extract_results(net):
    """
    Extract and format pandapower network results using bus numbers instead of names.

    Parameters
    ----------
    net : PandaPower Net
        Pandapower network object
    """
    
    # Bus Results
    bus_results = pd.DataFrame({
        'bus_number': net.bus.index.values, # organize by the bus number
        'voltage_pu': net.res_bus['vm_pu'], # get the voltage magnitude in pu
        'angle_degree': net.res_bus['va_degree'], # get the voltage angle in degrees
        'p_mw': net.res_bus['p_mw'], # get the active power in MW
        'q_mvar': net.res_bus['q_mvar'] # get the reactive power in MVar
    })
    
    # Results for the lines
    line_results = pd.DataFrame()
    if len(net.line) > 0:
        line_results = pd.DataFrame({
            'from_bus': net.line['from_bus'].values,   
            'to_bus': net.line['to_bus'].values, 
            'loading_percent': net.res_line['loading_percent'],
            'p_from_mw': net.res_line['p_from_mw'],
            'q_from_mvar': net.res_line['q_from_mvar'],
            'p_losses_mw': net.res_line['pl_mw'],
            'q_losses_mvar': net.res_line['ql_mvar']
        })
    
    # Results for the transformers
    trafo_results = pd.DataFrame()
    if len(net.trafo) > 0:
        trafo_results = pd.DataFrame({
            'from_bus': net.trafo['hv_bus'].values, 
            'to_bus': net.trafo['lv_bus'].values,
            'loading_percent': net.res_trafo['loading_percent'],
            'p_from_mw': net.res_trafo['p_hv_mw'],
            'q_from_mvar': net.res_trafo['q_hv_mvar'],
            'p_losses_mw': net.res_trafo['pl_mw'],
            'q_losses_mvar': net.res_trafo['ql_mvar']
        })
    
    branch_results = pd.concat([line_results, trafo_results], ignore_index=True)

    # Slack Bus Results
    slack_results = pd.DataFrame({
        'bus_number': net.ext_grid['bus'].values, 
        'slack_p_mw': net.res_ext_grid['p_mw'].values,
        'slack_q_mvar': net.res_ext_grid['q_mvar'].values
    })

    # PV Bus Results
    pv_bus_results = pd.DataFrame({
        'bus_number': net.gen['bus'].values,
        'pv_q_mvar': net.res_gen['q_mvar'].values
    }) if len(net.gen) > 0 else pd.DataFrame(columns=['bus_number', 'pv_q_mvar'])

    return bus_results, branch_results, slack_results, pv_bus_results


def main(input_file, output_prefix, count, compare_with_prebuilt=True):
    """
    Main function to process the IEEE Common Data Format file and run the power flow

    Parameters
    ----------
    input_file : str
        Path to the IEEE Common Data Format file
    output_prefix : str
        Prefix for the output files
    """

    # Check if output folder exists, if not, create
    if not os.path.exists("results"):
        os.makedirs("results")

    if not os.path.exists("networks"):
        os.makedirs("networks")

    bus_df, branch_df, mva_base = process_ieee_file(input_file)

    bus_df.to_csv(f"networks/{output_prefix}_bus_data.csv", index=False)
    branch_df.to_csv(f"networks/{output_prefix}_branch_data.csv", index=False)

    bus_results, branch_results, slack_results, pv_bus_results = run_power_flow(bus_df, branch_df, mva_base)

    pb_bus_results, pb_branch_result, pb_slack_results, pb_pv_bus_results = run_power_flow(None, None, mva_base, load_prebuilt_network(count))

    print(bus_results.head(10))
    print(pb_bus_results.head(10))

    # take the differences of voltage_pu, angle_degree, p_mw and q_mvar between the two and take the mean abs difference
    print("Mean absolute difference of voltage_pu: ", np.abs(bus_results['voltage_pu'] - pb_bus_results['voltage_pu']).mean())
    print("Mean absolute difference of angle_degree: ", np.abs(bus_results['angle_degree'] - pb_bus_results['angle_degree']).mean())
    print("Mean absolute difference of p_mw: ", np.abs(bus_results['p_mw'] - pb_bus_results['p_mw']).mean())
    print("Mean absolute difference of q_mvar: ", np.abs(bus_results['q_mvar'] - pb_bus_results['q_mvar']).mean())

    
    return

def load_prebuilt_network(count):
    """
    Load a prebuilt network from the pandapower networks module

    Parameters
    ----------
    count : int
        Number of buses in the network
    """

    if count == 14:
        return pn.case14()
    elif count == 30:
        return pn.case_ieee30()
    elif count == 57:
        return pn.case57()
    else:
        raise ValueError("Invalid network count")

if __name__ == "__main__":

    bus_counts = [30]

    for count in bus_counts:
        try:
            print(f"\nRunning the IEEE {count} bus system")
            main(f"inputs/ieee{count}cdf.txt", f"ieee{count}", count)
        except Exception as e:
            print(f"Error processing ieee{count}cdf.txt: {e}")

