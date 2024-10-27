import pandas as pd
import pandapower as pp
import numpy as np
import os

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
        if line.strip().startswith('-999'):
            break
        if not line.strip() or 'BUS DATA FOLLOWS' in line:
            continue

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
            'tap_bus': int(line[0:4].strip()),      # Tap side bus number
            'z_bus': int(line[5:9].strip()),        # Impedance side bus number
            'area': int(line[10:12].strip()) if line[10:12].strip() else 0,
            'zone': int(line[12:15].strip()) if line[12:15].strip() else 0,
            'circuit': int(line[16:17].strip()) if line[16:17].strip() else 1,
            'type': int(line[18:19].strip()) if line[18:19].strip() else 0,
            'r_pu': float(line[19:29].strip()),     # Branch resistance
            'x_pu': float(line[29:40].strip()),     # Branch reactance
            'b_pu': float(line[40:51].strip()),     # Line charging susceptance
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

def extract_results(net):
    """
    Extract and format pandapower network results.

    Parameters
    ----------
    net : PandaPower Net
        Pandapower network object
    """

    bus_results = pd.DataFrame({
        'bus_name': net.bus['name'],
        'voltage_pu': net.res_bus['vm_pu'],
        'angle_degree': net.res_bus['va_degree'],
        'p_mw': net.res_bus['p_mw'],
        'q_mvar': net.res_bus['q_mvar']
    })
    
    branch_results = pd.DataFrame({
        'from_bus': net.line['from_bus'],
        'to_bus': net.line['to_bus'],
        'loading_percent': net.res_line['loading_percent'],
        'p_from_mw': net.res_line['p_from_mw'],
        'q_from_mvar': net.res_line['q_from_mvar'],
        'p_losses_mw': net.res_line['pl_mw'],
        'q_losses_mvar': net.res_line['ql_mvar']
    })

    return bus_results, branch_results

def run_power_flow(bus_df, branch_df):
    net = create_pp_network(bus_df, branch_df)

    pp.runpp(
        net,
        calculate_voltage_angles=True,
        init='dc',
        enforce_q_lims=False,
        max_iteration=10000,
        tolerance_mva=1e-6,
        algorithm='nr'
    )

    bus_results, branch_results = extract_results(net)
    return bus_results, branch_results


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
    
    bus_start = next(i for i, line in enumerate(lines) if 'BUS DATA FOLLOWS' in line)
    branch_start = next(i for i, line in enumerate(lines) if 'BRANCH DATA FOLLOWS' in line)

    bus_df = parse_bus_data(lines[bus_start:branch_start])
    branch_df = parse_branch_data(lines[branch_start:])

    return bus_df, branch_df

def create_pp_network(bus_df, branch_df):
    """
    Create a PandaPower Network from the bus and branch dataframes.

    Parameters
    ----------
    bus_df : pd.DataFrame
        Dataframe containing the bus data
    branch_df : pd.DataFrame
        Dataframe containing the branch data
    """

    net = pp.create_empty_network()

    bus_indices = {}

    for _, bus in bus_df.iterrows():
        idx = pp.create_bus(
            net,
            vn_kv=bus['base_kv'] if bus['base_kv'] > 0 else 1.0,
            name=bus['name'],
        )

        bus_indices[bus['bus_number']] = idx

        if bus['load_mw'] != 0 or bus['load_mvar'] != 0:
            pp.create_load(
                net,
                bus=idx,
                p_mw=bus['load_mw'],
                q_mvar=bus['load_mvar'],
                name=f"Load {bus['bus_number']}"
            )

        if bus['type'] == 3:  # Slack bus
            pp.create_ext_grid(
                net,
                bus=idx,
                p_mw=bus['gen_mw'],
                vm_pu=bus['voltage_pu'],
                name=f"Slack_{bus['bus_number']}",
                slack=True
            )
        elif bus['type'] == 2:  # PV bus
            pp.create_gen(
                net,
                bus=idx,
                p_mw=bus['gen_mw'],
                vm_pu=bus['voltage_pu'],
                name=f"Gen_{bus['bus_number']}",
                max_q_mvar=bus['max_mvar'] if bus['max_mvar'] != 0 else 9999,
                min_q_mvar=bus['min_mvar'] if bus['min_mvar'] != 0 else -9999
            )

        if bus['shunt_b'] != 0 or bus['shunt_g'] != 0:
            pp.create_shunt(
                net,
                bus=idx,
                q_mvar=-bus['shunt_b'],
                p_mw=bus['shunt_g']
            )

    for _, branch in branch_df.iterrows():
        from_bus = bus_indices[branch['tap_bus']]
        to_bus = bus_indices[branch['z_bus']]

        if branch['type'] == 0:
            r_pu = max(branch['r_pu'], 1e-2)
            x_pu = max(branch['x_pu'], 1e-2)
            pp.create_line_from_parameters(
                net,
                from_bus=from_bus,
                to_bus=to_bus,
                length_km=1,
                r_ohm_per_km=r_pu,
                x_ohm_per_km=x_pu,
                c_nf_per_km=branch['b_pu'], # * 1e9 if branch['b_pu'] else 0, # why?
                max_i_ka=branch['mva_rating1'] if branch['mva_rating1'] > 0 else 1,
                name=f"Line_{branch['tap_bus']}_{branch['z_bus']}"
            )
        else:
            if branch['type'] == 2:
                control_mode = 'v' # Voltage control
                controlled_side = branch['side']
                vm_set_put = branch['max_limit']
            elif branch['type'] == 3:
                control_mode = 'q'  # MVar control
                controlled_side = branch['side']
            elif branch['type'] == 4:
                control_mode = 'p'  # MW control (phase shifter)
                controlled_side = branch['side']
            else:
                control_mode = None  # Fixed tap
                controlled_side = None
            
            pp.create_transformer_from_parameters(
                net,
                hv_bus=from_bus,
                lv_bus=to_bus,
                sn_mva=branch['mva_rating1'] if branch['mva_rating1'] > 0 else 100,
                vn_hv_kv=bus_df.loc[bus_df['bus_number'] == branch['tap_bus'], 'base_kv'].iloc[0],
                vn_lv_kv=bus_df.loc[bus_df['bus_number'] == branch['z_bus'], 'base_kv'].iloc[0],
                vk_percent=branch['x_pu'] * 100,
                vkr_percent=branch['r_pu'] * 100,
                pfe_kw=0,
                i0_percent=0,
                shift_degree=branch['phase_shift'],
                tap_pos=int((branch['tap_ratio'] - branch['min_tap']) / branch['step_size']) 
                        if branch['step_size'] != 0 else 0,
                tap_neutral=1,
                tap_step_percent=branch['step_size'] * 100 if branch['step_size'] != 0 else 0,
                tap_side="hv" if branch['side'] == 1 else "lv",
                tap_min=int((branch['min_tap'] - 1) / branch['step_size']) if branch['step_size'] != 0 else -10,
                tap_max=int((branch['max_tap'] - 1) / branch['step_size']) if branch['step_size'] != 0 else 10,
                name=f"Transformer_{branch['tap_bus']}_{branch['z_bus']}"
            )

            if control_mode and branch['ctrl_bus'] != 0:
                controlled_bus = bus_indices[branch['ctrl_bus']]
                pp.create_transformer_control(
                    net,
                    tid=len(net.transformer) - 1,
                    side="hv" if controlled_side == 1 else "lv",
                    controlled_bus=controlled_bus,
                    vm_set_pu=branch['max_limit'] if control_mode == 'v' else None,
                    vm_lower_pu=branch['min_limit'] if control_mode == 'v' else None,
                    vm_upper_pu=branch['max_limit'] if control_mode == 'v' else None,
                    q_set_mvar=branch['max_limit'] if control_mode == 'q' else None,
                    p_set_mw=branch['max_limit'] if control_mode == 'p' else None
                )

    scaling_factor = 0.01  # Start with 10%
    net.load['p_mw'] *= scaling_factor
    net.load['q_mvar'] *= scaling_factor
    net.gen['p_mw'] *= scaling_factor

    return net

def main(input_file, output_prefix):
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

    bus_df, branch_df = process_ieee_file(input_file)

    bus_df.to_csv(f"networks/{output_prefix}_bus_data.csv", index=False)
    branch_df.to_csv(f"networks/{output_prefix}_branch_data.csv", index=False)

    bus_results, branch_results = run_power_flow(bus_df, branch_df)

    bus_results.to_csv(f"results/{output_prefix}_bus_results.csv", index=False)
    branch_results.to_csv(f"results/{output_prefix}_branch_results.csv", index=False)

    return

if __name__ == "__main__":

    bus_counts = [14, 30, 57]

    for count in bus_counts:
        try:
            print(f"Running the IEEE {count} bus system")
            main(f"inputs/ieee{count}cdf.txt", f"ieee{count}")
        except Exception as e:
            print(f"Error processing ieee{count}cdf.txt: {e}")