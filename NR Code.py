#!/usr/bin/env python3
import pandapower as pp
import pandapower.networks as pn

net = pp.create_empty_network()

# Create the buses
pp.create_bus(net, vn_kv=1.0, name="Bus 0", type="slack")
pp.create_bus(net, vn_kv=1.0, name="Bus 1")
pp.create_bus(net, vn_kv=1.0, name="Bus 2")
pp.create_bus(net, vn_kv=1.0, name="Bus 3")
pp.create_bus(net, vn_kv=1.0, name="Bus 4")

# This is the slack bus
pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0, max_q_mvar=9999, min_q_mvar=-9999)

# These are the generators
pp.create_gen(net, bus=1, p_mw=0.883, vm_pu=1.0, name="G2", controllable=False)
pp.create_gen(net, bus=2, p_mw=0.2076, vm_pu=1.0, name="G3", controllable=False)

pp.create_shunt(net, bus=3, p_mw=0, q_mvar=1.0, name="Shunt D4")
pp.create_shunt(net, bus=4, p_mw=0, q_mvar=0.8, name="Shunt D5")

# These are the loads
pp.create_load(net, bus=2, p_mw=0.2, q_mvar=0.1, name="Load D3")
pp.create_load(net, bus=3, p_mw=1.7137, q_mvar=0.5983, name="Load D4")
pp.create_load(net, bus=4, p_mw=1.7355, q_mvar=0.5496, name="Load D5")

# These are the lines from that diagram assuming I understand it correctly...
pp.create_line_from_parameters(net, from_bus=0, to_bus=1, length_km=1,
                               r_ohm_per_km=0.0099, x_ohm_per_km=0.099,
                               c_nf_per_km=0, max_i_ka=1.0)
pp.create_line_from_parameters(net, from_bus=0, to_bus=3, length_km=1,
                               r_ohm_per_km=0.0099, x_ohm_per_km=0.099,
                               c_nf_per_km=0, max_i_ka=1.0)
pp.create_line_from_parameters(net, from_bus=1, to_bus=2, length_km=1,
                               r_ohm_per_km=0.0099, x_ohm_per_km=0.099,
                               c_nf_per_km=0, max_i_ka=1.0)
pp.create_line_from_parameters(net, from_bus=1, to_bus=3, length_km=1,
                               r_ohm_per_km=0.0099, x_ohm_per_km=0.099,
                               c_nf_per_km=0, max_i_ka=1.0)
pp.create_line_from_parameters(net, from_bus=2, to_bus=4, length_km=1,
                               r_ohm_per_km=0.0099, x_ohm_per_km=0.099,
                               c_nf_per_km=0, max_i_ka=1.0)
pp.create_line_from_parameters(net, from_bus=3, to_bus=4, length_km=1,
                               r_ohm_per_km=0.0099, x_ohm_per_km=0.099,
                               c_nf_per_km=0, max_i_ka=1.0)

# Run power flow
try:
    pp.runpp(net, max_iteration=5000, init='flat', tolerance_mva=1e-2, algorithm='nr')
except pp.LoadflowNotConverged:
    print("Power flow did not converge.")

# Output the results: slack bus power and line losses
print("Slack Bus Power (S1):")
print(net.res_ext_grid)

print("\nTotal Line Losses:")
print(net.res_line)

v4 = net.res_bus.loc[3, 'vm_pu']
v5 = net.res_bus.loc[4, 'vm_pu']
theta2 = net.res_bus.loc[1, 'va_degree']
theta3 = net.res_bus.loc[2, 'va_degree']
theta4 = net.res_bus.loc[3, 'va_degree']
theta5 = net.res_bus.loc[4, 'va_degree']

print(f"\nVoltage at Bus 4 (V4): {v4} p.u.")
print(f"Voltage at Bus 5 (V5): {v5} p.u.")
print(f"Theta at Bus 2 (theta2): {theta2} degrees")
print(f"Theta at Bus 3 (theta3): {theta3} degrees")
print(f"Theta at Bus 4 (theta4): {theta4} degrees")
print(f"Theta at Bus 5 (theta5): {theta5} degrees")

