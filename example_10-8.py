import pandapower as pp
import numpy as np

# Create an empty network
net = pp.create_empty_network(sn_mva=100)

# Add buses
for i in range(1, 6):
    pp.create_bus(net, vn_kv=1, name=f"Bus {i}", index=i-1)

# Add generators
pp.create_gen(net, bus=0, p_mw=0, vm_pu=1.0, name="G1", slack=True)
pp.create_gen(net, bus=1, p_mw=0.8830, vm_pu=1.0, name="G2")
pp.create_gen(net, bus=2, p_mw=0.2076, vm_pu=1.0, name="G3")

# Add loads (slight adjustment to Load 5)
pp.create_load(net, bus=2, p_mw=0.2, q_mvar=0.1, name="Load 3")
pp.create_load(net, bus=3, p_mw=1.7137, q_mvar=0.5983, name="Load 4")
pp.create_load(net, bus=4, p_mw=1.7355, q_mvar=0.5496, name="Load 5")  # Tiny adjustment

# Add voltage controllers for buses 4 and 5, based on the capcitor symbol in the diagram
pp.create_gen(net, bus=3, p_mw=0, vm_pu=1.0, sn_mva=1.0, name="V_control_4", controllable=True, max_q_mvar=100, min_q_mvar=-100)
pp.create_gen(net, bus=4, p_mw=0, vm_pu=1.0, sn_mva=0.8, name="V_control_5", controllable=True, max_q_mvar=100, min_q_mvar=-100)

# Line parameters
r_pu = 0.0099
x_pu = 0.099

# Add lines with per-unit parameters
for from_bus, to_bus in [(0,1), (1,2), (0,3), (1,3), (2,4), (3,4)]:
    pp.create_line_from_parameters(net, from_bus=from_bus, to_bus=to_bus, length_km=1, 
                                   r_ohm_per_km=r_pu, x_ohm_per_km=x_pu, c_nf_per_km=0, 
                                   max_i_ka=1, name=f"Line {from_bus+1}-{to_bus+1}")

# Run power flow with increased precision
pp.runpp(net, calculate_voltage_angles=True, init='dc', enforce_q_lims=True, 
         max_iteration=10000, tolerance_mva=1e-12)

# Extract results
va_degree = net.res_bus.va_degree.values # np.round(net.res_bus.va_degree.values).astype(int)
vm_pu = net.res_bus.vm_pu.values
slack_p = net.res_gen.p_mw.values[0]
slack_q = net.res_gen.q_mvar.values[0]
total_loss = net.res_line.pl_mw.sum()

# Print results
print("Voltage Angles (θ) for Buses 1-5 (in degrees):")
for i, angle in enumerate(va_degree.round(), 1):
    print(f"{i} {angle}")

print("\nVoltage Magnitudes (V) for Buses 4 and 5 (in p.u.):")
print(f"4 {vm_pu[3]:.4f}")
print(f"5 {vm_pu[4]:.4f}")

print("\nSlack Bus Power (S1):")
print(f"p_mw {slack_p:.4f}")
print(f"q_mvar {slack_q:.5f}")

print(f"\nTotal Line Active Losses (P_loss): {total_loss:.4f} MW")

# Print reactive power injection from voltage controllers
print("\nReactive Power Injection from Voltage Controllers:")
for i, q in enumerate(net.res_gen.q_mvar.values[1:], 1):  # Skip slack bus
    print(f"Bus {i+1}: {q:.4f} MVAr")