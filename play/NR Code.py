import pandapower as pp
import pandapower.networks as pn


net = pp.create_empty_network()

"""
Create Bus parameters:

net       : The network within which the bus is created. <- REQUIRED
vn_kv     : The grid voltage level. <- REQUIRED
name      : Name of the bus
Index     : Give a specified ID to the bus.
type      : n - node, b - busbar, m - muff
zone      : Grid region to group buses if needed
in_service: True if bus is in service, False if not
max_vm_pu : Maximum voltage magnitude at the bus in pu
min_vm_pu : Minimum voltage magnitude at the bus in pu
coords    : busbard coordinates to plot the bus with multiple points, typically a list of tuples
"""
pp.create_bus(net, vn_kv=1, name="Bus 1", index=1)
pp.create_bus(net, vn_kv=1, name="Bus 2", index=2)
pp.create_bus(net, vn_kv=1, name="Bus 3", index=3)
pp.create_bus(net, vn_kv=1, name="Bus 4", index=4)
pp.create_bus(net, vn_kv=1, name="Bus 5", index=5)

"""
create_ext_grid
Creates an external grid connection.
External grids represent the higher level power grid connection and are modelled as the slack bus in the power flow calculation.

net                              : pandapower network <- REQUIRED
bus (int)                        : bus where the slack is connected <- REQUIRED
vm_pu (float, default 1.0)       : voltage at the slack node in per unit
va_degree (float, default 0.)    : voltage angle at the slack node in degrees*
name (string, default None)      : name of of the external grid
in_service (boolean)             : True for in_service or False for out of service
s_sc_max_mva (float, NaN)        : maximal short circuit apparent power to calculate internal impedance of ext_grid for short circuit calculations
s_sc_min_mva (float, NaN)        : minimal short circuit apparent power to calculate internal impedance of ext_grid for short circuit calculations
rx_max (float, NaN)              : maximal R/X-ratio to calculate internal impedance of ext_grid for short circuit calculations
rx_min (float, NaN)              : minimal R/X-ratio to calculate internal impedance of ext_grid for short circuit calculations
max_p_mw (float, NaN)            : Maximum active power injection. Only respected for OPF
min_p_mw (float, NaN)            : Minimum active power injection. Only respected for OPF
max_q_mvar (float, NaN)          : Maximum reactive power injection. Only respected for OPF
min_q_mvar (float, NaN)          : Minimum reactive power injection. Only respected for OPF
r0x0_max (float, NaN)            : maximal R/X-ratio to calculate Zero sequence internal impedance of ext_grid
x0x_max (float, NaN)             : maximal X0/X-ratio to calculate Zero sequence internal impedance of ext_grid
slack_weight (float, default 1.0): Contribution factor for distributed slack power flow calculation (active power balancing) ** only considered in loadflow if calculate_voltage_angles = True
controllable (bool, NaN)         : True: p_mw, q_mvar and vm_pu limits are enforced for the ext_grid in OPF. The voltage limits set in the ext_grid bus are enforced.
                                   False: p_mw and vm_pu setpoints are enforced and limits are ignored. The vm_pu setpoint is enforced and limits of the bus table are ignored. defaults to False if “controllable” column exists in DataFrame
"""
pp.create_ext_grid(
    net,
    bus=1,
    vm_pu=1,
    va_degree=0,
    max_p_mw=10,
    min_p_mw=-10,
    controllable=True
)

"""
greate_gen
Adds a generator to the network.
Generators are always modelled as voltage controlled PV nodes, which is why the input parameter is active power and a voltage set point. If you want to model a generator as PQ load with fixed reactive power and variable voltage, please use a static generator instead.

net                              : The net within this generator should be created <- REQUIRED
bus (int)                        : The bus id to which the generator is connected <- REQUIRED
p_mw (float, default 0)          : The active power of the generator (positive for generation!) <- REQUIRED
vm_pu (float, default 0)         : The voltage set point of the generator.
sn_mva (float, NaN)              : Nominal power of the generator
name (string, None)              : The name for this generator
index (int, None)                : Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.
scaling (float, 1.0)             : scaling factor which for the active power of the generator
type (string, None)              : type variable to classify generators
controllable (bool, NaN)         : True: p_mw, q_mvar and vm_pu limits are enforced for this generator in OPF
                                   False: p_mw and vm_pu setpoints are enforced and limits are ignored. defaults to True if “controllable” column exists in DataFrame
slack_weight (float, default 0.0): Contribution factor for distributed slack power flow calculation (active power balancing)

powerflow
vn_kv (float, NaN)             : Rated voltage of the generator for short-circuit calculation
xdss_pu (float, NaN)           : Subtransient generator reactance for short-circuit calculation
rdss_ohm (float, NaN)          : Subtransient generator resistance for short-circuit calculation
cos_phi (float, NaN)           : Rated cosine phi of the generator for short-circuit calculation
pg_percent (float, NaN)        : Rated pg (voltage control range) of the generator for short-circuit calculation
power_station_trafo (int, None): Index of the power station transformer for short-circuit calculation
in_service (bool, True)        : True for in_service or False for out of service
max_p_mw (float, default NaN)  : Maximum active power injection - necessary for OPF
min_p_mw (float, default NaN)  : Minimum active power injection - necessary for OPF
max_q_mvar (float, default NaN): Maximum reactive power injection - necessary for OPF
min_q_mvar (float, default NaN): Minimum reactive power injection - necessary for OPF
min_vm_pu (float, default NaN) : Minimum voltage magnitude. If not set the bus voltage limit is taken.
                                 necessary for OPF.
max_vm_pu (float, default NaN) : Maximum voltage magnitude. If not set the bus voltage limit is taken.
                                 necessary for OPF
"""
pp.create_gen(net, 
                bus=2, 
                p_mw=0.8830, 
                vm_pu=1,
                controllable=True,
                #max_p_mw=0.8830, 
                #min_p_mw=0.8830, 
                max_q_mvar=1,
                min_q_mvar=-1,)
                #min_vm_pu=1,
                #max_vm_pu=1)
pp.create_gen(net, 
                bus=3, 
                p_mw=0.2076, 
                vm_pu=1,
                controllable=True,
                #max_p_mw=0.2076, 
                #min_p_mw=0.2076, 
                max_q_mvar=1,
                min_q_mvar=-1,)
                #min_vm_pu=1,
                #max_vm_pu=1)


"""
create_shunt
Creates a shunt element
net (pandapowerNet) - The pandapower network in which the element is created <- REQUIRED

net (pandapowerNet)       : The pandapower network in which the element is created <- REQUIRED
bus                       : bus number of bus to whom the shunt is connected to <- REQUIRED
p_mw                      : shunt active power in MW at v= 1.0 p.u. <- REQUIRED
q_mvar                    : shunt susceptance in MVAr at v= 1.0 p.u. <- REQUIRED
vn_kv (float, None)       : rated voltage of the shunt. Defaults to rated voltage of connected bus
step (int, 1)             : step of shunt with which power values are multiplied
max_step (boolean, True)  : True for in_service or False for out of service
name (str, None)          : element name
in_service (boolean, True): True for in_service or False for out of service
index (int, None)         : Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.
"""
#pp.create_shunt(net, bus=4, p_mw=0, q_mvar=1.0, index=1)
#pp.create_shunt(net, bus=5, p_mw=0, q_mvar=0.8, index=2)

"""
Create load
Adds one load in table net[“load”].
All loads are modelled in the consumer system, meaning load is positive and generation is negative active power. Please pay attention to the correct signing of the reactive power as well.

net                                : The net within this load should be created <- REQUIRED
bus (int)                          : The bus id to which the load is connected <- REQUIRED
p_mw (float, default 0)            : The active power of the load 
                                     positive value -> load 
                                     negative value -> generation
q_mvar (float, default 0)          : The reactive power of the load
const_z_percent (float, default 0) : percentage of p_mw and q_mvar that will be associated to constant impedance load at rated voltage
const_i_percent (float, default 0) : percentage of p_mw and q_mvar that will be associated to constant current load at rated voltage
sn_mva (float, default None)       : Nominal power of the load
name (string, default None)        : The name for this load
scaling (float, default 1.)        : An OPTIONAL scaling factor. Multiplies with p_mw and q_mvar.
type (string, ‘wye’)               : type variable to classify the load              : wye/delta
index (int, None)                  : Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.
in_service (boolean)               : True for in_service or False for out of service
max_p_mw (float, default NaN)      : Maximum active power load - necessary for controllable loads in for OPF
min_p_mw (float, default NaN)      : Minimum active power load - necessary for controllable loads in for OPF
max_q_mvar (float, default NaN)    : Maximum reactive power load - necessary for controllable loads in for OPF
min_q_mvar (float, default NaN)    : Minimum reactive power load - necessary for controllable loads in OPF
controllable (boolean, default NaN): States, whether a load is controllable or not. Only respected for OPF; defaults to False if “controllable” column exists in DataFrame
"""
pp.create_load(
    net,
    bus=3,
    p_mw=0.2,
    q_mvar = 0.1,
    #max_p_mw=0.2,
    #min_p_mw=0.2,
    #max_q_mvar=0.1,
    #min_q_mvar=0.1,
    controllable=False
)

pp.create_load(
    net,
    bus=4,
    p_mw=1.7137,
    q_mvar = 0.5983,
    #max_p_mw=1.7137,
    #min_p_mw=1.7137,
    #max_q_mvar=0.5983,
    #min_q_mvar=0.5983,
    controllable=False
)

pp.create_load(
    net,
    bus=5,
    p_mw=1.7355,
    q_mvar = 0.5496,
    #max_p_mw=1.7355,
    #min_p_mw=1.7355,
    #max_q_mvar=0.5496,
    #min_q_mvar=0.5496,
    controllable=False
)

"""
For lines, there are 2 options:
create_line and create_line_from_parameters
create_line uses standard lines, so i will use from_parameters here

pandapower.create_line_from_parameters(net, from_bus, to_bus, length_km, r_ohm_per_km, x_ohm_per_km, c_nf_per_km, max_i_ka, name=None, index=None, type=None, geodata=None, in_service=True, df=1.0, parallel=1, g_us_per_km=0.0, max_loading_percent=nan, alpha=nan, temperature_degree_celsius=nan, r0_ohm_per_km=nan, x0_ohm_per_km=nan, c0_nf_per_km=nan, g0_us_per_km=0, endtemp_degree=nan, **kwargs)
Creates a line element in net[“line”] from line parameters.

net                                         : The net within this line should be created <- REQUIRED
from_bus (int)                              : ID of the bus on one side which the line will be connected with <- REQUIRED
to_bus (int)                                : ID of the bus on the other side which the line will be connected with <- REQUIRED
length_km (float)                           : The line length in km <- REQUIRED
r_ohm_per_km (float)                        : line resistance in ohm per km <- REQUIRED
x_ohm_per_km (float)                        : line reactance in ohm per km <- REQUIRED
c_nf_per_km (float)                         : line capacitance (line-to-earth) in nano Farad per km <- REQUIRED
r0_ohm_per_km (float)                       : zero sequence line resistance in ohm per km <- REQUIRED
x0_ohm_per_km (float)                       : zero sequence line reactance in ohm per km <- REQUIRED
c0_nf_per_km (float)                        : zero sequence line capacitance in nano Farad per km <- REQUIRED
max_i_ka (float)                            : maximum thermal current in kilo Ampere <- REQUIRED
name (string, None)                         : A custom name for this line
index (int, None)                           : Force a specified ID if it is available. If None, the index one higher than the highest already existing index is selected.
in_service (boolean, True)                  : True for in_service or False for out of service
type (str, None)                            : type of line (“ol” for overhead line or “cs” for cable system)
df (float, 1)                               : derating factor                                 : maximal current of line in relation to nominal current of line (from 0 to 1)
g_us_per_km (float, 0)                      : dielectric conductance in micro Siemens per km
g0_us_per_km (float, 0)                     : zero sequence dielectric conductance in micro Siemens per km
parallel (integer, 1)                       : number of parallel line systems
geodata (array, default None, shape= (,2))  : The geodata of the line. The first row should be the coordinates of bus a and the last should be the coordinates of bus b. The points in the middle represent the bending points of the line
max_loading_percent (float)                 : maximum current loading (only needed for OPF)
alpha (float)                               : temperature coefficient of resistance           : R(T) = R(T_0) * (1 + alpha * (T - T_0)))
temperature_degree_celsius (float)          : line temperature for which line resistance is adjusted
tdpf (bool)                                 : whether the line is considered in the TDPF calculation
wind_speed_m_per_s (float)                  : wind speed at the line in m/s (TDPF)
wind_angle_degree (float)                   : angle of attack between the wind direction and the line (TDPF)
conductor_outer_diameter_m (float)          : outer diameter of the line conductor in m (TDPF)
air_temperature_degree_celsius (float)      : ambient temperature in °C (TDPF)
reference_temperature_degree_celsius (float): reference temperature in °C for which r_ohm_per_km for the line is specified (TDPF)
solar_radiation_w_per_sq_m (float)          : solar radiation on horizontal plane in W/m² (TDPF)
solar_absorptivity (float)                  : Albedo factor for absorptivity of the lines (TDPF)
emissivity (float)                          : Albedo factor for emissivity of the lines (TDPF)
r_theta_kelvin_per_mw (float)               : thermal resistance of the line (TDPF, only for simplified method)
mc_joule_per_m_k (float)                    : specific mass of the conductor multiplied by the specific thermal capacity of the material (TDPF, only for thermal inertia consideration with tdpf_delay_s parameter)
"""
pp.create_line_from_parameters(
    net=net,
    from_bus=1,
    to_bus=2,
    length_km=1,
    r_ohm_per_km=0.0099,
    x_ohm_per_km=0.099,
    c_nf_per_km=0,
    r0_ohm_per_km=0.0099,
    x0_ohm_per_km=0.099,
    c0_nf_per_km=0,
    max_i_ka=1, # i am not sure about this since i dont think its given?
    index=1
)

pp.create_line_from_parameters(
    net=net,
    from_bus=1,
    to_bus=4,
    length_km=1,
    r_ohm_per_km=0.0099,
    x_ohm_per_km=0.099,
    c_nf_per_km=0,
    r0_ohm_per_km=0.0099,
    x0_ohm_per_km=0.099,
    c0_nf_per_km=0,
    max_i_ka=1, # i am not sure about this since i dont think its given?
    index=2
)

pp.create_line_from_parameters(
    net=net,
    from_bus=2,
    to_bus=3,
    length_km=1,
    r_ohm_per_km=0.0099,
    x_ohm_per_km=0.099,
    c_nf_per_km=0,
    r0_ohm_per_km=0.0099,
    x0_ohm_per_km=0.099,
    c0_nf_per_km=0,
    max_i_ka=1, # i am not sure about this since i dont think its given?
    index=3
)

pp.create_line_from_parameters(
    net=net,
    from_bus=2,
    to_bus=4,
    length_km=1,
    r_ohm_per_km=0.0099,
    x_ohm_per_km=0.099,
    c_nf_per_km=0,
    r0_ohm_per_km=0.0099,
    x0_ohm_per_km=0.099,
    c0_nf_per_km=0,
    max_i_ka=1, # i am not sure about this since i dont think its given?
    index=4
)

pp.create_line_from_parameters(
    net=net,
    from_bus=3,
    to_bus=5,
    length_km=1,
    r_ohm_per_km=0.0099,
    x_ohm_per_km=0.099,
    c_nf_per_km=0,
    r0_ohm_per_km=0.0099,
    x0_ohm_per_km=0.099,
    c0_nf_per_km=0,
    max_i_ka=1, # i am not sure about this since i dont think its given?
    index=5
)

pp.create_line_from_parameters(
    net=net,
    from_bus=4,
    to_bus=5,
    length_km=1,
    r_ohm_per_km=0.0099,
    x_ohm_per_km=0.099,
    c_nf_per_km=0,
    r0_ohm_per_km=0.0099,
    x0_ohm_per_km=0.099,
    c0_nf_per_km=0,
    max_i_ka=1, # i am not sure about this since i dont think its given?
    index=6
)


"""
Runs a power flow

INPUT:
net - The pandapower format network
OPTIONAL:
algorithm (str, “nr”) - algorithm that is used to solve the power flow problem.

The following algorithms are available:

“nr” Newton-Raphson (pypower implementation with numba accelerations)
“iwamoto_nr” Newton-Raphson with Iwamoto multiplier (maybe slower than NR but more robust)
“bfsw” backward/forward sweep (specially suited for radial and weakly-meshed networks)
“gs” gauss-seidel (pypower implementation)
“fdbx” fast-decoupled (pypower implementation)
“fdxb” fast-decoupled (pypower implementation)
calculate_voltage_angles (str or bool, True) - consider voltage angles in loadflow calculation

If True, voltage angles of ext_grids and transformer shifts are considered in the loadflow calculation. Considering the voltage angles is only necessary in meshed networks that are usually found in higher voltage levels. calculate_voltage_angles in “auto” mode defaults to:

True, if the network voltage level is above 70 kV
False otherwise
The network voltage level is defined as the maximum rated voltage of any bus in the network that is connected to a line.

init (str, “auto”) - initialization method of the loadflow pandapower supports four methods for initializing the loadflow:

“auto” - init defaults to “dc” if calculate_voltage_angles is True or “flat” otherwise
“flat”- flat start with voltage of 1.0pu and angle of 0° at all PQ-buses and 0° for PV buses as initial solution, the slack bus is initialized with the values provided in net[“ext_grid”]
“dc” - initial DC loadflow before the AC loadflow. The results of the DC loadflow are used as initial solution for the AC loadflow. Note that the DC loadflow only calculates voltage angles at PQ and PV buses, voltage magnitudes are still flat started.
“results” - voltage vector of last loadflow from net.res_bus is used as initial solution. This can be useful to accelerate convergence in iterative loadflows like time series calculations.
Considering the voltage angles might lead to non-convergence of the power flow in flat start. That is why in “auto” mode, init defaults to “dc” if calculate_voltage_angles is True or “flat” otherwise

max_iteration (int, “auto”) - maximum number of iterations carried out in the power flow algorithm.

In “auto” mode, the default value depends on the power flow solver:

10 for “nr”
100 for “bfsw”
1000 for “gs”
30 for “fdbx”
30 for “fdxb”
30 for “nr” with “tdpf”
tolerance_mva (float, 1e-8) - loadflow termination condition referring to P / Q mismatch of node power in MVA

trafo_model (str, “t”) - transformer equivalent circuit model pandapower provides two equivalent circuit models for the transformer:

“t” - transformer is modeled as equivalent with the T-model.
“pi” - transformer is modeled as equivalent PI-model. This is not recommended, since it is less exact than the T-model. It is only recommended for valdiation with other software that uses the pi-model.
trafo_loading (str, “current”) - mode of calculation for transformer loading

Transformer loading can be calculated relative to the rated current or the rated power. In both cases the overall transformer loading is defined as the maximum loading on the two sides of the transformer.

“current”- transformer loading is given as ratio of current flow and rated current of the transformer. This is the recommended setting, since thermal as well as magnetic effects in the transformer depend on the current.
“power” - transformer loading is given as ratio of apparent power flow to the rated apparent power of the transformer.
enforce_q_lims (bool, False) - respect generator reactive power limits

If True, the reactive power limits in net.gen.max_q_mvar/min_q_mvar are respected in the loadflow. This is done by running a second loadflow if reactive power limits are violated at any generator, so that the runtime for the loadflow will increase if reactive power has to be curtailed.

Note: enforce_q_lims only works if algorithm=”nr”!

check_connectivity (bool, True) - Perform an extra connectivity test after the conversion from pandapower to PYPOWER

If True, an extra connectivity test based on SciPy Compressed Sparse Graph Routines is perfomed. If check finds unsupplied buses, they are set out of service in the ppc

voltage_depend_loads (bool, True) - consideration of voltage-dependent loads. If False, net.load.const_z_percent and net.load.const_i_percent are not considered, i.e. net.load.p_mw and net.load.q_mvar are considered as constant-power loads.

consider_line_temperature (bool, False) - adjustment of line impedance based on provided
line temperature. If True, net.line must contain a column “temperature_degree_celsius”. The temperature dependency coefficient alpha must be provided in the net.line.alpha column, otherwise the default value of 0.004 is used
distributed_slack (bool, False) - Distribute slack power
according to contribution factor weights for external grids and generators.
tdpf (bool, False) - Temperature Dependent Power Flow (TDPF). If True, line temperature is calculated based on the TDPF parameters in net.line table.

tdpf_delay_s (float, None) - TDPF parameter, specifies the time delay in s to consider thermal inertia of conductors.

KWARGS:

lightsim2grid ((bool,str), “auto”) - whether to use the package lightsim2grid for power flow backend

numba (bool, True) - Activation of numba JIT compiler in the newton solver

If set to True, the numba JIT compiler is used to generate matrices for the powerflow, which leads to significant speed improvements.

switch_rx_ratio (float, 2) - rx_ratio of bus-bus-switches. If the impedance of switches defined in net.switch.z_ohm is zero, buses connected by a closed bus-bus switch are fused to model an ideal bus. Closed bus-bus switches, whose impedance z_ohm is not zero, are modelled as branches with resistance and reactance according to net.switch.z_ohm and switch_rx_ratio.

delta_q - Reactive power tolerance for option “enforce_q_lims” in kvar - helps convergence in some cases.

trafo3w_losses - defines where open loop losses of three-winding transformers are considered. Valid options are “hv”, “mv”, “lv” for HV/MV/LV side or “star” for the star point.

v_debug (bool, False) - if True, voltage values in each newton-raphson iteration are logged in the ppc

init_vm_pu (string/float/array/Series, None) - Allows to define initialization specifically for voltage magnitudes. Only works with init == “auto”!

“auto”: all buses are initialized with the mean value of all voltage controlled elements in the grid
“flat” for flat start from 1.0
“results”: voltage magnitude vector is taken from result table
a float with which all voltage magnitudes are initialized
an iterable with a voltage magnitude value for each bus (length and order has to match with the buses in net.bus)
a pandas Series with a voltage magnitude value for each bus (indexes have to match the indexes in net.bus)
init_va_degree (string/float/array/Series, None) - Allows to define initialization specifically for voltage angles. Only works with init == “auto”!

“auto”: voltage angles are initialized from DC power flow if angles are calculated or as 0 otherwise
“dc”: voltage angles are initialized from DC power flow
“flat” for flat start from 0
“results”: voltage angle vector is taken from result table
a float with which all voltage angles are initialized
an iterable with a voltage angle value for each bus (length and order has to match with the buses in net.bus)
a pandas Series with a voltage angle value for each bus (indexes have to match the indexes in net.bus)
recycle (dict, none) - Reuse of internal powerflow variables for time series calculation

Contains a dict with the following parameters: bus_pq: If True PQ values of buses are updated trafo: If True trafo relevant variables, e.g., the Ybus matrix, is recalculated gen: If True Sbus and the gen table in the ppc are recalculated

neglect_open_switch_branches (bool, False) - If True no auxiliary buses are created for branches when switches are opened at the branch. Instead branches are set out of service

tdpf_update_r_theta (bool, True) - TDPF parameter, whether to update R_Theta in Newton-Raphson or to assume a constant R_Theta (either from net.line.r_theta, if set, or from a calculation based on the thermal model of Ngoko et.al.)

update_vk_values (bool, True) - If True vk and vkr values of trafo3w are recalculated based on characteristics, otherwise the values from the table are used. Can improve performance for large models.
"""

try:
    pp.runpp(
        net=net,
        algorithm='nr',
        calculate_voltage_angles=True,
        init='dc',
        max_iteration=100,
        tolerance_mva=1e-8,
        enforce_q_lims=True
    )
except pp.LoadflowNotConverged:
    print("Power flow did not converge.")



# Retrieve voltage angles (in degrees) for all buses
voltage_angles = net.res_bus.loc[[1, 2, 3, 4, 5], ["va_degree"]]
print("Voltage Angles (θ) for Buses 1-5 (in degrees):")
print(voltage_angles)


# Retrieve voltage magnitudes (in per unit) for buses 4 and 5
voltage_magnitudes = net.res_bus.loc[[4, 5], ["vm_pu"]]
print("\nVoltage Magnitudes (V) for Buses 4 and 5 (in p.u.):")
print(voltage_magnitudes)


# Retrieve slack bus power results
slack_bus_power = net.res_ext_grid.loc[0, ["p_mw", "q_mvar"]]
print("\nSlack Bus Power (S1):")
print(slack_bus_power)

print("\nDetailed Slack Bus Results:")
print(net.res_ext_grid)

# Calculate total active and reactive line losses
total_p_loss = net.res_line.pl_mw.sum()
total_q_loss = net.res_line.ql_mvar.sum()
print(f"\nTotal Line Active Losses (P_loss): {total_p_loss:.4f} MW")
print(f"Total Line Reactive Losses (Q_loss): {total_q_loss:.4f} MVAr")





"""
print("Slack Bus Power (S1):")
print(net.res_ext_grid)

print("\nTotal Line Losses:")
print(net.res_line)

v4 = net.res_bus.loc[3, 'vm_pu']
v5 = net.res_bus.loc[4, 'vm_pu']
theta1 = net.res_bus.loc[0, 'va_degree']
theta2 = net.res_bus.loc[1, 'va_degree']
theta3 = net.res_bus.loc[2, 'va_degree']
theta4 = net.res_bus.loc[3, 'va_degree']
theta5 = net.res_bus.loc[4, 'va_degree']


print(f"\nVoltage at Bus 4 (V4): {v4} p.u.")
print(f"Voltage at Bus 5 (V5): {v5} p.u.")
print(f"Theta at Bus 1 (theta1): {theta1} degrees")
print(f"Theta at Bus 2 (theta2): {theta2} degrees")
print(f"Theta at Bus 3 (theta3): {theta3} degrees")
print(f"Theta at Bus 4 (theta4): {theta4} degrees")
print(f"Theta at Bus 5 (theta5): {theta5} degrees")




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

"""