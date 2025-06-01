
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


def resistivite_eau_robinet(temp_celsius,sigma):

    sigma_25=sigma
    alpha=0.02
    sigma_T = sigma_25 * (1 + alpha * (temp_celsius - 25))
    rho_T = 1 / sigma_T
    return rho_T



def resistivite_eau_mer(temp_celsius, sigma_25=5.0, alpha=0.019):

    sigma_25=5.0
    alpha=0.02
    sigma_T = sigma_25 * (1 + alpha * (temp_celsius - 25))
    rho_T = 1 / sigma_T
    return rho_T


def compute_temperature_based_Q(cell_mass_g):
    temperature_ranges = [
        (40, 120, 0.5),
        (120, 250, 1.5),
        (250, 300, 5.0),
        (180, 280, 8.0),
        (220, 280, 6.0),
        (200, 600, 10.0),
        ]
    def Q_total(T):
        Q_watts_per_gram = 0.0
        for t_min, t_max, q_max in temperature_ranges:
            width = t_max - t_min
            center = (t_min + t_max) / 2
            sigma = width / 4
            Q_watts_per_gram += q_max * np.exp(-0.5 * ((T - center) / sigma)**2)
        return Q_watts_per_gram * cell_mass_g
    return Q_total

def resistance_from_salinity_precise(salinity, water_temp, sigma):
    l = 0.01
    A = 5e-4
    if salinity == 0:
        rho = resistivite_eau_robinet(water_temp,sigma)
    else : 
        rho = resistivite_eau_mer(water_temp)
    return rho * l / A
def compute_h_air(V, L):


    k = 0.0262        # W/m·K
    nu = 15.89e-6     # m²/s
    Pr = 0.71

    Re = V * L / nu

    if Re < 5e5:
        Nu = 0.664 * Re**0.5 * Pr**(1/3)
    else:
        Nu = 0.0371 * Re**(4/5) * Pr**(1/3)

    h = Nu * k / L
    return h

def get_water_properties(T_water_C):

    T_data = np.array([0, 20, 40, 60, 80, 100])
    nu_data = np.array([1.79, 1.00, 0.658, 0.476, 0.363, 0.294]) * 1e-6  # m²/s
    alpha_data = np.array([1.33, 1.43, 1.53, 1.64, 1.76, 1.88]) * 1e-7   # m²/s
    k_data = np.array([0.561, 0.586, 0.617, 0.641, 0.656, 0.025])        # W/m·K

    nu = np.interp(T_water_C, T_data, nu_data)
    alpha = np.interp(T_water_C, T_data, alpha_data)
    k = np.interp(T_water_C, T_data, k_data)
    Pr = nu / alpha

    return nu, alpha, Pr, k

def compute_h_water_stagnant(T_pack_C, T_water_C, L, T_surf_prev):

    T_mean_K = (T_surf_prev + T_water_C) / 2 + 273.15 
    beta = 1 / T_mean_K
    g = 9.81  

    nu, alpha, Pr, k = get_water_properties(T_water_C)
    delta_T = abs(T_surf_prev - T_water_C)

    Ra = g * beta * delta_T * L**3 / (nu * alpha)
    
    Nu = (0.825 + (0.387 * Ra**(1/6)) / (1 + (0.492 / Pr)**(9/16))**(8/27))**2
    h = Nu * k / L
    
    if T_surf_prev > 100 : 
        return compute_boiling_h(T_surf_prev, T_sat=373.15),Ra
    else : 
        
        return h,Ra
def compute_overall_U(h_ext, A, layers):

    R_cond = sum(e / (lamb) for e, lamb in layers)
    R_conv = 1 / (h_ext)
    R_total = R_cond + R_conv
    U = 1 / R_total
    return U, R_cond,R_conv

def interpolate_water_properties_inline(T):


    table = {
        "T (K)": [373.15, 400, 450, 500, 550, 600],
        "rho_l (kg/m3)": [1000/1.044, 1000/1.067, 1000/1.123, 1000/1.203, 1000/1.323, 1000/1.541],
        "rho_v (kg/m3)": [1/1.679, 1/0.731, 1/0.208, 1/0.0766, 1/0.0317, 1/0.0137],
        "k_l (W/m.K)": [0.680, 0.688, 0.678, 0.642, 0.58, 0.497],
        "k_v (W/m.K)": [0.0248, 0.0272, 0.0331, 0.0423, 0.0583, 0.0929],
        "mu_l (Pa.s)": [279e-6, 217e-6, 152e-6, 118e-6, 97e-6, 81e-6],
        "mu_v (Pa.s)": [12.02e-6, 1.305e-5, 1.485e-5, 1.660e-5, 1.860e-5, 2.270e-5],
        "cp_l (J/kg.K)": [4217, 4256, 4400, 4660, 5240, 7000],
        "cp_v (J/kg.K)": [2029, 2158, 2600, 3270, 4640, 8750],
        "Pr_l": [1.76, 1.73, 0.99, 0.86, 0.87, 1.14],
        "h_lv (J/kg)": [2.257e6, 2.183e6, 2.024e6, 1.825e6, 1.564e6, 1.176e6],
        "sigma (N/m)": [0.0589, 0.0536, 0.0429, 0.0316, 0.0197, 0.0084],
    }
    
    props = {}

    T_array = table["T (K)"]
    if T < T_array[0] or T > T_array[-1]:
        for key in table.items(): 
            props["rho_l (kg/m3)"] = 1000/1.541
            props["rho_v (kg/m3)"] = 1/0.0137
            props["k_l (W/m.K)"] = 0.497
            props["k_v (W/m.K)"] = 0.0929
            props["mu_l (Pa.s)"] = 81e-6 
            props["mu_v (Pa.s)"] = 2.270e-5
            props["cp_l (J/kg.K)"] = 7000
            props["cp_v (J/kg.K)"] = 8750
            props["Pr_l"] = 1.14
            props["h_lv (J/kg)"] = 1.176e6
            props["sigma (N/m)"] = 0.0084
            
        return props
    else:
        for key, values in table.items():
            if key != "T (K)":
                props[key] = np.interp(T, T_array, values)
        return props



def compute_boiling_h(T_surf, T_sat=373.15):
    
    Csf = 0.013  
    n = 1.0
    g = 9.81  
    
    T_surf = T_surf + 273.15

    deltaT_sat = T_surf - T_sat
    
    props = interpolate_water_properties_inline(T_surf)
    rho_l = props["rho_l (kg/m3)"]
    rho_v = props["rho_v (kg/m3)"]
    mu_l = props["mu_l (Pa.s)"] 
    mu_v = props["mu_v (Pa.s)"]
    cp_l = props["cp_l (J/kg.K)"]
    cp_v = props["cp_v (J/kg.K)"]
    Pr_l = props["Pr_l"]
    h_lv = props["h_lv (J/kg)"]
    sigma = props["sigma (N/m)"] 
    

    
    q_nuc = mu_l * h_lv * ((sigma / ((rho_l - rho_v) * g)) ** -0.5) * ((cp_l * deltaT_sat / (Csf * h_lv * (Pr_l ** n))) ** 3)
    h_nucleate = q_nuc / deltaT_sat

    T_vfm = (T_surf+ T_sat)/2
    props2 = interpolate_water_properties_inline(T_vfm)
    rho_vfm = props2["rho_v (kg/m3)"]
    k_vfm = props2["k_v (W/m.K)"]
    h_lv_prime = h_lv + 0.5 * cp_v * deltaT_sat
    hc = 0.425 * (
        ((g * (rho_l - rho_v) * rho_vfm * k_vfm**3 * h_lv_prime) / (mu_v * deltaT_sat)) *
        ((g * (rho_l - rho_v) / sigma) ** 0.5)
    ) ** 0.25
    

    q_max = 0.149 * rho_v * h_lv * ((sigma * (rho_l - rho_v) * g / rho_v**2) ** 0.25)
    
    if q_nuc < q_max:

        return h_nucleate
    else:
        
        return hc      
    

print(compute_boiling_h(200, T_sat=373.15))
def simulate(water_values, T_air_values):

    print(1)
    T_init = 180 
    
    T_water_init = 25.0 
    T_surf_init = 25.0 
    T_air = T_air_values
    TOER_initial = 130
    TOER_post = 43.0
    T_max = 25
    t_immersion_start_h = 2
    t_immersion_duration_h = 24
    t_total_h = 100
    
    n_cells = 4416
    
    t_prop_init = 160
    t_prop_min = 5
    delta = 0.4 
    
    Q_nom_j = 1600 
    t_Q = 25 
    cell_mass_g = 65.0 
    Q_nom = Q_nom_j * cell_mass_g/t_Q

    V_cell = 3.6
    salinity = 0
    sigma = 1 #[0.05,0.5,1,2,5] #conductivité de l'eau 
    water_penetration = water_values
    
    n_used_cells = 0

    

    m_pack = 478
    Cp_pack = 1000 
    A = 8.97
    A_env = 2.4*6
    L_epp = 0.33 #eppaisseur de la batterie ou longueur si on la met verticalement : 2.18 m
    L_long = 2.18
    V_air = 1 #vitesse du vent : [1-10]
    h_air =  compute_h_air(V_air, L_long)
    
    layers = [
    (0.003, 185),     # boîtier Alu
    (0.003, 0.04),     # mousse isolante 1-5
    (0.0015, 50),     # tôle acier
    (0.002,0.24) #2-5 PP
]

    dt = 1.0
    t_total = t_total_h * 3600.0
    n_steps = int(t_total // dt)

    m_water = 5000
    Cp_water = 4186

    immersion_start_index = int(t_immersion_start_h * 3600.0 // dt)
    immersion_end_index = int((t_immersion_start_h + t_immersion_duration_h) * 3600.0 // dt)


    T = np.zeros(n_steps)
    T_water = np.zeros(n_steps)
    T_surf = np.zeros(n_steps)
    h_tab = np.zeros(n_steps)
    Q_profile = np.zeros(n_steps)
    R_a = np.zeros(n_steps)
    N_TR_profile = np.zeros(n_steps, dtype=int)
    t_prop_profile = np.zeros(n_steps)
    n_fresh_profile = np.zeros(n_steps)
    T[0] = T_init
    T_water[0] = T_water_init
    T_surf[0] = T_surf_init
    time_h = np.linspace(0, t_total / 3600.0, n_steps)


    N_TR = 1
    active_pulses = []


    for i in range(1, n_steps):
        t = i * dt
        T_prev = T[i - 1]
        Tw_prev = T_water[i - 1]
        T_surf_prev = T_surf[i-1]
        R_ext = resistance_from_salinity_precise(salinity, Tw_prev, sigma)
        h_water,R_a[i] = compute_h_water_stagnant(T_prev, Tw_prev, L_epp, T_surf_prev)
        t_dis = 4.8*3600/(3.6/R_ext)
        tau_short = t_dis/4.605
        if i == 1:
            for j in range(N_TR-1):
                active_pulses.append(t_Q)  
        
        
        if immersion_start_index <= i <= immersion_end_index:
            T_env = Tw_prev
            h = h_water
            
        else:
            T_env = T_air
            h = h_air
        
        if N_TR == 0 : 
            t_prop = t_prop_init
        else : 
            t_prop = max(t_prop_min, int(t_prop_init / (N_TR**delta)))
        
        n_fresh_cells = n_cells - n_used_cells
        n_fresh_profile[i] = n_fresh_cells

        if i <= immersion_end_index:
            if T_prev < TOER_initial:
                N_TR = 0
            elif T_prev > TOER_initial and (t % t_prop == 0) and N_TR < n_cells:
                N_TR += 1
                active_pulses.append(t_Q)
                n_used_cells +=1
        else:
            if T_prev < TOER_post:
                N_TR = 0
            elif T_prev > TOER_post and (t % t_prop == 0) and N_TR < n_cells and n_fresh_cells > 1:
                #N_TR += 1
                #active_pulses.append(t_Q)
                #n_used_cells +=1
                n = 0

        active_pulses = [p - 1 for p in active_pulses if p > 1]

    # Génération de chaleur due au TR
        Q_gen = Q_nom * len(active_pulses) 

        N_TR_profile[i] = N_TR
        t_prop_profile[i] = t_prop
        h_tab[i] = h


        if i > immersion_end_index  and water_penetration > 0:
            Qw = water_penetration * n_cells * (V_cell**2) / R_ext * np.exp(-(t - (immersion_end_index )) / tau_short)
            Q_gen += Qw
        
        Q_profile[i] = Q_gen
        U,R_cond,R_conv = compute_overall_U(h,A,layers)
        
        dTdt = (-U * A * (T_prev - T_env) + Q_gen) / (m_pack * Cp_pack)
        T[i] = T_prev + dTdt * dt
        
        

        if immersion_start_index <= i <= immersion_end_index:
            q_batt_to_water = U * A * (T_prev - Tw_prev)
            q_water_to_air = h_air * A_env * (Tw_prev - T_air)
            T_surf[i] = T_prev - q_batt_to_water * (R_cond/A)
            dTw = (q_batt_to_water - q_water_to_air) / (m_water * Cp_water)
            T_water[i] = Tw_prev + dTw * dt
        else:
            T_water[i] = Tw_prev
            q_batt_to_air = U * A * (T_prev - T_air)
            T_surf[i] = T_prev - q_batt_to_air * (R_cond/A)
            dTw = -h_air * A_env * (Tw_prev - T_air)/(m_water * Cp_water)
            T_water[i] = Tw_prev + dTw * dt
        
        if i>= immersion_end_index : 
            if T[i] >= T_max : 
                T_max = T[i]

    return T_prev, Tw_prev, n_fresh_cells, T_max
            


    TOER_post = 43

    last_hour_mask = time_h >= (time_h[-1] - 1)

    time_last = time_h[last_hour_mask]
    T_last = T[last_hour_mask]
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 14))
    axs[0].plot(time_h, T , color='b')
    axs[0].axhline(TOER_post, color='r', linestyle='--', label='TOER post')
    axs[0].axhline(TOER_initial, color='purple', linestyle='--', label='TOER initial')
    axs[0].axvline(t_immersion_start_h, color='gray', linestyle=':', label='Start immersion')
    axs[0].axvline(t_immersion_start_h + t_immersion_duration_h, color='black', linestyle=':', label='End immersion')
    axs[0].set_title("Battery Pack Temperature")
    axs[0].set_xlabel("Time [h]")
    axs[0].set_ylabel("Temperature [°C]")
    axs[0].legend()
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    axs[1].plot(time_h, Q_profile, color='orange')
    axs[1].set_title("Total Heat Generation ")
    axs[1].axvline(t_immersion_start_h, color='gray', linestyle=':')
    axs[1].axvline(t_immersion_start_h + t_immersion_duration_h, color='black', linestyle=':')
    axs[1].set_xlabel("Time [h]")
    axs[1].set_ylabel("Power [W]")
    #axs[1].legend()
    axs[1].grid(True)
    
    axs[2].plot(time_h, T_water, color='blue')
    axs[2].set_title("Water Temperature")
    axs[2].axvline(t_immersion_start_h, color='gray', linestyle=':')
    axs[2].axvline(t_immersion_start_h + t_immersion_duration_h, color='black', linestyle=':')
    axs[2].axhline(TOER_post, color='r', linestyle='--', label='TOER post')
    axs[2].set_xlabel("Time [h]")
    axs[2].set_ylabel("Temperature [°C]")
    #axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    #plt.savefig("CAS2-5mm-EPP.png", format="png", dpi=300, bbox_inches="tight")
    plt.show()
    
    
#plt.show()
# if __name__ == "__main__":
#     simulate()


if __name__ == "__main__":
    # Liste des valeurs pour l'analyse paramétrique
    water_values= np.linspace(0,1,4)  
    T_air_values = [10,20,30,40]
    T = np.zeros(len(water_values))
    T_w = np.zeros(len(water_values))
    T_max = np.zeros(len(water_values))
    n_used = np.zeros(len(water_values))
    colors = ['blue', 'green', 'red', 'orange']
    T_air = 40
    plt.figure()
    for j in range(len(T_air_values)) : 
        
        for i in range(len(water_values)):

            T[i], T_w[i], n_used[i], T_max[i] = simulate(water_values[i], T_air_values[j])  # ← ta fonction principale
        plt.plot(water_values, T_max,label=f'{T_air_values[j]}°C', color = colors[j])
        plt.xlabel("L", fontsize = 16)
        plt.ylabel(r"$T_{\mathrm{max}}$ °C", fontsize = 16)
        plt.title(r'$\sigma_{\mathrm{water}} = 2~\mathrm{S/m}$',fontsize = 18)
        plt.legend()
        


    plt.grid()
    plt.savefig("test.png", format="png", dpi=300, bbox_inches="tight")
    plt.show()
    
