def Aerodynamic_efficiency(L: float, D: float) -> float:
    # Efficiency
    Eff = L / D
    return Eff

def Aerodynamic_trimming(L: float, weight_req: float) -> float:
    # Lift constraint
    lift_req = weight_req * 9.81
    Lift_const = (L - lift_req) / lift_req
    return Lift_const
