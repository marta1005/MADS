def calculate_reynolds_number(characteristic_length: float, airspeed: float, density: float, dynamic_viscosity: float
                              ) -> float:
    """Returns the Reynolds number"""
    reynolds_number = (density * airspeed * characteristic_length) / dynamic_viscosity
    return reynolds_number
