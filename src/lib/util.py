def convert_depth_to_m(depth_map):
    return (depth_map + 1) * 2.0

def convert_scaled_mse_to_m_mse(scaled_mse):
    return 4 * scaled_mse