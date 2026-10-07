RHO_GRID=[0.0,0.05,0.10,0.20,0.40]

def controller(rho_current,forgetting,domain_improvement,forgetting_threshold=0.10,improvement_threshold=0.01):
    index=RHO_GRID.index(rho_current)

    if forgetting>forgetting_threshold:
        if index<len(RHO_GRID)-1:
            return RHO_GRID[index+1]

    elif domain_improvement>improvement_threshold:
        if index>0:
            return RHO_GRID[index-1]

    return rho_current