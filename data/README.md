Data saved from running simulations. JLD2 is the preferred method for saving data. Save data as a named tuple and not as a struct to avoid type issues.

For example:
```julia
using JLD2

params = (
    param1=param1,
    param2=param2,
)

jldsave(filename;
    params=params,
    times=times,
    xdata=xdata,
)
```