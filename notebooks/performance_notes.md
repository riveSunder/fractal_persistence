
Synchronous parallelization across threads with mpi4py

```
# git hash: 7ac189a66e4fdbb0fcb487ba93744085115275d9
> python -m fracatal.scripts.mpi_sweep -r 3 -a zoom_lentil -g 256  -p 128 -n asymdrop -w 45 -nkr 13.0 -nmu 0.12 -ns 0.00125 -xs 0.015 -ndt 0.01 -xdt 0.5 -t 10.0 -v 0

total elapsed: 375.064 s, last sweep: 375.064
    $\Delta t$ from 1.00e-02 to 5.00e-01
   $\sigma$ from 1.250000e-03 to 1.50e-02 
```

Make workers asynchronous

```
# git hash: ba08d28807f9af994ee4b7b51f8c7903e1b7ab1a
> python -m fracatal.scripts.mpi_sweep -r 3 -a zoom_lentil -g 256  -p 128 -n asymdrop -w 45 -nkr 13.0 -nmu 0.12 -ns 0.00125 -xs 0.015 -ndt 0.01 -xdt 0.5 -t 10.0 -v 0

total elapsed: 250.616 s, last sweep: 250.615 
    $\Delta t$ from 1.00e-02 to 5.00e-01 
   $\sigma$ from 1.250000e-03 to 1.50e-02
```

Use jit compilation on `update_step` by default in `make_update_step`

```
# git hash: 53d59dbd25e7e654571e16fe357e8a85328e6007
> python -m fracatal.scripts.mpi_sweep -r 3 -a zoom_lentil -g 256  -p 128 -n asymdrop -w 45 -nkr 13.0 -nmu 0.12 -ns 0.00125 -xs 0.015 -ndt 0.01 -xdt 0.5 -t 10.0 -v 0

total elapsed: 260.446 s, last sweep: 260.446 
    $\Delta t$ from 1.00e-02 to 5.00e-01
   $\sigma$ from 1.250000e-03 to 1.50e-02

```

Change jit defaults (`use_jit=False`)
```
# git hash: b11abb28967b6e46b5885697fd757901282813b3
> python -m fracatal.scripts.mpi_sweep -r 3 -a zoom_lentil -g 256  -p 128 -n asymdrop -w 45 -nkr 13.0 -nmu 0.12 -ns 0.00125 -xs 0.015 -ndt 0.01 -xdt 0.5 -t 10.0 -v 0
total elapsed: 252.984 s, last sweep: 252.984
    $\Delta t$ from 1.00e-02 to 5.00e-01
   $\sigma$ from 1.250000e-03 to 1.50e-02 

# reducing the simulation grid size (-g) from 256 to 64 saves a few additional seconds, 
# but keep in mind this is only available when the max kernel radius is relatively small (<= grid_dim//2 - 1)
total elapsed: 241.758 s, last sweep: 241.743
    $\Delta t$ from 1.00e-02 to 5.00e-01
   $\sigma$ from 1.250000e-03 to 1.50e-02
                                            
```
