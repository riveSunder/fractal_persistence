# Continuous Cellular Automata 
## Lenia ([Chan 2019](https://www.complex-systems.com/abstracts/v28_i03_a01/))

<div align="center">
  <img src="https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/docs/assets/kernel_growth.png" width=512 alt="neighborhood kernel and growth function used in Lenia">
</div>

{:style="text-align:center;"}
![Lenia update equation](https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/docs/assets/lenia_update.png)

```
# given
# a: cell states
# k: neighborhood kernel
# dt: step size
# grow: growth function
# conv: convolution function
# clip: clipping function (truncates values)

a = clip(a + dt * grow(conv(a,k)), 0., 1.)
```


{:style="text-align:center;"}
[Previous slide](https://rivesunder.github.io/fractal_persistence/al24_slide_000) -- [Supporting resources](https://rivesunder.github.io/fractal_persistence) -- [Next slide](https://rivesunder.github.io/fractal_persistence/al24_slide_001b)
