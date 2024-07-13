# Asymptotic Lenia


### * [Kawaguchi _et al._](https://direct.mit.edu/isal/proceedings/isal2021/33/91/102916) developed an alternative variant of Lenia, replacing the growth function with a target function.
### * Called "Asymptotic Lenia" (ALenia here). The update is proportional to the difference between grid state and target function output.
### * Note the absence of a clipping procedure in the ALenia update.

{:style="text-align:center;"}
![ALenia update equation](https://raw.githubusercontent.com/riveSunder/fractal_persistence/master/docs/assets/alenia_update.png)

```
given
# a: cell states
# k: neighborhood kernel
# dt: step size
# calc_target: growth function
# conv: convolution function

a = a + dt * (a - calc_target(conv(k,a)))``
```

{:style="text-align:center;"}
[Previous slide](https://rivesunder.github.io/fractal_persistence/al24_slide_003) -- [Supporting resources](https://rivesunder.github.io/fractal_persistence) -- [Next slide](https://rivesunder.github.io/fractal_persistence/al24_slide_005)
