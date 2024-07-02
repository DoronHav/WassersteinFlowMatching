from jax import random   # type: ignore

def uniform(size, minval, maxval, key = random.key(0)):

    subkey, key = random.split(key)
    noise_samples = random.uniform(subkey, size,
                                    minval = minval, maxval = maxval)
    return(noise_samples)

def normal(size, minval, maxval, key = random.key(0)):

    subkey, key = random.split(key)
    noise_samples = random.truncated_normal(subkey, shape = size, upper = 3, lower = -3)
    noise_samples = minval + (maxval - minval) * (noise_samples + 3) / 6
    return(noise_samples)