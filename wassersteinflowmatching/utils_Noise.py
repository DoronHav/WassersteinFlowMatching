from jax import random    

def uniform(size, minval, maxval, key = random.key(0)):

    subkey, key = random.split(key)
    noise_samples = random.uniform(subkey, size,
                                    minval = minval, maxval = maxval)
    return(noise_samples)