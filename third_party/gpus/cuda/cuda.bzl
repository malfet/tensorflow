# -*- Python -*-

def if_cuda_armhf(a, b=[]):
  return select({
      "//third_party/gpus/cuda:cuda_cpu_armhf_condition": a,
      "//conditions:default": b,
  })

