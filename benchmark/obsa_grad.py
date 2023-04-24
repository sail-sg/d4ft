import time

import d4ft
import jax
from absl import app, flags
from d4ft.energy import get_os_intor, prescreen

jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS
flags.DEFINE_string("geometry", "h2", "")
flags.DEFINE_integer("iters", 10, "")
flags.DEFINE_string("basis_set", "sto-3g", "which basis set to use")
flags.DEFINE_integer("spin", 0, "total spin")
flags.DEFINE_bool("pre_cal", False, "whether to pre-calculate the integrals")


def benchmark_mol(mol_name: str = "o"):
  geometry = getattr(d4ft.geometries, mol_name + "_geometry")

  mol = d4ft.Molecule(
    geometry,
    spin=FLAGS.spin,
    level=1,
    basis=FLAGS.basis_set,
    restricted_mo=True,
    algo="sgd",
    xc="lda"
  )

  params = mol._init_param(137)

  print("=======PRESCREEN===============")
  idx_count = prescreen(mol)
  print("===============================")

  kin, ext, eri, precal = get_os_intor(
    mol,
    use_horizontal=False,
    pre_cal=FLAGS.pre_cal,
    idx_count=idx_count,
  )

  # intors = dict(kin=jax.jit(kin), ext=jax.jit(ext), eri=jax.jit(eri))
  intors = dict(eri=jax.jit(eri))

  jit_iters = 5

  n_iters = FLAGS.iters

  for _, fn_jit in intors.items():
    fn_jit(params, idx_count, precal)

  # with jax.profiler.trace("./jax-trace", create_perfetto_link=True):
  # for _, fn_jit in intors.items():
  #   with jax.profiler.TraceAnnotation(f"{fn_name} warm up"):
  #     fn_jit(params)

  for fn_name, fn_jit in intors.items():
    total_t = 0
    for t in range(n_iters):
      params = mol._init_param(t)
      with jax.profiler.TraceAnnotation(f"{fn_name} step {t}"):
        start_t = time.time()
        fn_jit(params, idx_count, precal).block_until_ready()
        iter_t = time.time() - start_t
        print(t, iter_t)
        if t >= jit_iters:
          total_t += iter_t
    print(mol_name, fn_name, total_t / (n_iters - jit_iters))

  jax.profiler.save_device_memory_profile("memory.prof")


def main(_):
  benchmark_mol(FLAGS.geometry)


if __name__ == '__main__':
  app.run(main)
