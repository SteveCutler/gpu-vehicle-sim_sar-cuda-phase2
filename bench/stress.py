import argparse
import csv
import datetime
import itertools
import json
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNELS = (
    'particle_cull_atomic_kernel',
    'p_cull_prefixlocal_kernel',
    'p_cull_prefixlocal_kernel_smem',
    'prefix_kernel',
    'p_cull_sum_scatter',
)


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def numbers(text):
    return [int(x) for x in text.split(',')]


def extract_kernel(source, name):
    match = re.search(r'__global__\s+void\s+' + name + r'\s*\(', source)
    if not match:
        raise RuntimeError(f'Missing project kernel: {name}')
    opening = source.index('{', match.end())
    depth = 0
    for i in range(opening, len(source)):
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0:
                return source[match.start():i + 1]
    raise RuntimeError(f'Unclosed project kernel: {name}')


def compile_compaction(temp):
    kernels = (ROOT / 'src/CUDA_simulation.cu').read_text()
    header = temp / 'compaction_kernels.cuh'
    header.write_text('\n\n'.join(extract_kernel(kernels, name) for name in KERNELS) + '\n')
    binary = temp / 'compaction_benchmark'
    run(['nvcc', '-std=c++17', '-O3', '-I', str(ROOT / 'include'),
         '-I', str(temp), str(ROOT / 'bench/compaction_benchmark.cu'), '-o', str(binary)])
    return binary


def gpu_state():
    p = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,temperature.gpu',
                        '--format=csv,noheader,nounits'], text=True, capture_output=True)
    return p.stdout.strip().splitlines()[0] if p.returncode == 0 else ''


def compare_fluid(reference, candidate):
    with reference.open(newline='') as a, candidate.open(newline='') as b:
        ra, rb = csv.reader(a), csv.reader(b)
        if next(ra) != next(rb):
            raise RuntimeError('Fluid headers differ')
        for index, (left, right) in enumerate(itertools.zip_longest(ra, rb)):
            if left is None or right is None or left[:2] != right[:2]:
                raise RuntimeError(f'Fluid grid shape/coordinates differ at row {index}')
            for column, (x, y) in enumerate(zip(left[2:], right[2:]), 2):
                x, y = float(x), float(y)
                if abs(x - y) > 1e-5 + 1e-4 * abs(x):
                    raise RuntimeError(f'Fluid mismatch at row {index}, column {column}: {x} vs {y}')


def pressure_copy(temp, grid):
    source = temp / 'source'
    tracked = run(['git', '-C', str(ROOT), 'ls-files', '-z'], capture_output=True).stdout.split('\0')
    for name in filter(None, tracked):
        src = ROOT / name
        dst = source / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    main = source / 'src/main.cpp'
    code = main.read_text()
    for axis in ('width', 'height'):
        old = f'constexpr std::size_t {axis} = 500;'
        if code.count(old) != 1:
            raise RuntimeError(f'Unexpected {axis} declaration')
        code = code.replace(old, f'constexpr std::size_t {axis} = {grid};')
    main.write_text(code)
    cuda = source / 'src/CUDA_simulation.cu'
    code = cuda.read_text()
    old = 'target_found = vehicles.targetCheck(vs, this->target_, env_.m_height);'
    if code.count(old) != 1:
        raise RuntimeError('Unexpected target check')
    cuda.write_text(code.replace(old, 'target_found = false;'))
    return source


def config_files(temp):
    result = {}
    for mode in ('global', 'tiled', 'temporal'):
        path = temp / f'{mode}.cfg'
        path.write_text(f'pressure={mode}\ncompaction=atomic\nplanner=precomputed\nhits=bytes\nuse_graph=false\n')
        result[mode] = path
    return result


def launch_sim(binary, config, steps, output):
    p = run([str(binary), '1', str(steps), str(output), str(config)], capture_output=True)
    done = re.search(r'Completed steps:\s*(\d+)', p.stdout)
    seconds = re.search(r'Execution time:\s*([\d.eE+-]+)', p.stdout)
    gpu = re.search(r'Pressure GPU ms:\s*([\d.eE+-]+)', p.stdout)
    samples = re.search(r'Pressure samples:\s*(\d+)', p.stdout)
    if not all((done, seconds, gpu, samples)):
        raise RuntimeError(f'Missing timer output:\n{p.stdout[-1000:]}')
    return int(done[1]), float(seconds[1]), float(gpu[1]), int(samples[1])


def measure_pressure(args, writer, temp, commit):
    for grid in args.grids:
        source = pressure_copy(temp / str(grid), grid)
        build = temp / str(grid) / 'build'
        run(['cmake', '-S', str(source), '-B', str(build), '-G', 'Unix Makefiles',
             '-DCMAKE_BUILD_TYPE=Release', '-DENABLE_RENDERER=OFF',
             '-DENABLE_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=native'])
        run(['cmake', '--build', str(build), '-j', str(args.jobs)])
        configs = config_files(temp / str(grid))
        binary = build / 'vehicle_sim'
        reference = temp / str(grid) / 'verify-global.csv'
        for mode in ('global', 'tiled', 'temporal'):
            output = temp / str(grid) / f'verify-{mode}.csv'
            done, _, _, samples = launch_sim(binary, configs[mode], args.verify_steps, output)
            if done != args.verify_steps or samples != done:
                raise RuntimeError(f'Correctness workload ended early: {grid} {mode} {done}')
            if mode != 'global':
                compare_fluid(Path(str(reference) + '.fluid.csv'), Path(str(output) + '.fluid.csv'))
        output = temp / str(grid) / 'discard.csv'
        for suffix in ('', '.particles.csv', '.fluid.csv'):
            Path(str(output) + suffix).symlink_to('/dev/null')
        for mode in ('global', 'tiled', 'temporal'):
            launch_sim(binary, configs[mode], args.steps[0], output)
        for steps in args.steps:
            for trial in range(args.repeats):
                modes = ['global', 'tiled', 'temporal']
                random.Random(steps * 1000 + trial).shuffle(modes)
                for mode in modes:
                    before = gpu_state()
                    done, seconds, gpu_ms, samples = launch_sim(binary, configs[mode], steps, output)
                    after = gpu_state()
                    status = 'OK' if done == steps and samples == done else 'EARLY_STOP'
                    writer.writerow(['pressure', commit, grid, '', '', steps, done, mode, trial,
                                     seconds, gpu_ms, samples, before, after, status])
                    print(f'pressure grid={grid} steps={steps} trial={trial} mode={mode} {status} gpu_ms={gpu_ms}', flush=True)


def measure_compaction(args, writer, temp, commit):
    binary = compile_compaction(temp)
    modes = ['atomic', 'prefix_global', 'prefix_shared']
    for count in args.particles:
        for dead in args.dead_percent:
            result = run([str(binary), str(count), str(dead), str(args.repeats)], capture_output=True)
            for line in result.stdout.splitlines():
                n, fraction, mode, trial, survivors, gpu_ms, wall_ms = line.split(',')
                writer.writerow(['compaction', commit, '', int(n), int(fraction), '', '',
                                 modes[int(mode)], int(trial), float(wall_ms) / 1000,
                                 float(gpu_ms), survivors, '', '', 'OK'])
            print(f'compaction particles={count} dead={dead}% validated', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'bench/results')
    parser.add_argument('--grids', type=numbers, default=[256, 500, 1024])
    parser.add_argument('--steps', type=numbers, default=[100, 500])
    parser.add_argument('--particles', type=numbers, default=[1000, 10000, 100000])
    parser.add_argument('--dead-percent', type=numbers, default=[0, 50, 90])
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--verify-steps', type=int, default=10)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--only', choices=['all', 'pressure', 'compaction'], default='all')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    commit = run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True).stdout.strip()
    name = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest = args.out / f'stress-{name}.csv'
    (args.out / f'stress-{name}.json').write_text(json.dumps({
        'commit': commit, 'dirty': bool(run(['git', '-C', str(ROOT), 'status', '--porcelain'], capture_output=True).stdout.strip()),
        'arguments': vars(args) | {'out': str(args.out)}, 'gpu': gpu_state(),
        'pressure_benchmark_target_detection': False,
    }, indent=2))
    with tempfile.TemporaryDirectory(prefix='sar-stress-') as directory, dest.open('w', newline='') as stream:
        temp = Path(directory)
        writer = csv.writer(stream)
        writer.writerow(['test', 'commit', 'grid', 'particles', 'dead_percent', 'requested_steps',
                         'completed_steps', 'mode', 'trial', 'wall_seconds', 'gpu_ms',
                         'samples_or_survivors', 'gpu_before', 'gpu_after', 'status'])
        if args.only in ('all', 'pressure'):
            measure_pressure(args, writer, temp, commit)
            stream.flush()
        if args.only in ('all', 'compaction'):
            measure_compaction(args, writer, temp, commit)
    print(dest)


if __name__ == '__main__':
    main()
