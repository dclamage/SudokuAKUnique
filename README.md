# CUDA Anti-Knight 8 Given Unique Solution Search

## About

This project uses a "gauntlet" approach and an essential solution list to find all ways to give 8 givens to solve uniquely as an anti-knight Sudoku.

Anti-knight Sudoku follows Classic Sudoku rules, but also digits cannot be the same when they are a knight's move (as in chess) apart. From any given cell, this is up to 8 relative locations: (-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 2).

## Essential Solution List

There are `8490104 * 9! = 3080888939520` anti-knight solutions to a 9x9 empty board. The `9!` symmetry is due to "renumbering" where you can always, say, swap all 1s and 2s and have another valid solution.

The "Essential Solution List" is the 8490104 solutions to a grid with `123456789` filled as the top row.

You can generate this list yourself using any brute-force solver, or download the list used by this project from [Google Drive](https://drive.google.com/file/d/1XxxN-t2a5445mPJioeT1NCv1XrDeRj_V/view?usp=sharing). It is too large to provide within this repository (54MB compressed, 663MB uncompressed).

The format of the text file is one solution per line which contains 81 characters which represent the value of each cell from left to right and top to bottom. For example, the first 10 lines of the file are:

```
123456789456789123789123456231564897564897231897231564312645978645978312978312645
123456789456789123789123456231564897564897231897231564312645978648972315975318642
123456789456789123789123456231564897564897231897231564315648972648972315972315648
123456789456789123789123456231564897564897231897231564372615948615948372948372615
123456789456789123789123456231564897564897231897231564372615948618942375945378612
123456789456789123789123456231564897564897231897231564372615948645978312918342675
123456789456789123789123456231564897564897231897231564372615948648972315915348672
123456789456789123789123456231564897564897231897231564375618942618942375942375618
123456789456789123789123456231564897564897231897231564375618942648972315912345678
123456789456789123789123456231564897564897231897231564912345678345678912678912345
```

## Results

There are exactly `4 * 8 * 9!` ways to have 8 givens solve uniquely in anti-knight. This program produces just the 4 essential ones, and they can be rotated/reflected (`8`) as well as renumbered (`9!`) to produce all of them.

Here is the list of 4 grids:

```
................1.....................2.......3.4.......5.6.......7.........8....
..........................................1.....2.3.4...5.6.7.......8............
..........................................1.....2.3.....4.5.6.7.....8............
..........................................1.....2.3.....4.5.6.7.......8..........
```

## Building

There are two projects in this repo, in the CPU and GPU folders.

### GPU

Building the GPU project is optional, as the resulting `kernel.ptx` file is committed in the repository at `CPU/cuda/kernel.ptx`.

Building the GPU requires the CUDA SDK installed.

The following `nvcc` command will build:

`nvcc -ptx -o kernel.ptx kernel.cu`

To use the ptx file, copy it to the `CPU/cuda` folder and build the CPU project.

### CPU

**Requirements**

1. The CPU project requires Rust v1.58.0 or later. Use `rustup update stable` to update to this version.
2. Download the list of essential solutions from [Google Drive](https://drive.google.com/file/d/1XxxN-t2a5445mPJioeT1NCv1XrDeRj_V/view?usp=sharing) and extract it to the CPU folder. It should be named `ak_solutions.txt`
3. Change to the CPU directory: `cd CPU`

**Building**

```
cargo build --release
```

**Running**

```
cargo run --release
```

## License

This software is free and open source under the MIT License. See LICENSE.md for details.
