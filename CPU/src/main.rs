#[macro_use]
extern crate static_assertions;

pub mod board;

use board::{Board, LogicResult};
use cust::function::{BlockSize, GridSize};
use cust::prelude::*;
use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{prelude::*, BufReader, BufWriter};
use std::time::{Duration, Instant};

static PTX: &str = include_str!("../cuda/kernel.ptx");

fn main() -> Result<(), Box<dyn Error>> {
    const TOTAL_COMBINATIONS: usize = 32164253550; // 81 choose 8
    const COMBINATIONS_PER_PASS: usize = 2 * 3 * 3 * 3 * 3 * 5 * 5;
    const_assert!(
        (TOTAL_COMBINATIONS / COMBINATIONS_PER_PASS) * COMBINATIONS_PER_PASS == TOTAL_COMBINATIONS
    );

    // Get the thread-local random number generator
    let mut rng = thread_rng();

    // Read in all the solution grids and convert them to bitmasks
    println!("Reading in all the grids...");
    let read_start = Instant::now();
    let mut grids: Vec<u16> = Vec::new();
    let file = File::open("ak_solutions.txt")?;
    let lines = BufReader::new(file).lines();
    for line in lines {
        let line = line?;
        for c in line.chars() {
            if c >= '1' && c <= '9' {
                let n = (c as u16) - '1' as u16;
                grids.push(!(1u16 << n));
            }
        }
    }
    let num_grids = (grids.len() / 81) as u32;
    println!("Read {} grids in {:?}", num_grids, read_start.elapsed());
    assert_eq!(grids.len(), num_grids as usize * 81);
    let copy_start = Instant::now();

    // Shuffle the list of grids
    grids = {
        let mut indices: Vec<u32> = (0..num_grids).collect();
        indices.shuffle(&mut rng);

        let mut grids_shuffled: Vec<u16> = Vec::new();
        grids_shuffled.reserve_exact(grids.len());
        for i in indices {
            grids_shuffled.extend_from_slice(&grids[i as usize * 81..(i + 1) as usize * 81]);
        }

        grids_shuffled
    };

    // Determine how many grids to do in each pass
    let pass1 = (0u32, 1000u32);
    let pass2 = (pass1.1, pass1.1 + 10000u32);
    let pass3 = (pass2.1, pass2.1 + 100000u32);
    let pass4 = (pass3.1, num_grids);
    assert!(num_grids > pass4.0);

    // initialize CUDA, this will pick the first available device and will
    // make a CUDA context from it.
    // We don't need the context for anything but it must be kept alive.
    let _ctx = cust::quick_init()?;

    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // they can be made from PTX code, cubins, or fatbins.
    let module = Module::from_str(PTX)?;

    // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
    // GPU calls.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // allocate the GPU memory needed to house all the grids and copy them over.
    let mut grids_gpu = grids.as_slice().as_dbuf()?;

    println!(
        "Copied {} grids to GPU in {:?}",
        grids_gpu.len() / 81,
        copy_start.elapsed()
    );

    // retrieve the add kernel from the module so we can calculate the right launch config.
    let func = module.get_function("increment_if_unique")?;

    // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
    // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
    // current CUDA device/architecture.
    let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;

    let out_file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("ak_unique.txt")?;
    let mut out_file = BufWriter::new(out_file);

    let mut indices: Vec<u8> = Vec::new();
    indices.reserve(COMBINATIONS_PER_PASS * 8);

    let mut progress_index = 0;
    let mut num_written = 0;
    let mut num_invalid: usize = 0;
    let loop_start_time = Instant::now();
    let mut last_print_time = Instant::now();
    for cur_indices in (0..81).combinations(8) {
        if !is_minlexed(cur_indices.as_slice()) {
            progress_index += 1;

            if last_print_time.elapsed().as_secs() > 10 {
                let real_progress = progress_index;
                let elapsed_duration = loop_start_time.elapsed();
                let expected_duration = if real_progress > 0 {
                    elapsed_duration * ((TOTAL_COMBINATIONS) / progress_index) as u32
                } else {
                    Duration::from_secs(1)
                };
                let percent_complete = (progress_index as f64 / TOTAL_COMBINATIONS as f64) * 100.0;

                println!("{}", indices_to_string(&cur_indices));
                println!("[{elapsed_duration:?}][{expected_duration:?}] Skipping {progress_index} / {TOTAL_COMBINATIONS} ({percent_complete}%) [Num written: {num_written}] [Num invalid: {num_invalid}]");
                last_print_time = Instant::now();
            }
            continue;
        }

        indices.extend(cur_indices.iter());

        if indices.len() < COMBINATIONS_PER_PASS * 8 {
            continue;
        }

        progress_index += COMBINATIONS_PER_PASS;

        if last_print_time.elapsed().as_secs() > 10 {
            let real_progress = progress_index - COMBINATIONS_PER_PASS;
            let elapsed_duration = loop_start_time.elapsed();
            let expected_duration = if real_progress > 0 {
                elapsed_duration
                    * ((TOTAL_COMBINATIONS) / (progress_index - COMBINATIONS_PER_PASS)) as u32
            } else {
                Duration::from_secs(1)
            };
            let percent_complete = (progress_index as f64 / TOTAL_COMBINATIONS as f64) * 100.0;

            println!("{}", indices_to_string(&indices[0..8]));
            println!("[{elapsed_duration:?}][{expected_duration:?}] Launching {progress_index} / {TOTAL_COMBINATIONS} ({percent_complete}%) [Num written: {num_written}] [Num invalid: {num_invalid}]");
            last_print_time = Instant::now();
        }

        let mut launch = |grid_start: u32,
                          grid_end: u32,
                          indices: &Vec<u8>,
                          out: &mut Vec<u32>|
         -> Result<(), Box<dyn Error>> {
            let num_grids = grid_end - grid_start;
            let num_combinations = (indices.len() / 8) as u32;

            // allocate the GPU memory for the current indices and copy them over.
            let mut indices_gpu = indices.as_slice().as_dbuf()?;

            // allocate our output buffer.
            let mut out_buf = out.as_slice().as_dbuf()?;

            let (grid_block_size, counter_block_size) = {
                let mut grid_block_size = 2;

                while num_combinations < block_size / grid_block_size {
                    grid_block_size *= 2;
                }

                if grid_block_size <= block_size {
                    (grid_block_size, block_size / grid_block_size)
                } else {
                    (block_size, 1)
                }
            };

            let grid_size = GridSize {
                x: (num_grids + grid_block_size - 1) / grid_block_size,
                y: (num_combinations + counter_block_size - 1) / counter_block_size,
                z: 1,
            };
            let block_size = BlockSize {
                x: grid_block_size,
                y: counter_block_size,
                z: 1,
            };

            unsafe {
                launch!(
                    func<<<grid_size, block_size, 0, stream>>>(
                        grids_gpu.as_device_ptr(),
                        grid_start,
                        num_grids,
                        indices_gpu.as_device_ptr(),
                        out_buf.as_device_ptr(),
                        num_combinations,
                    )
                )?;
            }

            stream.synchronize()?;

            // copy back the data from the GPU.
            out_buf.copy_to(out)?;

            Ok(())
        };

        // Launch the first pass
        let mut out = vec![0u32; COMBINATIONS_PER_PASS];
        launch(pass1.0, pass1.1, &indices, &mut out)?;

        // Filter out combinations with more than one solution
        let (filtered, mut out) = {
            let mut filtered: Vec<u8> = Vec::new();
            let mut new_out: Vec<u32> = Vec::new();
            for i in 0..out.len() {
                if out[i] == 1 || out[i] == 0 && !is_obviously_invalid(&indices[i * 8..(i + 1) * 8])
                {
                    filtered.extend_from_slice(&indices[i * 8..(i + 1) * 8]);
                    new_out.push(out[i]);
                }
            }
            (filtered, new_out)
        };
        indices.clear();

        if filtered.len() == 0 {
            continue;
        }

        // Launch the second pass
        launch(pass2.0, pass2.1, &filtered, &mut out)?;
        let (filtered, mut out) = {
            let mut new_filtered: Vec<u8> = Vec::new();
            let mut new_out: Vec<u32> = Vec::new();
            for i in 0..out.len() {
                if out[i] <= 1 {
                    new_filtered.extend_from_slice(&filtered[i * 8..(i + 1) * 8]);
                    new_out.push(out[i]);
                }
            }
            (new_filtered, new_out)
        };

        if filtered.len() == 0 {
            continue;
        }

        // Launch the third pass
        launch(pass3.0, pass3.1, &filtered, &mut out)?;
        let (filtered, mut out) = {
            let mut new_filtered: Vec<u8> = Vec::new();
            let mut new_out: Vec<u32> = Vec::new();
            for i in 0..out.len() {
                if out[i] <= 1 {
                    new_filtered.extend_from_slice(&filtered[i * 8..(i + 1) * 8]);
                    new_out.push(out[i]);
                }
            }
            (new_filtered, new_out)
        };

        if filtered.len() == 0 {
            continue;
        }

        // Launch the fourth pass
        launch(pass4.0, pass4.1, &filtered, &mut out)?;

        let mut any_written = false;
        for i in 0..out.len() {
            if out[i] == 1 {
                let board = indices_to_string(&filtered[i * 8..(i + 1) * 8]);
                let line = format!("{board}\n");

                out_file.write_all(line.as_bytes())?;
                num_written += 1;
                any_written = true;
            } else if out[i] == 0 {
                num_invalid += 1;
            }
        }
        if any_written {
            out_file.flush()?;
        }
    }

    let elapsed_duration = loop_start_time.elapsed();
    println!("[{elapsed_duration:?}] Complete! [Num written: {num_written}] [Num invalid: {num_invalid}]");

    Ok(())
}

fn indices_to_string(indices: &[u8]) -> String {
    let mut board = ['.'; 81];
    for i in 0..8 {
        board[indices[i] as usize] = (i as u8 + '1' as u8) as char;
    }
    board.iter().collect()
}

fn is_obviously_invalid(indices: &[u8]) -> bool {
    let board = Board::from_indices(indices);
    if board.is_none() {
        return true;
    }

    let mut board = board.unwrap();
    if board.set_singles() == LogicResult::Invalid {
        return true;
    }

    false
}

fn is_minlexed(indices: &[u8]) -> bool {
    for rot in 1..8 {
        let rotated_indices: Vec<u8> = match rot {
            1 => indices
                .iter()
                .map(|&x| remap_index(x, |i, j| (i, 8 - j)))
                .sorted()
                .collect(),
            2 => indices
                .iter()
                .map(|&x| remap_index(x, |i, j| (8 - i, j)))
                .sorted()
                .collect(),
            3 => indices
                .iter()
                .map(|&x| remap_index(x, |i, j| (8 - i, 8 - j)))
                .sorted()
                .collect(),
            4 => indices
                .iter()
                .map(|&x| remap_index(x, |i, j| (j, i)))
                .sorted()
                .collect(),
            5 => indices
                .iter()
                .map(|&x| remap_index(x, |i, j| (j, 8 - i)))
                .sorted()
                .collect(),
            6 => indices
                .iter()
                .map(|&x| remap_index(x, |i, j| (8 - j, i)))
                .sorted()
                .collect(),
            7 => indices
                .iter()
                .map(|&x| remap_index(x, |i, j| (8 - j, 8 - i)))
                .sorted()
                .collect(),
            _ => unreachable!(),
        };

        if is_less(indices, &rotated_indices) {
            return false;
        }
    }
    true
}

fn remap_index(index: u8, rf: fn(u8, u8) -> (u8, u8)) -> u8 {
    let (x, y) = rf(index / 9, index % 9);
    x * 9 + y
}

fn is_less(a: &[u8], b: &[u8]) -> bool {
    for i in 0..8 {
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    #[test]
    fn test_is_less() {
        assert!(super::is_less(
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[1, 2, 3, 4, 5, 6, 7, 9]
        ));
        assert!(super::is_less(
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[2, 3, 4, 5, 6, 7, 8, 9]
        ));
        assert!(super::is_less(
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[1, 2, 3, 4, 6, 7, 8, 9]
        ));
    }

    fn minlex(indices: &[u8]) -> Vec<u8> {
        let mut best = indices.to_vec();
        for rot in 1..8 {
            let rotated_indices = match rot {
                1 => indices
                    .iter()
                    .map(|&x| super::remap_index(x, |i, j| (i, 8 - j)))
                    .sorted()
                    .collect(),
                2 => indices
                    .iter()
                    .map(|&x| super::remap_index(x, |i, j| (8 - i, j)))
                    .sorted()
                    .collect(),
                3 => indices
                    .iter()
                    .map(|&x| super::remap_index(x, |i, j| (8 - i, 8 - j)))
                    .sorted()
                    .collect(),
                4 => indices
                    .iter()
                    .map(|&x| super::remap_index(x, |i, j| (j, i)))
                    .sorted()
                    .collect(),
                5 => indices
                    .iter()
                    .map(|&x| super::remap_index(x, |i, j| (j, 8 - i)))
                    .sorted()
                    .collect(),
                6 => indices
                    .iter()
                    .map(|&x| super::remap_index(x, |i, j| (8 - j, i)))
                    .sorted()
                    .collect(),
                7 => indices
                    .iter()
                    .map(|&x| super::remap_index(x, |i, j| (8 - j, 8 - i)))
                    .sorted()
                    .collect(),
                _ => indices.to_vec(),
            };
            if super::is_less(&best, &rotated_indices) {
                best = rotated_indices;
            }
        }

        best
    }

    #[test]
    fn test_is_minlexed() {
        assert!(!super::is_minlexed(&[1, 2, 3, 4, 5, 6, 7, 8]));
        assert!(!super::is_minlexed(&[1, 2, 3, 4, 5, 6, 7, 9]));
        assert!(!super::is_minlexed(&[1, 10, 19, 28, 37, 46, 55, 64]));
        assert!(super::is_minlexed(&[73, 74, 75, 76, 77, 78, 79, 80]));
        assert!(super::is_minlexed(&minlex(&[1, 2, 3, 4, 5, 6, 7, 8])));
        assert!(super::is_minlexed(&minlex(&[1, 2, 3, 4, 5, 6, 7, 9])));
        assert!(super::is_minlexed(&minlex(&[
            1, 10, 19, 28, 37, 46, 55, 64
        ])));
        assert!(super::is_minlexed(&minlex(&[
            73, 74, 75, 76, 77, 78, 79, 80
        ])));
        assert!(super::is_minlexed(&minlex(&[1, 3, 45, 62, 63, 72, 78, 80])));
        assert!(super::is_minlexed(&minlex(&[
            25, 40, 54, 64, 70, 71, 75, 80
        ])));
        assert!(super::is_minlexed(&minlex(&[
            7, 19, 49, 58, 59, 67, 75, 77
        ])));
    }

    #[test]
    fn test_obviously_invalid() {
        assert!(super::is_obviously_invalid(&[0, 1, 2, 3, 4, 5, 6, 16]));
        assert!(super::is_obviously_invalid(&[0, 1, 2, 6, 12, 13, 14]));
        assert!(super::is_obviously_invalid(&[0, 1, 2, 6, 7, 12, 13]));
        assert!(!super::is_obviously_invalid(&[0, 1, 2, 3, 4, 5, 6, 7]));
    }
}
