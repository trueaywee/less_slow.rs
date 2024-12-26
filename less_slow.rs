#![allow(dead_code)]
#![allow(unused_imports)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::cmp;
use std::iter;
use std::ops::RangeInclusive;

/// For demonstration, we'll replicate the pipeline on [3..=49].
const PIPE_START: u64 = 3;
const PIPE_END: u64 = 49;

// ---------------------------------------------------------
//  Utility functions: is_power_of_two, is_power_of_three, prime_factors
// ---------------------------------------------------------

#[inline(always)]
fn is_power_of_two(x: u64) -> bool {
    // A number is a power of two if x & (x - 1) == 0, for x != 0
    x != 0 && (x & (x - 1)) == 0
}

#[inline(always)]
fn is_power_of_three(x: u64) -> bool {
    // The largest power of three in 64-bit is 3^40 = 12157665459056928801
    const MAX_POWER_OF_THREE: u64 = 12157665459056928801;
    x > 0 && (MAX_POWER_OF_THREE % x == 0)
}

#[inline(always)]
fn prime_factors(mut number: u64) -> Vec<u64> {
    let mut factors = Vec::new();

    // Factor out powers of 2
    while number % 2 == 0 && number > 0 {
        factors.push(2);
        number /= 2;
    }

    // Factor out odd numbers
    let mut factor = 3;
    while factor * factor <= number {
        while number % factor == 0 {
            factors.push(factor);
            number /= factor;
        }
        factor += 2;
    }

    // If leftover, it's a prime
    if number > 1 {
        factors.push(number);
    }

    factors
}

// ---------------------------------------------------------
//  1) Pipeline using closures
// ---------------------------------------------------------

/// Finds prime factors of a number and applies the callback for each factor.
fn prime_factors_callback<F>(mut number: u64, mut callback: F)
where
    F: FnMut(u64),
{
    let mut factor = 2;

    while number > 1 {
        if number % factor == 0 {
            callback(factor);
            number /= factor;
        } else {
            factor += if factor == 2 { 1 } else { 2 };
        }
    }
}

fn pipeline_closures() -> (u64, u64) {
    let mut sum = 0u64;
    let mut count = 0u64;

    for value in PIPE_START..=PIPE_END {
        if !is_power_of_two(value) && !is_power_of_three(value) {
            prime_factors_callback(value, |factor| {
                sum += factor;
                count += 1;
            });
        }
    }

    (sum, count)
}
// ---------------------------------------------------------
//  3) Pipeline using Iterators (Ranges)
// ---------------------------------------------------------

/// Lazily evaluates the prime factors of a number.
struct PrimeFactors {
    number: u64,
    factor: u64,
}

impl PrimeFactors {
    fn new(number: u64) -> Self {
        PrimeFactors { number, factor: 2 }
    }
}

impl Iterator for PrimeFactors {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        while self.number > 1 {
            if self.number % self.factor == 0 {
                self.number /= self.factor;
                return Some(self.factor);
            }
            if self.factor == 2 {
                self.factor = 3;
            } else {
                self.factor += 2;
            }
        }
        None
    }
}

fn pipeline_iterators() -> (u64, u64) {
    (PIPE_START..=PIPE_END)
        .filter(|&v| !is_power_of_two(v) && !is_power_of_three(v))
        .flat_map(|v| PrimeFactors::new(v)) // Use the lazy iterator
        .fold((0, 0), |(sum, count), f| (sum + f, count + 1))
}

// ---------------------------------------------------------
//  4) Polymorphism: trait objects akin to virtual classes
// ---------------------------------------------------------

trait PipelineStage {
    fn process(&self, data: &mut Vec<u64>);
}

/// A stage that pushes [start..=end] into `data`.
struct ForRangeVirtual {
    start: u64,
    end: u64,
}
impl PipelineStage for ForRangeVirtual {
    fn process(&self, data: &mut Vec<u64>) {
        data.clear();
        for v in self.start..=self.end {
            data.push(v);
        }
    }
}

/// A stage that filters out elements according to a function predicate.
struct FilterVirtual {
    predicate: fn(u64) -> bool,
}
impl PipelineStage for FilterVirtual {
    fn process(&self, data: &mut Vec<u64>) {
        data.retain(|&x| !(self.predicate)(x));
    }
}

/// A stage that expands numbers into their prime factors.
struct PrimeFactorsVirtual;
impl PipelineStage for PrimeFactorsVirtual {
    fn process(&self, data: &mut Vec<u64>) {
        let mut expanded = Vec::new();
        for &value in data.iter() {
            expanded.extend(prime_factors(value));
        }
        *data = expanded;
    }
}

/// A pipeline that holds multiple stages as trait objects (like `virtual` in C++).
struct HomogeneousVirtualPipeline {
    stages: Vec<Box<dyn PipelineStage>>,
}
impl HomogeneousVirtualPipeline {
    fn new() -> Self {
        HomogeneousVirtualPipeline { stages: Vec::new() }
    }
    fn add_stage(&mut self, stage: Box<dyn PipelineStage>) {
        self.stages.push(stage);
    }
}
impl PipelineStage for HomogeneousVirtualPipeline {
    fn process(&self, data: &mut Vec<u64>) {
        for stage in &self.stages {
            stage.process(data);
        }
    }
}

fn pipeline_virtual() -> (u64, u64) {
    let mut pipeline = HomogeneousVirtualPipeline::new();
    pipeline.add_stage(Box::new(ForRangeVirtual {
        start: PIPE_START,
        end: PIPE_END,
    }));
    pipeline.add_stage(Box::new(FilterVirtual {
        predicate: is_power_of_two,
    }));
    pipeline.add_stage(Box::new(FilterVirtual {
        predicate: is_power_of_three,
    }));
    pipeline.add_stage(Box::new(PrimeFactorsVirtual));

    let mut data = Vec::new();
    pipeline.process(&mut data);

    let sum = data.iter().copied().sum();
    let count = data.len() as u64;
    (sum, count)
}

// ---------------------------------------------------------
//  Criterion Benchmarks
// ---------------------------------------------------------

pub fn benchmark_closures(c: &mut Criterion) {
    c.bench_function("pipeline_closures", |b| {
        b.iter(|| {
            let (sum, count) = pipeline_closures();
            black_box(sum);
            black_box(count);
        })
    });
}

pub fn benchmark_iterators(c: &mut Criterion) {
    c.bench_function("pipeline_iterators", |b| {
        b.iter(|| {
            let (sum, count) = pipeline_iterators();
            black_box(sum);
            black_box(count);
        })
    });
}

pub fn benchmark_virtual(c: &mut Criterion) {
    c.bench_function("pipeline_virtual", |b| {
        b.iter(|| {
            let (sum, count) = pipeline_virtual();
            black_box(sum);
            black_box(count);
        })
    });
}

// The group needs a trailing comma after each target:
criterion_group!(
    name = benchmarks;
    config = Criterion::default();
    targets =
        benchmark_closures,
        benchmark_iterators,
        benchmark_virtual,
);

criterion_main!(benchmarks);
