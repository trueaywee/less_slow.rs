//! # Less Slow
//!
//! ## Low-level microbenchmarks for building a performance-first mindset
//!
//! This module provides microbenchmarks to build a performance-first mindset.
//! Focus is on tradeoffs of abstractions, covering costs of numerical operations,
//! micro-kernels, parallelism, and more.
//!
//! Key principles are written in Rust but apply universally, and have been reproduced
//! in other languages:
//!
//! - [C++ Benchmarks](https://github.com/ashvardanian/less_slow.cpp)
//! - [Python Benchmarks](https://github.com/ashvardanian/less_slow.py)
//! - [Go Benchmarks](https://github.com/ashvardanian/less_slow.go)
//!
#![allow(dead_code)]
#![allow(unused_imports)]
#![feature(coroutines)] // Called `generator` before.
#![feature(coroutine_trait)]
#![feature(stmt_expr_attributes)]

use std::cmp;
use std::iter;
use std::ops::RangeInclusive;
use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// region: Pipelines and Abstractions

//? Designing efficient micro-kernels is hard, but composing them into
//? high-level pipelines without losing performance is just as difficult.
//?
//? Consider a hypothetical numeric processing pipeline:
//?
//?    1. Generate all integers in a given range (e.g., [1, 49]).
//?    2. Filter out integers that are perfect squares.
//?    3. Expand each remaining number into its prime factors.
//?    4. Sum all prime factors from the filtered numbers.
//?
//? We'll explore four implementations of this pipeline:
//?
//?   - **Callback-based Pipeline** using closures,
//?   - **Iterator-based Pipeline** using a custom `PrimeFactors` iterator,
//?   - **Coroutine/Generator-based Pipeline** using the [`async-stream`] crate,
//?   - **Trait Objects** ("virtual" functions in C++-speak) to compose stages dynamically.

/// For demonstration, we'll replicate the pipeline on [3..=49].
const PIPE_START: u64 = 3;
const PIPE_END: u64 = 49;

/// Checks if an integer is a power of two.
#[inline(always)]
fn is_power_of_two(x: u64) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

/// Checks if an integer is a power of three.
#[inline(always)]
fn is_power_of_three(x: u64) -> bool {
    const MAX_POWER_OF_THREE: u64 = 12157665459056928801;
    x > 0 && (MAX_POWER_OF_THREE % x == 0)
}

// region: Closures

/// Factorizes `number` into primes, invoking `callback` for each factor.
fn prime_factors_closure<F>(mut number: u64, mut callback: F)
where
    F: FnMut(u64),
{
    let mut factor: u64 = 2;
    while number > 1 {
        if number % factor == 0 {
            callback(factor);
            number /= factor;
        } else {
            factor += if factor == 2 { 1 } else { 2 };
        }
    }
}

/// Callback-based pipeline.
fn pipeline_closures() -> (u64, u64) {
    let mut sum = 0;
    let mut count = 0;

    for value in PIPE_START..=PIPE_END {
        if !is_power_of_two(value) && !is_power_of_three(value) {
            prime_factors_closure(value, |factor| {
                sum += factor;
                count += 1;
            });
        }
    }
    (sum, count)
}

// endregion: Closures

// region: Iterators

/// Lazily evaluates the prime factors of a single integer.
struct PrimeFactors {
    number: u64,
    factor: u64,
}

impl PrimeFactors {
    /// Creates a new `PrimeFactors` iterator from `number`.
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

/// Iterator-based pipeline.
fn pipeline_iterators() -> (u64, u64) {
    (PIPE_START..=PIPE_END)
        .filter(|&v| !is_power_of_two(v) && !is_power_of_three(v))
        .flat_map(|v| PrimeFactors::new(v)) // Use our lazy iterator
        .fold((0, 0), |(sum, count), factor| (sum + factor, count + 1))
}

// endregion: Iterators

// region: Polymorphism

/// A trait that processes a vector of data as a pipeline stage.
trait PipelineStage {
    fn process(&self, data: &mut Vec<u64>);
}

/// A stage that pushes `[start..=end]` into `data`, clearing it first.
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

/// A stage that filters out elements according to a function `predicate`.
struct FilterVirtual {
    predicate: fn(u64) -> bool,
}
impl PipelineStage for FilterVirtual {
    fn process(&self, data: &mut Vec<u64>) {
        data.retain(|&x| !(self.predicate)(x));
    }
}

/// A stage that expands numbers into their prime factors, pushing them into `data`.
///
/// Uses `prime_factors_closure` to avoid building an intermediate `Vec<u64>`.
struct PrimeFactorsVirtual;
impl PipelineStage for PrimeFactorsVirtual {
    fn process(&self, data: &mut Vec<u64>) {
        let mut expanded = Vec::new();
        for &value in data.iter() {
            prime_factors_closure(value, |f| expanded.push(f));
        }
        *data = expanded;
    }
}

/// A pipeline that holds multiple stages as trait objects (akin to a "homogeneous virtual pipeline" in C++).
struct HomogeneousVirtualPipeline {
    stages: Vec<Box<dyn PipelineStage>>,
}

impl HomogeneousVirtualPipeline {
    fn new() -> Self {
        Self { stages: Vec::new() }
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

/// Trait-object-based pipeline.
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

// endregion: Polymorphism

// region: Coroutines

/// A generator that produces integers in the range [start..=end].
fn for_range_coroutine(start: u64, end: u64) -> impl Coroutine<Yield = u64, Return = ()> {
    #[coroutine]
    move || {
        for value in start..=end {
            yield value;
        }
    }
}

/// A generator that filters values from the input generator based on a predicate.
fn filter_coroutine<G>(
    input: G,
    predicate: fn(u64) -> bool,
) -> impl Coroutine<Yield = u64, Return = ()>
where
    G: Coroutine<Yield = u64, Return = ()> + Unpin,
{
    #[coroutine]
    move || {
        let mut input = input;
        while let CoroutineState::Yielded(value) = Pin::new(&mut input).resume(()) {
            if !predicate(value) {
                yield value;
            }
        }
    }
}

/// A generator that expands values into their prime factors.
fn prime_factors_coroutine<G>(input: G) -> impl Coroutine<Yield = u64, Return = ()>
where
    G: Coroutine<Yield = u64, Return = ()> + Unpin,
{
    #[coroutine]
    move || {
        let mut input = input;
        while let CoroutineState::Yielded(mut number) = Pin::new(&mut input).resume(()) {
            let mut factor: u64 = 2;
            while number > 1 {
                if number % factor == 0 {
                    yield factor;
                    number /= factor;
                } else {
                    factor += if factor == 2 { 1 } else { 2 };
                }
            }
        }
    }
}

/// Coroutine-based pipeline.
fn pipeline_coroutines() -> (u64, u64) {
    let range_gen = for_range_coroutine(PIPE_START, PIPE_END);
    let filtered_gen = filter_coroutine(
        filter_coroutine(range_gen, is_power_of_two),
        is_power_of_three,
    );
    let factor_gen = prime_factors_coroutine(filtered_gen);

    let mut sum: u64 = 0;
    let mut count: u64 = 0;
    let mut generator = factor_gen;

    while let CoroutineState::Yielded(factor) = Pin::new(&mut generator).resume(()) {
        sum += factor;
        count += 1;
    }

    (sum, count)
}

// endregion: Coroutines

// endregion: Pipelines and Abstractions

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

pub fn benchmark_coroutines(c: &mut Criterion) {
    c.bench_function("pipeline_coroutines", |b| {
        b.iter(|| {
            let (sum, count) = pipeline_coroutines();
            black_box(sum);
            black_box(count);
        })
    });
}

// Group them into a benchmark suite.
criterion_group!(
    name = benchmarks;
    config = Criterion::default();
    targets =
        benchmark_closures,
        benchmark_iterators,
        benchmark_virtual,
        benchmark_coroutines,
);

criterion_main!(benchmarks);
