//! Talus
//! =====
//!
//! A collection of computational topology algorithms written in Rust, with Python bindings.
///
/// The current use case covered by this crate is the creation of kNN graphs, and the computation
/// of the MorseSmaleComplex of those graphs (and the corresponding persistence values for the
/// extrema in the graph).
use std::fs::File;
use std::f64;
use std::error::Error;
use std::path::Path;
use std::io::BufReader;
use csv::StringRecord;

#[macro_use] extern crate cpython;

pub mod morse;
pub mod graph;
pub mod python;


pub trait Metric {
    fn distance(&self, other: &Self) -> f64;
}

impl Metric for Vec<f64> {
    fn distance(&self, other:&Self) -> f64 {
        self.iter().zip(other.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>().sqrt()
    }
}

pub trait PreMetric {
    fn predistance(&self, other: &Self) -> f64;
}

impl PreMetric for Vec<f64> {
    fn predistance(&self, other:&Self) -> f64 {
        self.distance(other)
    }
}

/// A point in a graph that contains enough information to allow for Morse complex construction
///
///
#[derive(Debug)]
pub struct LabeledPoint<T> {
    /// An identifier for this point. Assumed to be unique.
    pub id: i64,

    /// The value denoting the point's location in some space. Used for distance computations.
    pub point: T,

    /// The scalar value associated with this point. 
    ///
    /// This is the value that is used to determine extrema in the graph.
    ///
    /// Mathematically speaking, this corresponds to the value of some morse function at this
    /// point.
    pub value: f64
}


impl<T: Clone> Clone for LabeledPoint<T> {
    fn clone(&self) -> Self {
        LabeledPoint{value: self.value, point: self.point.clone(), id: self.id}
    }
}

impl LabeledPoint<Vec<f64>> {
    pub fn from_record(record: &StringRecord) -> LabeledPoint<Vec<f64>> {
        let id = record[0].parse::<i64>().expect("Expected an int");
        let value = record[1].parse::<f64>().expect("Expected a float");
        let point = record.iter()
            .skip(2)
            .map(|v| v.parse::<f64>().expect("Expected a float"))
            .collect();
        LabeledPoint{id, point, value}
    }

    pub fn points_from_file<P: AsRef<Path>>(filename: P) -> Result<Vec<LabeledPoint<Vec<f64>>>, Box<dyn Error>> {
        let f = File::open(filename).expect("Unable to open file");
        let f = BufReader::new(f);
        let mut points = Vec::with_capacity(16);
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(f);
        for result in rdr.records() {
            let mut record = result?;
            record.trim();
            points.push(LabeledPoint::from_record(&record));
        }
        Ok(points)
    }
}
