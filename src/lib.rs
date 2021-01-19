//! Talus
//! =====
//!
//! A collection of computational topology algorithms written in Rust, with Python bindings.
//!
//! Talus provides functionality for creating graphs in the graph module, and provides the ability
//! compute the approximate Morse-Smale complex in the morse module. See the documentation for
//! those modules for further details and explanations
//!
//! Information on Python bindings can be found under the Python module.
//!
use std::fs::File;
use std::f64;
use std::error::Error;
use std::path::Path;
use std::io::BufReader;
use csv::StringRecord;

pub mod morse;
pub mod graph;
pub mod simplex;
pub mod python;
pub mod binomial;


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

impl From<python::MorseNode> for LabeledPoint<Vec<f64>> {
    fn from(item: python::MorseNode) -> Self {
        LabeledPoint{value: item.value, point: item.vector, id: item.identifier}
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
