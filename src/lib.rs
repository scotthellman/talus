#![crate_type = "cdylib"]
use ndarray::prelude::*;

use std::fs::File;
use std::f64;
use std::error::Error;
use std::io::BufReader;
use csv::StringRecord;

pub mod morse;
pub mod graph;
#[macro_use] extern crate cpython;
use cpython::{PyResult, Python};


// add bindings to the generated python module
// N.B: names: "rust2py" must be the name of the `.so` or `.pyd` file
py_module_initializer!(topology, inittopology, PyInit_topology, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    m.add(py, "sum_as_string", py_fn!(py, sum_as_string_py(a: i64, b:i64)))?;
    Ok(())
});

// logic implemented as a normal rust function
fn sum_as_string(a:i64, b:i64) -> String {
    format!("{}", a + b).to_string()
}

// rust-cpython aware function. All of our python interface could be
// declared in a separate module.
// Note that the py_fn!() macro automatically converts the arguments from
// Python objects to Rust values; and the Rust return value back into a Python object.
fn sum_as_string_py(_: Python, a:i64, b:i64) -> PyResult<String> {
    let out = sum_as_string(a, b);
    Ok(out)
}

#[derive(Debug)]
pub struct LabeledPoint {
    pub point: Array1<f64>,
    pub label: f64
}

impl LabeledPoint {
    pub fn from_record(record: &StringRecord) -> LabeledPoint {
        let label = record[record.len() - 1].parse::<f64>().expect("Expected a float");
        let point = record.iter()
            .take(record.len() - 1)
            .map(|v| v.parse::<f64>().expect("Expected a float"))
            .collect();
        LabeledPoint{point, label}
    }

    pub fn to_owned(&self) -> LabeledPoint {
        // This is basically clone? I'm just copying the name from ndarray for now
        LabeledPoint{label: self.label, point: self.point.to_owned()}
    }

    pub fn points_from_file(filename: &str) -> Result<Vec<LabeledPoint>, Box<dyn Error>> {
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

    fn grade(&self, other: &LabeledPoint) -> f64{
        let diff = &self.point - &other.point;
        let distance = diff.dot(&diff).sqrt();
        let label_diff = self.label - other.label;
        label_diff / distance
    }
}
