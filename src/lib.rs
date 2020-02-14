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
use std::collections::HashMap;
use std::path::Path;
use std::io::BufReader;
use csv::StringRecord;
use petgraph::graph::{UnGraph, NodeIndex};


pub mod morse;
pub mod graph;

#[macro_use] extern crate cpython;
use cpython::{PyResult, Python, PyList, PyTuple, PyObject, ToPyObject, FromPyObject};
use crate::cpython::ObjectProtocol;


py_module_initializer!(talus, inittalus, PyInit_talus, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    m.add(py, "_persistence", py_fn!(py, persistence_py(nodes: PyList, edges: PyList)))?;
    m.add(py, "_persistence_by_knn", py_fn!(py, knn_persistence_py(points: PyList, k: usize)))?;
    m.add(py, "_persistence_by_approximate_knn", py_fn!(py, approximate_knn_persistence_py(points: PyList, k: usize, sample_rate: f64, precision: f64)))?;
    Ok(())
});

fn approximate_knn_persistence_py(py: Python, points: PyList, k: usize, sample_rate: f64, precision: f64) -> PyResult<PyTuple> {
    let mut labeled_points = Vec::with_capacity(points.len(py));
    for point in points.iter(py) {
        labeled_points.push(point.extract(py)?);
    }
    let g = graph::build_knn_approximate(&labeled_points, k, sample_rate, precision);
    let complex = morse::MorseSmaleComplex::from_graph(&g);
    let data = complex.to_data(&g);
    Ok(data.into_py_object(py))
}

fn knn_persistence_py(py: Python, points: PyList, k: usize) -> PyResult<PyTuple> {
    let mut labeled_points = Vec::with_capacity(points.len(py));
    for point in points.iter(py) {
        labeled_points.push(point.extract(py)?);
    }
    let g = graph::build_knn(&labeled_points, k);
    let complex = morse::MorseSmaleComplex::from_graph(&g);
    let data = complex.to_data(&g);
    Ok(data.into_py_object(py))
}

fn persistence_py(py: Python, nodes: PyList, edges: PyList) -> PyResult<PyTuple> {
    let mut labeled_nodes: Vec<NodeIndex> = Vec::with_capacity(nodes.len(py));
    let mut id_lookup: HashMap<i64, (usize, NodeIndex)> = HashMap::with_capacity(nodes.len(py));
    let mut g = UnGraph::new_undirected();
    for (i, node) in nodes.iter(py).enumerate() {
        let point: LabeledPoint<Vec<f64>> = node.extract(py)?;
        let node = g.add_node(point.clone());
        labeled_nodes.push(node);
        id_lookup.insert(point.id, (i, node));
    }
    for edge in edges.iter(py) {
        let node_tuple: PyTuple = edge.extract(py)?;
        let left: i64 = node_tuple.get_item(py, 0).extract(py)?;
        let right: i64 = node_tuple.get_item(py, 1).extract(py)?;
        g.add_edge((id_lookup.get(&left).unwrap()).1, id_lookup.get(&right).unwrap().1, 1.);
    }
    let complex = morse::MorseSmaleComplex::from_graph(&g);
    let data = complex.to_data(&g);
    Ok(data.into_py_object(py))
}

impl ToPyObject for morse::MorseComplexData {
    type ObjectType = PyTuple;
    fn to_py_object(&self, py: Python) -> Self::ObjectType {
        (self.lifetimes.clone(), self.filtration.clone(), self.complex.clone()).to_py_object(py)
    }

    fn into_py_object(self, py: Python) -> Self::ObjectType {
        (self.lifetimes, self.filtration, self.complex).to_py_object(py)
    }
}


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

    /// FIXME The vector denoting the points location in some space. Used for distance computations.
    pub point: T,

    /// The scalar value associated with this point. 
    ///
    /// This is the value that is used to determine extrema in the graph.
    ///
    /// Mathematically speaking, this corresponds to the value of some morse function at this
    /// point.
    pub value: f64
}

impl<'s> FromPyObject<'s> for LabeledPoint<Vec<f64>> {
    fn extract(py: Python, obj: &'s PyObject) -> PyResult<Self>{
        let id: i64 = obj.getattr(py, "identifier")?.extract(py)?;
        let value: f64 = obj.getattr(py, "value")?.extract(py)?;
        let list: PyList = obj.getattr(py, "vector")?.extract(py)?;
        let mut point: Vec<f64> = Vec::with_capacity(list.len(py));
        for value in list.iter(py) {
            let v = value.extract(py)?;
            point.push(v);
        };
        Ok(LabeledPoint{id, value, point})
    }
}

impl<T: Clone> Clone for LabeledPoint<T> {
    fn clone(&self) -> Self {
        LabeledPoint{value: self.value, point: self.point.clone(), id: self.id}
    }
}

impl LabeledPoint<Vec<f64>> {
    // FIXME: move hte vec stuff to its own impl
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
