use ndarray::prelude::*;

use std::fs::File;
use std::f64;
use std::error::Error;
use std::collections::HashMap;
use std::io::BufReader;
use csv::StringRecord;
use petgraph::graph::{Graph, NodeIndex};

pub mod morse;
pub mod graph;
#[macro_use] extern crate cpython;
use cpython::{PyResult, Python, PyList, PyTuple, PyObject, PyFloat, PyInt};


// add bindings to the generated python module
// N.B: names: "rust2py" must be the name of the `.so` or `.pyd` file
py_module_initializer!(topology, inittopology, PyInit_topology, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    m.add(py, "persistence", py_fn!(py, persistence_py(nodes: PyList, edges: PyList)))?;
    Ok(())
});

// rust-cpython aware function. All of our python interface could be
// declared in a separate module.
// Note that the py_fn!() macro automatically converts the arguments from
// Python objects to Rust values; and the Rust return value back into a Python object.
fn persistence_py(py: Python, nodes: PyList, edges: PyList) -> PyResult<String> {
    let mut labeled_nodes: Vec<NodeIndex> = Vec::with_capacity(nodes.len(py));
    let mut id_lookup: HashMap<i64, (usize, NodeIndex)> = HashMap::with_capacity(nodes.len(py));
    let mut g = Graph::new_undirected();
    for (i, node) in nodes.iter(py).enumerate() {
        // FIXME: so this really shows how bad LabeledPoint is, no way to store the id
        let node_tuple: PyTuple = node.extract(py)?;
        let id: i64 = node_tuple.get_item(py, 0).extract(py)?;
        let label: f64 = node_tuple.get_item(py, 1).extract(py)?;
        let point = LabeledPoint{point:arr1(&[]), label};
        let node = g.add_node(point.to_owned());
        labeled_nodes.push(node);
        id_lookup.insert(id, (i, node));
    }
    for edge in edges.iter(py) {
        let node_tuple: PyTuple = edge.extract(py)?;
        let left: i64 = node_tuple.get_item(py, 0).extract(py)?;
        let right: i64 = node_tuple.get_item(py, 1).extract(py)?;
        g.add_edge((id_lookup.get(&left).unwrap()).1, id_lookup.get(&right).unwrap().1, 0.);
    }
    let mut complex = morse::MorseComplex::from_graph(&mut g);
    let lifetimes = complex.compute_persistence();
    Ok(format!("{:?}", lifetimes))
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
