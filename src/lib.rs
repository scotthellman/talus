use ndarray::prelude::*;

use std::fs::File;
use std::f64;
use std::error::Error;
use std::collections::HashMap;
use std::path::Path;
use std::io::BufReader;
use csv::StringRecord;
use petgraph::graph::{Graph, NodeIndex};


pub mod morse;
pub mod graph;

#[macro_use] extern crate cpython;
use cpython::{PyResult, Python, PyList, PyTuple, PyDict, PyObject, ToPyObject, FromPyObject};
use crate::cpython::ObjectProtocol;


py_module_initializer!(talus, inittalus, PyInit_talus, |py, m| {
    m.add(py, "__doc__", "This module is implemented in Rust.")?;
    m.add(py, "_persistence", py_fn!(py, persistence_py(nodes: PyList, edges: PyList)))?;
    m.add(py, "_persistence_by_knn", py_fn!(py, knn_persistence_py(points: PyList, k: usize)))?;
    Ok(())
});

fn knn_persistence_py(py: Python, points: PyList, k: usize) -> PyResult<PyDict> {
    let mut labeled_points = Vec::with_capacity(points.len(py));
    for point in points.iter(py) {
        labeled_points.push(point.extract(py)?);
    }
    println!("about to build the graph");
    let mut g = graph::build_knn(&labeled_points, k);
    println!("built");
    let mut complex = morse::MorseComplex::from_graph(&mut g);
    let lifetimes = complex.compute_persistence();
    let lifetimes: HashMap<i64, f64> = lifetimes.iter()
        .map(|(k,v)| {
            let id = g.node_weight(*k).unwrap().id;
            (id, *v)
        })
        .collect();
    Ok(lifetimes.to_py_object(py))
}

fn persistence_py(py: Python, nodes: PyList, edges: PyList) -> PyResult<PyDict> {
    let mut labeled_nodes: Vec<NodeIndex> = Vec::with_capacity(nodes.len(py));
    let mut id_lookup: HashMap<i64, (usize, NodeIndex)> = HashMap::with_capacity(nodes.len(py));
    let mut g = Graph::new_undirected();
    for (i, node) in nodes.iter(py).enumerate() {
        let point: LabeledPoint = node.extract(py)?;
        let node = g.add_node(point.to_owned());
        labeled_nodes.push(node);
        id_lookup.insert(point.id, (i, node));
    }
    for edge in edges.iter(py) {
        let node_tuple: PyTuple = edge.extract(py)?;
        let left: i64 = node_tuple.get_item(py, 0).extract(py)?;
        let right: i64 = node_tuple.get_item(py, 1).extract(py)?;
        g.add_edge((id_lookup.get(&left).unwrap()).1, id_lookup.get(&right).unwrap().1, 0.);
    }
    let mut complex = morse::MorseComplex::from_graph(&mut g);
    let lifetimes = complex.compute_persistence();
    let lifetimes: HashMap<i64, f64> = lifetimes.iter()
        .map(|(k,v)| {
            let id = g.node_weight(*k).unwrap().id;
            (id, *v)
        })
        .collect();
    Ok(lifetimes.to_py_object(py))
}

#[derive(Debug)]
pub struct LabeledPoint {
    pub id: i64,
    pub point: Vec<f64>,
    pub value: f64
}

impl<'s> FromPyObject<'s> for LabeledPoint {
    fn extract(py: Python, obj: &'s PyObject) -> PyResult<Self>{
        let id: i64 = obj.getattr(py, "identifier")?.extract(py)?;
        let value: f64 = obj.getattr(py, "value")?.extract(py)?;
        let list: PyList = obj.getattr(py, "vector")?.extract(py)?;
        let mut point: Vec<f64> = Vec::with_capacity(list.len(py));
        for (i, value) in list.iter(py).enumerate() {
            let v = value.extract(py)?;
            point.push(v);
        };
        Ok(LabeledPoint{id, value, point})
    }

}

impl LabeledPoint {
    pub fn from_record(record: &StringRecord) -> LabeledPoint {
        let id = record[0].parse::<i64>().expect("Expected an int");
        let value = record[1].parse::<f64>().expect("Expected a float");
        let point = record.iter()
            .skip(2)
            .map(|v| v.parse::<f64>().expect("Expected a float"))
            .collect();
        LabeledPoint{id, point, value}
    }

    pub fn to_owned(&self) -> LabeledPoint {
        // This is basically clone? I'm just copying the name from ndarray for now
        LabeledPoint{value: self.value, point: self.point.to_owned(), id: self.id}
    }

    pub fn points_from_file<P: AsRef<Path>>(filename: P) -> Result<Vec<LabeledPoint>, Box<dyn Error>> {
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
