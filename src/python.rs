use std::collections::HashMap;
use petgraph::graph::{UnGraph, NodeIndex};
use super::{graph, morse, LabeledPoint};

use cpython::{PyResult, Python, PyList, PyTuple, PyObject, ToPyObject, FromPyObject, PyErr, exc, PyString};
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
    let g = match graph::build_knn_approximate(&labeled_points, k, sample_rate, precision){
        Err(err) => return Err(PyErr::new::<exc::Exception, PyString>(py, PyString::new(py, &format!("{:?}", err)))),
        Ok(g) => g
    };
    let complex = match morse::MorseSmaleComplex::from_graph(&g) {
        Err(err) => return Err(PyErr::new::<exc::Exception, PyString>(py, PyString::new(py, &format!("{:?}", err)))),
        Ok(complex) => complex
    };
    let data = complex.to_data(&g);
    Ok(data.into_py_object(py))
}

fn knn_persistence_py(py: Python, points: PyList, k: usize) -> PyResult<PyTuple> {
    let mut labeled_points = Vec::with_capacity(points.len(py));
    for point in points.iter(py) {
        labeled_points.push(point.extract(py)?);
    }
    let g = match graph::build_knn(&labeled_points, k) {
        Err(err) => return Err(PyErr::new::<exc::Exception, PyString>(py, PyString::new(py, &format!("{:?}", err)))),
        Ok(g) => g
    };
    let complex = match morse::MorseSmaleComplex::from_graph(&g) {
        Err(err) => return Err(PyErr::new::<exc::Exception, PyString>(py, PyString::new(py, &format!("{:?}", err)))),
        Ok(complex) => complex
    };
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
    let complex = match morse::MorseSmaleComplex::from_graph(&g) {
        Err(err) => return Err(PyErr::new::<exc::Exception, PyString>(py, PyString::new(py, &format!("{:?}", err)))),
        Ok(complex) => complex
    };
    let data = complex.to_data(&g);
    Ok(data.into_py_object(py))
}

impl ToPyObject for MorseComplexData {
    type ObjectType = PyTuple;
    fn to_py_object(&self, py: Python) -> Self::ObjectType {
        (self.lifetimes.clone(), self.filtration.clone(), self.complex.clone()).to_py_object(py)
    }

    fn into_py_object(self, py: Python) -> Self::ObjectType {
        (self.lifetimes, self.filtration, self.complex).to_py_object(py)
    }
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


/// A struct that captures the important data about a MorseSmaleComplex
pub struct MorseComplexData {
    pub lifetimes: HashMap<i64, f64>,
    pub filtration: Vec<(f64, i64, i64)>,
    pub complex: Vec<(i64, i64)>
}


impl morse::MorseSmaleComplex {
    fn to_data<T>(&self, graph: &UnGraph<LabeledPoint<T>, f64>) -> (MorseComplexData, MorseComplexData) {
        (self.descending_complex.to_data(graph), self.ascending_complex.to_data(graph))
    }
}

impl morse::MorseComplex {
    fn to_data<T>(&self, graph: &UnGraph<LabeledPoint<T>, f64>) -> MorseComplexData {
        let lifetimes = self.get_persistence();
        let filtration = &self.filtration;
        let lifetimes: HashMap<i64, f64> = lifetimes.iter()
            .map(|(k,v)| {
                let id = graph.node_weight(*k).unwrap().id;
                (id, *v)
            })
            .collect();
        let filtration: Vec<(f64, i64, i64)> = filtration.iter()
            .map(|filtration| {
                (filtration.time, graph.node_weight(filtration.destroyed_cell).unwrap().id, graph.node_weight(filtration.owning_cell).unwrap().id)
            })
            .collect();
        let complex: Vec<(i64, i64)> = self.get_complex().iter()
            .map(|(node, ancestor)| {
                (graph.node_weight(*node).unwrap().id, graph.node_weight(*ancestor).unwrap().id)
            })
            .collect();
        MorseComplexData{lifetimes, filtration, complex}
    }
}
