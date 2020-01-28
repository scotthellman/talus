use ndarray::prelude::*;
use petgraph::graph::Graph;
use petgraph::dot::Dot;

use std::f64;

use topology::LabeledPoint;

fn main() {
    let points = match LabeledPoint::points_from_file("points.txt") {
        Ok(points) => points,
        Err(e) => {
            println!("Failed to parse points: {}", e);
            panic!();
        }
    };
    let mut graph = topology::graph::build_knn(&points, 2);
    println!("Graph is {:?}", graph);
    println!("{:?}", Dot::with_config(&graph, &[]));
    let mut complex = topology::morse::MorseComplex::from_graph(&mut graph);
    let lifetimes = complex.compute_persistence();
    println!("Lifetimes were {:?}", lifetimes);
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single() {
        let mut graph = Graph::new_undirected();
        let points = [
            LabeledPoint{label: -1., point: arr1(&[0., 0.])},
            LabeledPoint{label: 1., point: arr1(&[1., 0.])},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 0.);
        let mut complex = topology::morse::MorseComplex::from_graph(&mut graph);
        let lifetimes = complex.compute_persistence();
        assert_eq!(lifetimes[&node_lookup[0]], 0.);
        assert_eq!(lifetimes[&node_lookup[1]], f64::INFINITY);
    }

    #[test]
    fn test_triangle() {
        let mut graph = Graph::new_undirected();
        let points = [
            LabeledPoint{label: -1., point: arr1(&[0., 0.])},
            LabeledPoint{label: 0., point: arr1(&[1., 1.])},
            LabeledPoint{label: 1., point: arr1(&[1., 0.])},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 0.);
        graph.add_edge(node_lookup[0], node_lookup[2], 0.);
        graph.add_edge(node_lookup[1], node_lookup[2], 0.);
        let mut complex = topology::morse::MorseComplex::from_graph(&mut graph);
        let lifetimes = complex.compute_persistence();
        assert_eq!(lifetimes[&node_lookup[0]], 0.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], f64::INFINITY);
    }

    #[test]
    fn test_square() {
        let mut graph = Graph::new_undirected();
        let points = [
            LabeledPoint{label: 1., point: arr1(&[0., 0.])},
            LabeledPoint{label: -1., point: arr1(&[1., 0.])},
            LabeledPoint{label: 0., point: arr1(&[0., 1.])},
            LabeledPoint{label: 2., point: arr1(&[1., 1.])},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 0.);
        graph.add_edge(node_lookup[0], node_lookup[2], 0.);
        graph.add_edge(node_lookup[1], node_lookup[3], 0.);
        graph.add_edge(node_lookup[2], node_lookup[3], 0.);
        let mut complex = topology::morse::MorseComplex::from_graph(&mut graph);
        let lifetimes = complex.compute_persistence();
        assert_eq!(lifetimes[&node_lookup[0]], 1.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], 0.);
        assert_eq!(lifetimes[&node_lookup[3]], f64::INFINITY);
    }

    #[test]
    fn test_big_square() {
        let mut graph = Graph::new_undirected();
        let points = [
            LabeledPoint{label: 6., point: arr1(&[0., 0.])},
            LabeledPoint{label: 2., point: arr1(&[1., 0.])},
            LabeledPoint{label: 3., point: arr1(&[2., 0.])},
            LabeledPoint{label: 5., point: arr1(&[0., 1.])},
            LabeledPoint{label: 4., point: arr1(&[1., 1.])},
            LabeledPoint{label: -5., point: arr1(&[1., 2.])},
            LabeledPoint{label: 0., point: arr1(&[0., 2.])},
            LabeledPoint{label: 1., point: arr1(&[1., 2.])},
            LabeledPoint{label: 10., point: arr1(&[2., 2.])},
        ];
        let mut node_lookup = Vec::with_capacity(points.len());
        for point in &points {
            let node = graph.add_node(point.to_owned());
            node_lookup.push(node);
        }
        graph.add_edge(node_lookup[0], node_lookup[1], 0.);
        graph.add_edge(node_lookup[1], node_lookup[2], 0.);
        graph.add_edge(node_lookup[0], node_lookup[3], 0.);
        graph.add_edge(node_lookup[1], node_lookup[4], 0.);
        graph.add_edge(node_lookup[2], node_lookup[5], 0.);
        graph.add_edge(node_lookup[3], node_lookup[4], 0.);
        graph.add_edge(node_lookup[4], node_lookup[5], 0.);
        graph.add_edge(node_lookup[3], node_lookup[6], 0.);
        graph.add_edge(node_lookup[4], node_lookup[7], 0.);
        graph.add_edge(node_lookup[5], node_lookup[8], 0.);
        graph.add_edge(node_lookup[6], node_lookup[7], 0.);
        graph.add_edge(node_lookup[7], node_lookup[8], 0.);
        let mut complex = topology::morse::MorseComplex::from_graph(&mut graph);
        let lifetimes = complex.compute_persistence();
        assert_eq!(lifetimes[&node_lookup[0]], 5.);
        assert_eq!(lifetimes[&node_lookup[1]], 0.);
        assert_eq!(lifetimes[&node_lookup[2]], 1.);
        assert_eq!(lifetimes[&node_lookup[3]], 0.);
        assert_eq!(lifetimes[&node_lookup[4]], 0.);
        assert_eq!(lifetimes[&node_lookup[5]], 0.);
        assert_eq!(lifetimes[&node_lookup[6]], 0.);
        assert_eq!(lifetimes[&node_lookup[7]], 0.);
        assert_eq!(lifetimes[&node_lookup[8]], f64::INFINITY);
    }
}
