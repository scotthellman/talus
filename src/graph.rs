use ndarray::prelude::*;
use itertools::Itertools;
use petgraph::graph::{Graph, NodeIndex};
use std::f64;

use super::LabeledPoint;

fn pairwise_distance(points: &[LabeledPoint]) -> Array2<f64> {
    let mut pairwise = Array2::zeros((points.len(), points.len()));
    for (i, row) in points.iter().enumerate() {
        for (j, other) in points[i..].iter().enumerate() {
            let j = j+i;
            let distance = if i == j {
                0.
            } else {
                let diff = &row.point - &other.point;
                diff.dot(&diff).sqrt()
            };
            pairwise[[i,j]] = distance;
            pairwise[[j,i]] = distance;
        }
    }
    pairwise
}

pub fn build_knn(points: &[LabeledPoint], k: usize) -> Graph<LabeledPoint, f64, petgraph::Undirected> {
    let mut neighbor_graph = Graph::new_undirected();
    let mut node_lookup = Vec::with_capacity(points.len());
    for point in points {
        let node = neighbor_graph.add_node(point.to_owned());
        node_lookup.push(node);
    }
    let pairwise = pairwise_distance(points);
    for (i, _) in points.iter().enumerate() {
        pairwise.slice(s![i, ..]).into_iter().enumerate()
            .filter(|(j, _)| i != *j)
            .sorted_by(|(_, val), (_, other)| val.partial_cmp(other).unwrap())
            .take(k)
            .for_each(|(j, val)| {
                neighbor_graph.add_edge(node_lookup[i], node_lookup[j], *val);
            });
    }
    neighbor_graph
}

fn find_steepest_neighbor(node: NodeIndex,
                          graph: &Graph<LabeledPoint, f64, petgraph::Undirected>) -> Option<NodeIndex> {
    let this_point = graph.node_weight(node).unwrap();
    let result = graph.neighbors(node)
        .map(|n_idx| (n_idx, graph.node_weight(n_idx).unwrap()))
        .filter(|(_, n)| n.value > this_point.value)
        .map(|(n_idx, n)| (n_idx, n, this_point.grade(&n)))
        .max_by(|a, b| a.2.partial_cmp(&b.2).expect("Nan in the values"));
    match result {
        None => None,
        Some((idx, _, _)) => Some(idx)
    }
}

fn partition_graph_by_steepest_ascent(graph: &Graph<LabeledPoint, f64, petgraph::Undirected>) {
    //FIXME actually implement this
    for node in graph.node_indices() {
        let neighbor = find_steepest_neighbor(node, graph);
        println!("For {:?}, steepest neighbor was {:?}", node, neighbor);
    }
}
