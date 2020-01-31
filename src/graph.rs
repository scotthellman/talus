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
                neighbor_graph.update_edge(node_lookup[i], node_lookup[j], *val);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_knn() {
        let points = [
            LabeledPoint{id: 0, value: 6., point: arr1(&[0., 0.])},
            LabeledPoint{id: 1, value: 2., point: arr1(&[1., 0.])},
            LabeledPoint{id: 2, value: 3., point: arr1(&[1.5, 0.])},
            LabeledPoint{id: 3, value: 5., point: arr1(&[0., 0.7])},
            LabeledPoint{id: 4, value: 4., point: arr1(&[1., 1.])},
            LabeledPoint{id: 5, value: -5., point: arr1(&[0., 2.])},
            LabeledPoint{id: 6, value: 0., point: arr1(&[2., 3.])}
        ];
        let mut expected_adjacencies = HashMap::with_capacity(7);
        expected_adjacencies.insert(0, vec![1, 3]);
        expected_adjacencies.insert(1, vec![0, 2, 4]);
        expected_adjacencies.insert(2, vec![1, 4]);
        expected_adjacencies.insert(3, vec![0, 4, 5]);
        expected_adjacencies.insert(4, vec![1, 2, 3, 5, 6]);
        expected_adjacencies.insert(5, vec![3, 4, 6]);
        expected_adjacencies.insert(6, vec![4, 5]);

        let g = build_knn(&points, 2);
        for node in g.node_indices() {
            let id = g.node_weight(node).unwrap().id;
            let adj_ids: HashSet<i64> = g.neighbors(node)
                .map(|n| g.node_weight(n).unwrap().id)
                .collect();
            let expected = expected_adjacencies.get(&id).unwrap();
            assert_eq!(expected.len(), adj_ids.len());
            for exp in expected {
                assert!(adj_ids.contains(exp));
            }

        }
    }
}
