use ndarray::prelude::*;
use itertools::Itertools;
use petgraph::graph::{Graph, NodeIndex};
use kdtree::KdTree;
use kdtree::distance::squared_euclidean;
use kdtree::ErrorKind;
use std::f64;

use super::LabeledPoint;

pub fn build_knn(points: &[LabeledPoint], k: usize) -> Graph<LabeledPoint, f64, petgraph::Undirected> {
    let dim = points[0].point.len();
    let mut tree = KdTree::new(dim);
    for (i, point) in points.iter().enumerate() {
        tree.add(&point.point, i).unwrap();
    }
    let mut neighbor_graph = Graph::new_undirected();
    let mut node_lookup = Vec::with_capacity(points.len());
    for point in points {
        let node = neighbor_graph.add_node(point.to_owned());
        node_lookup.push(node);
    }
    for (i, point) in points.iter().enumerate() {
        tree.iter_nearest(&point.point, &squared_euclidean)
            .unwrap()
            .skip(1)  // always returns itself as the first one
            .take(k)
            .for_each(|(dist, &j)| {
                neighbor_graph.update_edge(node_lookup[i], node_lookup[j], dist);
            })
    }
    neighbor_graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_knn() {
        let points = [
            LabeledPoint{id: 0, value: 6., point: vec![0., 0.]},
            LabeledPoint{id: 1, value: 2., point: vec![1., 0.]},
            LabeledPoint{id: 2, value: 3., point: vec![1.5, 0.]},
            LabeledPoint{id: 3, value: 5., point: vec![0., 0.7]},
            LabeledPoint{id: 4, value: 4., point: vec![1., 1.]},
            LabeledPoint{id: 5, value: -5., point: vec![0., 2.]},
            LabeledPoint{id: 6, value: 0., point: vec![2., 3.]}
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
