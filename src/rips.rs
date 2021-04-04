use super::simplex::{Simplex, Dimension, CNS, SimplexConverter};
use std::collections::{HashMap, HashSet};
use itertools::Itertools;


// FIXME: notation issue, i say lifetime when birthtime is more accurate

#[derive(Debug, PartialEq)]
struct RichSimplex {
    simplex: CNS,
    dimension: Dimension,
    lifetime: f64
}

impl RichSimplex {
    fn from_vertices(vertices: &[usize], lifetime: f64, converter: &SimplexConverter) -> RichSimplex {
        let simplex = converter.simplex_to_cns(&Simplex::construct_simplex(vertices, lifetime));
        let dimension = Dimension::from(vertices.len() - 1);
        RichSimplex{simplex, dimension, lifetime}
    }
}

fn insert_vertices(vertices: &[usize], lifetime: f64, converter: &SimplexConverter, max_dim: Dimension,
                       neighbors: &HashMap<usize, HashSet<usize>>) -> Vec<RichSimplex> {
    // TODO: throttle this according to max dim
    println!("Finding cofaces of {:?} given these neighbors {:?}", vertices, neighbors);
    let mut simplices: Vec<RichSimplex> = Vec::with_capacity(5); // FIXME: probably a better heuristic for capacity
    // need to do this in a constructive manner
    let mut queue: Vec<Vec<usize>> = vec![vertices.iter().copied().collect()];
    while let Some(current) = queue.pop() {
        if current.len() > usize::from(max_dim)+1 {
            continue;
        }
        let current_simplex = RichSimplex::from_vertices(&current, lifetime, converter);
        let unseen = simplices.iter().all(|s| s != &current_simplex);
        if !unseen {
            continue;
        }
        simplices.push(current_simplex);
        let common_neighbors: HashSet<usize> = current.iter()
            .filter_map(|v| neighbors.get(v))
            .fold(None, |acc, x| {
                match acc {
                    None => Some(x.clone()),
                    Some(candidates) => Some(candidates.intersection(x).copied().collect())
                }
            })
            .unwrap_or_else(HashSet::new);
        let mut new_vertices: Vec<Vec<usize>> = common_neighbors.iter()
            .map(|v| {
                let mut new_vertices: Vec<usize> = current.iter().copied().collect();
                new_vertices.push(*v);
                new_vertices
            })
            .collect();
        queue.append(&mut new_vertices);
    }
    simplices
}


fn rips(distances: Vec<Vec<f64>>, max_dim: Dimension, max_distance: Option<f64>) -> Vec<RichSimplex> {
    // TODO: there's a faster algorithm out there

    // We're going to iterate over ever cell in distances in ascending distance order
    let mut labeled_indices: Vec<(f64, usize, usize)> = distances.iter().enumerate()
        .map(|(i, row)| {
            row.iter().enumerate()
               .skip(i+1)
               .map(move |(j, &val)| {  // NOTE to my rusty-rust self, this is because we need to move i
                   (val, i, j)
               })
        })
        .flatten()
        .collect();
    labeled_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // First we need a converter to do simplex->cns conversion
    let converter = SimplexConverter::construct_for_vertex_count_and_dim(distances.len(), usize::from(max_dim));

    // build 0-dimensional simplices
    let mut simplices: Vec<RichSimplex> = (0..distances.len())
        .map(|i| {
            let simplex = CNS::from(i);
            let dimension = Dimension::from(0);
            let lifetime = 0.0;
            RichSimplex{simplex, dimension, lifetime}
        })
        .collect();

    // General idea: 
    // add one-simplices in ascending distance order
    // a one simplex connects two zero simplices
    // if both of those simplices are connected to other vertices,
    // higher-order simplices are created
    // So we need to be able to quickly track neighbors of each vertex
    // (One could say a graph library could do this for us, but that feels heavyweight)

    let mut neighbor_lookup: HashMap<usize, HashSet<usize>> = simplices.iter()
        .map(|s| {
            (usize::from(s.simplex), HashSet::with_capacity(distances.len()))
        })
        .collect();

    for (distance, row, col) in labeled_indices {
        if let Some(max_dist) = max_distance {
            if distance > max_dist {
                break;
            }
        }
        neighbor_lookup.get_mut(&row).unwrap().insert(col);
        neighbor_lookup.get_mut(&col).unwrap().insert(row);

        //let rich_simplex = RichSimplex::from_vertices(&[col, row], distance, &converter);
        //simplices.push(rich_simplex);

        let cofaces = insert_vertices(&[col, row], distance, &converter, max_dim, &neighbor_lookup);
        println!("cofaces {:?}", cofaces);
        //simplices.extend(find_all_cofaces(&[col, row], distance, &converter, max_dim, &neighbor_lookup));
        simplices.extend(cofaces);
        println!("{:?} simplices - {:?}", simplices.len(), simplices);
    }
    simplices
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::collection::hash_set;

    #[test]
    fn test_rips_small() {
        let dists = vec![vec![0., 1., 1., 2.],
                         vec![1., 0., 2., 1.],
                         vec![1., 2., 0., 1.],
                         vec![2., 1., 1., 0.]];
        // 4 choose 1 = 4
        // 4 choose 2 = 6
        // 4 choose 3 = 4
        // 4 choose 4 = 1
        // Sums to 15
        let complex = rips(dists, Dimension::from(4), None);
        assert_eq!(complex.len(), 15);
    }

    #[test]
    fn test_rips_small_max_dist() {
        let dists = vec![vec![0., 1., 1., 2.],
                         vec![1., 0., 2., 1.],
                         vec![1., 2., 0., 1.],
                         vec![2., 1., 1., 0.]];
        let complex = rips(dists, Dimension::from(4), Some(1.0));
        assert_eq!(complex.len(), 8);
    }

    #[test]
    fn test_rips_small_max_dim() {
        let dists = vec![vec![0., 1., 1., 2.],
                         vec![1., 0., 2., 1.],
                         vec![1., 2., 0., 1.],
                         vec![2., 1., 1., 0.]];
        let complex = rips(dists, Dimension::from(2), None);
        assert_eq!(complex.len(), 14);
    }
}
