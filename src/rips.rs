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
    use super::super::binomial::BinomialCoeff;
    use std::cmp;
    use proptest::prelude::*;
    use proptest::collection::vec;

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

    proptest! {
        #[test]
        fn test_rips_correct_count(distances in vec(0.01f64..1000.0, 0..100), max_dim in 0usize..4) {
            // invert the formula for arithmetic sum to get n^2 + n - 2s = 0
            let n = 1 + ((((1 + 8*distances.len()) as f64).sqrt() - 1.0) / 2.0).floor() as usize;

            let mut dists: Vec<Vec<f64>> = (0..n).map(|_| {
                (0..n).map(|_| 0.0).collect()
            }).collect();

            let mut diag = 1;
            let mut offset = 0;
            for distance in distances {
                if diag == n {
                    break;
                }
                dists[diag][offset] = distance;
                dists[offset][diag] = distance;
                offset += 1;
                if offset >= diag {
                    offset = 0;
                    diag += 1
                }
            }
            println!("Distances: {:?}", dists);

            let binomial = BinomialCoeff::construct_for_max_k_and_n(n+1, max_dim+1);

            // expected number is n choose k, 0 <= k <= max_dim
            // and stopping if k > n
            let expected: usize = (0..cmp::min(max_dim+1, n)).map(|k| {
                binomial.binomial(n, k+1)
            }).sum();

            let complex = rips(dists, Dimension::from(max_dim), None);
            prop_assert_eq!(expected, complex.len());
        }
    }
}
