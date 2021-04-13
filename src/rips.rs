use super::simplex::{Simplex, Dimension, CNS, SimplexConverter};
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use super::binomial::BinomialCoeff;
use itertools::Itertools;


// FIXME: notation issue, i say lifetime when birthtime is more accurate
// ripser paper uses "diameter" but i think that's confusing without context

#[derive(Debug, PartialEq, Clone)]
pub struct RichSimplex {
    pub lifetime: f64,
    pub dimension: Dimension,
    pub simplex: CNS
}

impl PartialOrd for RichSimplex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.lifetime == other.lifetime {
            if self.dimension == other.dimension {
                return other.simplex.partial_cmp(&self.simplex)
            }
            return self.dimension.partial_cmp(&other.dimension)
        }
        self.lifetime.partial_cmp(&other.lifetime)
    }
}

impl RichSimplex {
    pub fn from_vertices(vertices: &[usize], lifetime: f64, converter: &SimplexConverter) -> RichSimplex {
        let simplex = converter.simplex_to_cns(&Simplex::construct_simplex(vertices, lifetime));
        let dimension = Dimension::from(vertices.len() - 1);
        RichSimplex{simplex, dimension, lifetime}
    }

    pub fn cofacets<'a>(&self, converter: &'a SimplexConverter) -> CofacetIterator<'a> {
        let vertices = converter.cns_to_vector(self.simplex, self.dimension);
        CofacetIterator{
            dimension: self.dimension,
            binomial: &converter.binomial,
            left_sum: 0,
            right_sum: usize::from(self.simplex) as i64,
            k: usize::from(self.dimension),
            j: converter.vertex_count - 1,
            vertices,
        }
    }
}

pub struct CofacetIterator<'a> {
    dimension: Dimension,
    binomial: &'a BinomialCoeff,
    vertices: Vec<usize>,
    left_sum: i64,
    right_sum: i64,
    k: usize,
    j: usize
}

impl<'a> Iterator for CofacetIterator<'a> {
    // we will be counting with usize
    type Item = RichSimplex;

    // next() is the only required method
    fn next(&mut self) -> Option<Self::Item> {
        let dim = usize::from(self.dimension);
        loop {
            while self.j < self.vertices[self.k] {
                self.left_sum += self.binomial.binomial(self.vertices[self.k], self.k+2) as i64;
                self.right_sum -= self.binomial.binomial(self.vertices[self.k], self.k) as i64;
                if self.k == 0{
                    break;
                }
                self.k -= 1;
            }
            if self.j == 0 {
                break None;
            }
            self.j -= 1;
            if !(self.j+1 == self.vertices[self.k]) {
                let cns =  self.left_sum + self.right_sum + self.binomial.binomial(self.j+1, self.k+2) as i64;
                let cns = CNS::from(cns as usize);
                break Some(RichSimplex{
                    simplex: cns,
                    dimension: Dimension::from(dim+1),
                    lifetime: f64::NAN //FIXME this is not a Good way to indicate lifetime isn't set
                })
            }
        }
    }
}

fn insert_vertices(vertices: &[usize], lifetime: f64, converter: &SimplexConverter, max_dim: Dimension,
                       neighbors: &HashMap<usize, HashSet<usize>>) -> Vec<RichSimplex> {
    // TODO: throttle this according to max dim
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
        //simplices.extend(find_all_cofaces(&[col, row], distance, &converter, max_dim, &neighbor_lookup));
        simplices.extend(cofaces);
    }

    // Ripser assumes the complex is ordered by lifetime, then dimension, then by CNS
    simplices.sort_by(|a, b| a.partial_cmp(b).unwrap()); // TODO: plausibly we can handle nan more gracefully
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
        // Expected simplices, in order:
        // We need rips to be reverse lexicographic for identical lifetime and dim, so we can assume the
        // full ordering here
        // NOTE it's reverse reverse lexicographic, as we go backwards through the vertices
        // 3, 2, 1, 0, 23, 13, 02, 01, 12, 03, 123, 023, 013, 012, 0123
        let converter = SimplexConverter::construct_for_vertex_count_and_dim(dists.len(), 4);
        let expected = vec![
            RichSimplex::from_vertices(&[3], 0., &converter),
            RichSimplex::from_vertices(&[2], 0., &converter),
            RichSimplex::from_vertices(&[1], 0., &converter),
            RichSimplex::from_vertices(&[0], 0., &converter),
            RichSimplex::from_vertices(&[2, 3], 1., &converter),
            RichSimplex::from_vertices(&[1, 3], 1., &converter),
            RichSimplex::from_vertices(&[0, 2], 1., &converter),
            RichSimplex::from_vertices(&[0, 1], 1., &converter),
            RichSimplex::from_vertices(&[0, 3], 2., &converter),
            RichSimplex::from_vertices(&[1, 2], 2., &converter),
            RichSimplex::from_vertices(&[1, 2, 3], 2., &converter),
            RichSimplex::from_vertices(&[0, 2, 3], 2., &converter),
            RichSimplex::from_vertices(&[0, 1, 3], 2., &converter),
            RichSimplex::from_vertices(&[0, 1, 2], 2., &converter),
            RichSimplex::from_vertices(&[0, 1, 2, 3], 2., &converter),
        ];
        let complex = rips(dists, Dimension::from(4), None);
        assert_eq!(expected, complex);
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

    #[test]
    fn test_cofacet_iterator() {
        let converter = SimplexConverter::construct_for_vertex_count_and_dim(7, 4);
        let simplex = RichSimplex::from_vertices(&[0, 3, 5], 1., &converter);
        let expected = [28, 12, 7 ,6];
        for (i, coface) in simplex.cofacets(&converter).enumerate() {
            assert_eq!(CNS::from(expected[i]), coface.simplex);
        }
    }

    proptest! {
        #[test]
        fn test_rips_correct_counts(distances in vec(0.01f64..1000.0, 0..100), max_dim in 0usize..4) {
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

            let binomial = BinomialCoeff::construct_for_max_k_and_n(n+1, max_dim+1);
            let complex = rips(dists, Dimension::from(max_dim), None);

            // expected number is n choose k, 0 <= k <= max_dim
            // and stopping if k > n
            for k in 0..cmp::min(max_dim+1, n){
                let simplex_count = complex.iter()
                    .filter(|s| s.dimension == Dimension::from(k))
                    .count();
                let expected = binomial.binomial(n, k+1);
                prop_assert_eq!(expected, simplex_count);
            }
        }
    }
}
