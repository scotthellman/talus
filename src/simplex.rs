use std::cmp::Reverse;
use super::binomial::BinomialCoeff;

pub struct SimplexConverter {
    pub binomial: BinomialCoeff,
    vertex_count: usize
}

impl SimplexConverter {
    pub fn construct_for_vertex_count_and_dim(vertex_count: usize, dim: usize) -> Self {
        let binomial = BinomialCoeff::construct_for_max_k_and_n(vertex_count+1, dim+1);
        SimplexConverter{binomial, vertex_count}
    }

    pub fn simplex_to_cns(&self, simplex: &Simplex) -> CNS {
        let value: usize = simplex.vertices.iter().enumerate()
            .map(|(d, &v)| self.binomial.binomial(v, d+1))
            .sum();
        CNS::from(value)
    }

    pub fn cns_to_vector(&self, n: CNS, dim: Dimension) -> Vec<usize> {
        let dim = usize::from(dim);
        let mut vertices: Vec<usize> = std::iter::repeat(0).take(dim+1).collect();
        let mut n = usize::from(n);
        let mut limit = if n > dim {n} else {dim};

        for d in (0..dim+1).rev() {
            limit = usize::from(self.binary_search_for_cns(CNS::from(n), Dimension::from(d), limit));
            vertices[d] = limit;
            n -= self.binomial.binomial(limit, d+1)
        }
        vertices
    }

    pub fn cns_to_simplex(&self, n: CNS, dim: Dimension, value: f64) -> Simplex {
        Simplex {
            vertices: self.cns_to_vector(n, dim),
            value
        }
    }

    fn binary_search_for_cns(&self, n: CNS, d: Dimension, limit: usize) -> CNS {
        // FIXME: there's an issue here where if mid is too big for self.binomial
        // then we always return limit
        let d = usize::from(d);
        let n = usize::from(n);

        let mut upper = limit;
        let mut lower = 0;
        let mut mid;

        while upper > lower {
            mid = (upper + lower) / 2;
            let value = self.binomial.binomial(mid, d+1);

            if value <= n {
                lower = mid + 1
            } else {
                upper = mid - 1
            }
        }
        if self.binomial.binomial(lower, d+1) > n {  
            lower -= 1
        }
        CNS::from(lower)
    }
}




#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct CNS(usize);
impl From<usize> for CNS {
    fn from(val: usize) -> CNS {
        CNS(val)
    }
}

impl From<CNS> for usize {
   fn from(n: CNS) -> usize {
       n.0
   }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Dimension(usize);
impl From<usize> for Dimension {
    fn from(val: usize) -> Dimension {
        Dimension(val)
    }
}

impl From<Dimension> for usize {
   fn from(d: Dimension) -> usize {
       d.0
   }
}


#[derive(Debug, PartialEq)]
pub struct Simplex {
    // TODO: this is related to the LabeledPoint
    vertices: Vec<usize>,
    value: f64,
}


impl Simplex {
    pub fn construct_simplex(vertices: &[usize], value: f64) -> Self {
        let mut sorted_vertices: Vec<usize> = vertices.iter()
            .cloned()
            .collect();
        // nb the paper sorts high to low, but we do low to high
        // this lets dimensionality align with index
        sorted_vertices.sort();
        Simplex{vertices: sorted_vertices, value}
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::collection::hash_set;

    #[test]
    fn test_cns_paper_case_38() {
        let converter = SimplexConverter::construct_for_vertex_count_and_dim(20, 10);
        let n = CNS::from(38);
        let d = Dimension::from(2);
        let simplex = converter.cns_to_simplex(n, d, 2.0);
        assert_eq!(simplex.vertices, vec![0, 3, 7]);
        let new_n = converter.simplex_to_cns(&simplex);
        assert_eq!(new_n ,n);
    }

    #[test]
    fn test_cns_paper_case_0() {
        let converter = SimplexConverter::construct_for_vertex_count_and_dim(10, 10);
        let n = CNS::from(0);
        let d = Dimension::from(2);
        let simplex = converter.cns_to_simplex(n, d, 2.0);
        assert_eq!(simplex.vertices, vec![0, 1, 2]);
        let new_n = converter.simplex_to_cns(&simplex);
        assert_eq!(new_n ,n);
    }

    proptest! {
        #[test]
        // FIXME: need to use bigints + memoize or something to handle reasonable values of n
        fn test_cns_isomorphism_from_int_and_dim(n in 0usize..20, d in 1usize..3, v in -100f64..100.0) {
            let converter = SimplexConverter::construct_for_vertex_count_and_dim(n, d);
            let n = CNS::from(n);
            let d = Dimension::from(d);
            let simplex = converter.cns_to_simplex(n, d, v);
            prop_assert_eq!(n, converter.simplex_to_cns(&simplex));
        }
    }
}
