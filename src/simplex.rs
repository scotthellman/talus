use std::cmp::Reverse;

fn choose(n: usize, k: usize) -> usize {
    if k > n {
        return 0
    };

    let mut numerator = 1;
    let mut denominator = 1;
    for i in 1..k+1 {
        numerator *= n + 1 - i;
        denominator *= i;
    }
    numerator / denominator
}

fn binary_search_for_cns(n: CNS, d: Dimension, limit: usize) -> CNS {
    let d = usize::from(d);
    let n = usize::from(n);

    let mut upper = limit;
    let mut lower = 0;
    let mut mid;

    while upper > lower {
        mid = (upper + lower) / 2;
        let value = choose(mid, d+1);

        if value <= n {
            lower = mid + 1
        } else {
            upper = mid - 1
        }
    }
    if choose(lower, d+1) > n {  
        lower -= 1
    }
    CNS::from(lower)
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CNS(usize);
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

struct Dimension(usize);
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
struct Simplex {
    // TODO: this is related to the LabeledPoint
    vertices: Vec<usize>,
    value: f64,
}


impl Simplex {
    fn construct_simplex(vertices: &[usize], value: f64) -> Self {
        let mut sorted_vertices: Vec<usize> = vertices.iter()
            .cloned()
            .collect();
        // nb the paper sorts high to low, but we do low to high
        // this lets dimensionality align with index
        sorted_vertices.sort();
        Simplex{vertices: sorted_vertices, value}
    }

    // FIXME: this can be done as a proper conversion (other way can't be)
    fn to_cns(&self) -> CNS {
        let value: usize = self.vertices.iter().enumerate()
            .map(|(d, &v)| choose(v, d+1))
            .sum();
        CNS::from(value)
    }
}


impl CNS {
    fn to_vector(self, dim: Dimension) -> Vec<usize> {
        let dim = usize::from(dim);
        let mut vertices: Vec<usize> = std::iter::repeat(0).take(dim+1).collect();
        let mut n = usize::from(self);
        let mut limit = if n > dim {n} else {dim};

        for d in (0..dim+1).rev() {
            limit = usize::from(binary_search_for_cns(CNS::from(n), Dimension::from(d), limit));
            vertices[d] = limit;
            n -= choose(limit, d+1)
        }
        vertices
    }

    fn to_simplex(self, dim: Dimension, value: f64) -> Simplex {
        Simplex {
            vertices: self.to_vector(dim),
            value
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::collection::hash_set;

    //fn vec_from_length(max_val: usize, max_length: usize) -> impl Strategy<Value = Vec<usize>> {
    //    hash_set(0..max_val, 1..max_length).iter().collect()
    //}

    proptest! {
        #[test]
        // FIXME: need to use bigints + memoize or something to handle reasonable values of n
        fn test_cns_isomorphism_from_int_and_dim(n in 0usize..20, d in 1usize..3, v in -100f64..100.0) {
            let n = CNS::from(n);
            let d = Dimension::from(d);
            prop_assert_eq!(n, n.to_simplex(d, v).to_cns());
        }

        //#[test]
        // FIXME: need to use bigints + memoize or something to handle reasonable values of n
        //fn test_cns_isomorphism_from_simplex(n in 0usize..20, d in 1usize..3, v in -100f64..100.0) {
            //let n = CNS::from(n);
            //let d = Dimension::from(d);
            //prop_assert_eq!(n, n.to_simplex(d, v).to_cns());
        //}
    }
}
