struct BinomialCoeff {
    coeffs:  Vec<Vec<usize>>
}

impl BinomialCoeff {
    fn construct_for_max_k_and_n(n: usize, k: usize) -> Self {
        let mut coeffs: Vec<Vec<usize>> = Vec::with_capacity(n);
        for i in 0..n {
            let row = (0..k).map(|j| {
                if j == 0 || j >= i {
                    1
                } else {
                    coeffs[i-1][j-1] + coeffs[i-1][j]
                }
            }).collect();
            coeffs.push(row);
        }
        BinomialCoeff{coeffs}
    }

    fn binomial(&self, n: usize, k: usize) -> usize {
        // wonder how hard it is to get away from the panic
        self.coeffs[n-1][k]
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::collection::hash_set;

    #[test]
    fn test_simple_binomial() {
        let coeffs = BinomialCoeff::construct_for_max_k_and_n(5, 4);
        assert_eq!(coeffs.binomial(1,1), 1);
        assert_eq!(coeffs.binomial(2,1), 1);
        assert_eq!(coeffs.binomial(2,2), 1);
        assert_eq!(coeffs.binomial(4,2), 3);
    }

    //fn vec_from_length(max_val: usize, max_length: usize) -> impl Strategy<Value = Vec<usize>> {
    //    hash_set(0..max_val, 1..max_length).iter().collect()
    //}

    /*proptest! {
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
    */
}
