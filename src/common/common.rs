// Version 2: use array
// struct KNNResultSet<const N: usize> {
//     indices: [usize; N],
//     dists: [f64; N], // dists might still need Vec for dynamic resizing
//     capacity: usize,
//     count: usize,
// }
//
// impl<const N: usize> KNNResultSet<N> {
//     fn new(capacity: usize) -> Self {
//         Self {
//             indices: [0; N],  // Initialized to 0
//             dists: [f64::MAX; N],
//             capacity,
//             count: 0,
//         }
//     }
//
//     fn size(&self) -> usize {
//         self.count
//     }
//
//     fn is_full(&self) -> bool {
//         self.count == self.capacity
//     }
//
//     fn add_point(dist: f64, index: usize) {
//
//     }
// }

use std::cmp::PartialEq;
use std::ops::Index;
use kiddo::{ImmutableKdTree, KdTree, SquaredEuclidean};

pub(crate) trait ResultSet<DistType> {
    fn init(&mut self, indices: &mut Vec<usize>, dists: &mut Vec<DistType>);
    fn size(&self) -> usize;
    fn is_full(&self) -> bool;
    fn add_point(&mut self, dist: DistType, index: usize) -> bool;
    fn worst_dist(&self) ->  DistType;
}

// Version 1: used vectors
#[derive(Debug)]
pub(crate) struct KNNResultSet<DistType> {
    pub(crate) indices: Vec<usize>,
    pub(crate) dists: Vec<DistType>,
    capacity: usize,
    count: usize,
}

impl KNNResultSet<f64> {

    // Since size is fixed, we can just offload all the allocation work in the initialization
    // Remove overhead of calling Vec::push()
    #[inline]
    pub(crate) fn new_with_capacity(capacity: usize) -> Self {
        unsafe {
            // Fastest way to allocate without default value that i know of
            let indices = Vec::from_raw_parts(
                std::alloc::alloc(std::alloc::Layout::array::<usize>(capacity).unwrap()) as *mut usize,
                capacity,
                capacity,
            );
            let mut dists = Vec::from_raw_parts(
                std::alloc::alloc(std::alloc::Layout::array::<f64>(capacity).unwrap()) as *mut f64,
                capacity,
                capacity,
            );
            dists[capacity - 1] = f64::MAX;
            Self {
                indices,
                dists,
                capacity,
                count: 0,
            }
        }
    }
}
impl ResultSet<f64> for KNNResultSet<f64> {
    #[inline]
    fn init(&mut self, indices: &mut Vec<usize>, dists: &mut Vec<f64>) {
        self.indices = indices.clone();
        self.dists = dists.clone();
        if self.capacity > 0 {
            self.dists[self.capacity - 1] = f64::MAX;
        }
    }
    #[inline]
    fn size(&self) -> usize {
        self.count
    }
    #[inline]
    fn is_full(&self) -> bool {
        self.count == self.capacity
    }
    #[inline]
    fn add_point(&mut self, dist: f64, index: usize) -> bool {
        let mut i = self.count;
        while i > 0 {
            if self.dists[i - 1] > dist {
                if i < self.capacity {
                    self.dists[i] = self.dists[i - 1];
                    self.indices[i] = self.indices[i - 1];
                }
            } else {
                break;
            }
            i -= 1;
        }
        if i < self.capacity {
            self.dists[i] = dist;
            self.indices[i] = index;
        }
        if self.count < self.capacity {
            self.count += 1;
        }
        true
    }
    #[inline]
    fn worst_dist(&self) -> f64 {
        self.dists[self.capacity - 1]
    }
}
pub(crate) struct RadiusResultSet<DistType> {
    pub(crate) radius: DistType,
    pub(crate) indices_dists: Vec<(usize, DistType)>, // (Index, Distance)
}


impl RadiusResultSet<f64> {
    #[inline]
    pub fn new_with_radius(radius: f64) -> Self {
        Self {
            radius,
            indices_dists: vec![]
        }
    }
    #[inline]
    pub fn clear(&mut self) {
        self.indices_dists.clear();
    }

    #[inline]
    pub fn set_radius_and_clear(&mut self, radius: f64) {
        self.radius = radius;
        self.clear();
    }

    #[inline]
    pub fn worst_item(&self) -> (usize, f64) {
        assert!(self.indices_dists.len() > 0);
        self.indices_dists.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .copied().unwrap() // To return an owned (usize, f64) instead of a reference
    }

}
impl ResultSet<f64> for RadiusResultSet<f64> {
    fn init(&mut self, indices: &mut Vec<usize>, dists: &mut Vec<f64>) {
        // self.indices = indices.clone();
        // self.dists = dists.clone();
        // if self.capacity > 0 {
        //     self.dists[self.capacity - 1] = f64::MAX;
        // }
    }
    #[inline]
    fn size(&self) -> usize {
        self.indices_dists.len()
    }

    #[inline]
    fn is_full(&self) -> bool {
        true
    }

    #[inline]
    fn add_point(&mut self, dist: f64, index: usize) -> bool{
        if dist < self.radius { self.indices_dists.push((index, dist)) };
        true
    }

    #[inline]
    fn worst_dist(&self) -> f64 {
        self.radius
    }
}


// Equivalent to IndexDist_Sorter in C++ nanoflann. If same dist, return the earliest index
// fn sort_by_index_distance<T: Ord, U: PartialOrd>(vec: &mut Vec<(T, U)>) {
//     vec.sort_by(|a, b| {
//         match a.1.partial_cmp(&b.1) {
//             Some(std::cmp::Ordering::Equal) => a.0.cmp(&b.0),
//             Some(ordering) => ordering,
//             None => std::cmp::Ordering::Greater, // Treat NaN as larger
//         }
//     });
// }

pub(crate) struct SearchParams<DistType> {
    checks: usize,
    eps: DistType,
    sorted: bool,
}

impl SearchParams<f64> {
    #[inline]
    pub fn new() -> Self {
        Self {
            checks: 32,
            eps: 0.0,
            sorted: true,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Point<DistType> {
    pub(crate) x: DistType,
    pub(crate) y: DistType,
    pub(crate) z: DistType,
}

impl Index<usize> for Point<f64> {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Invalid index")
        }
    }
}

// Hard code a data source for the KDTree
#[derive(Clone, Debug)]
pub(crate) struct DataSource<DistType> {
    pub(crate) vec: Vec<Point<DistType>>
}

impl DataSource<f64> {
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            vec: Vec::with_capacity(capacity)
        }
    }

    #[inline]
    pub fn add_point(&mut self, x: f64, y: f64, z: f64) {
        self.vec.push(Point { x, y, z });
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.vec.len()
    }

    #[inline]
    pub fn get_point_count(&self) -> usize {
        self.vec.len()
    }

    #[inline]
    pub fn get_point(&self, index: usize, dim: usize) -> f64 {
        match dim {
            0 => self.vec[index].x,
            1 => self.vec[index].y,
            2 => self.vec[index].z,
            _ => panic!("Invalid dimension")
        }
    }

    #[inline]
    pub fn get_squared_distance(&self, point1: &Point<f64>, point2_idx: usize) -> f64 {
        let mut dist = 0.0;
        let point2 = &self.vec[point2_idx];
        for i in 0..3 {
            let diff = point1[i] - point2[i];
            dist += diff * diff;
        }
        dist
    }
}

pub(crate) struct KDTreeSingleIndexParams {
    pub(crate) leaf_max_size: usize,
}

impl KDTreeSingleIndexParams {
    #[inline]
    pub fn new() -> Self {
        Self {
            leaf_max_size: 10
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub(crate) node_type: NodeType,         // Replaces the union
    pub(crate) child1: Option<Box<Node>>,   // Replaces Node* child1
    pub(crate) child2: Option<Box<Node>>,   // Replaces Node* child2
}

#[derive(Clone, Debug, PartialEq)]
pub enum NodeType {
    Leaf {
        left: usize,
        right: usize,
    },
    NonLeaf {
        div_feat: i32,      // Dimension used for subdivision
        div_low: f64,      // The low value for subdivision
        div_high: f64,     // The high value for subdivision
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Interval<DistType> {
    pub(crate) low: DistType,
    pub(crate) high: DistType,
}

#[derive(Clone, Debug)]
pub(crate) struct BoundingBox<DistType> {
    pub(crate) bounds: Vec<Interval<DistType>>,
}

impl BoundingBox<f64> {
    #[inline]
    pub fn new(dim: usize) -> Self {
        let mut bounds = Vec::with_capacity(dim);
        for _ in 0..dim {
            bounds.push(Interval { low: 0.0, high: 0.0 });
        }
        Self { bounds }
    }
}

#[cfg(test)]
mod knnresult_set_tests {
    use approx::relative_eq;
    use super::*;

    #[test]
    fn test_results_new() {
        let sut = KNNResultSet::new_with_capacity(10);
        assert_eq!(sut.indices.len(), 10);
        assert_eq!(sut.dists.len(), 10);
    }

    #[test]
    fn test_add_point() {
        let mut sut = KNNResultSet::new_with_capacity(10);
        sut.add_point(1.0, 1);
        sut.add_point(3.0, 2);
        sut.add_point(2.0, 3);
        sut.add_point(5.0, 4);
        sut.add_point(4.0, 5);
        sut.add_point(7.0, 6);
        sut.add_point(6.0, 7);
        sut.add_point(8.0, 8);
        sut.add_point(9.0, 9);
        sut.add_point(10.0, 10);
        assert_eq!(sut.indices, [1, 3, 2, 5, 4, 7, 6, 8, 9, 10])
    }

    #[test]
    fn test_worst_dist() {
        let mut sut = KNNResultSet::new_with_capacity(10);
        sut.add_point(1.0, 1);
        sut.add_point(2.0, 2);
        sut.add_point(3.0, 3);
        sut.add_point(4.0, 4);
        sut.add_point(5.0, 5);
        sut.add_point(6.0, 6);
        sut.add_point(7.0, 7);
        sut.add_point(8.0, 8);
        sut.add_point(9.0, 9);
        sut.add_point(10.0, 10);
        relative_eq!(sut.worst_dist(),  10.0);
    }
}

#[cfg(test)]
mod radius_result_set_tests {
    use approx::relative_eq;
    use super::*;

    #[test]
    fn test_results_new() {
        let sut = RadiusResultSet::new_with_radius(10.0);
        assert_eq!(sut.indices_dists, []);
        relative_eq!(sut.radius, 10.0);
    }

    #[test]
    fn test_add_point() {
        let mut sut = RadiusResultSet::new_with_radius(5.0);
        sut.add_point(1.0, 1);
        sut.add_point(2.0, 2);
        sut.add_point(3.0, 3);
        sut.add_point(4.0, 4);
        sut.add_point(5.0, 5);
        sut.add_point(6.0, 6);
        relative_eq!(sut.indices_dists[0].1, 1.0);
        relative_eq!(sut.indices_dists[1].1, 2.0);
        relative_eq!(sut.indices_dists[2].1, 3.0);
        relative_eq!(sut.indices_dists[3].1, 4.0);
        assert_eq!(sut.indices_dists.len(), 4);
    }

    // #[test]
    // fn test_sort() {
    //     let mut sut = RadiusResultSet::new_with_radius(5.0);
    //     sut.add_point(2.0, 1);
    //     sut.add_point(1.0, 2);
    //     sut.add_point(2.0, 3);
    //     sut.add_point(3.0, 4);
    //
    //     assert_eq!(sut.indices_dists[0].0, 2);
    //     assert_eq!(sut.indices_dists[1].0, 1);
    //     assert_eq!(sut.indices_dists[2].0, 3);
    //     assert_eq!(sut.indices_dists[3].0, 4);
    //     relative_eq!(sut.indices_dists[0].1, 1.0);
    //     relative_eq!(sut.indices_dists[1].1, 2.0);
    //     relative_eq!(sut.indices_dists[2].1, 2.0);
    //     relative_eq!(sut.indices_dists[3].1, 3.0);
    // }

    #[test]
    fn test_clear() {
        let mut sut = RadiusResultSet::new_with_radius(5.0);
        sut.add_point(1.0, 1);
        sut.add_point(2.0, 2);
        sut.add_point(3.0, 3);
        sut.add_point(4.0, 4);
        sut.clear();
        assert_eq!(sut.indices_dists.len(), 0);
    }

    #[test]
    fn test_radius_and_clear() {
        let mut sut = RadiusResultSet::new_with_radius(5.0);
        sut.add_point(1.0, 1);
        sut.add_point(2.0, 2);
        sut.add_point(3.0, 3);
        sut.add_point(4.0, 4);
        sut.set_radius_and_clear(10.0);
        assert_eq!(sut.indices_dists.len(), 0);
        relative_eq!(sut.radius, 10.0);
    }

    #[test]
    fn test_worst_item() {
        let mut sut = RadiusResultSet::new_with_radius(11.0);
        sut.add_point(1.0, 1);
        sut.add_point(10.0, 2);
        sut.add_point(2.0, 3);
        sut.add_point(5.0, 4);
        sut.add_point(4.0, 5);
        assert_eq!(sut.worst_item().0, 2);
        relative_eq!(sut.worst_item().1, 10.0);
    }
}
