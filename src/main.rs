// DistType = f64
// IndexType = usize

// Version 3: unsafe rust

// Version 2: use array
// struct KNNResultSet<const N: usize> {
//     indices: [usize; N],
//     dists: [f64; N], // dists might still need Vec for dynamic resizing
//     capacity: usize,
//     count: usize,
// }
//
// impl<const N: usize> crate::KNNResultSet<N> {
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

trait ResultSet {
    fn init(&mut self, indices: &mut Vec<usize>, dists: &mut Vec<f64>);
    fn size(&self) -> usize;
    fn is_full(&self) -> bool;
    fn add_point(&mut self, dist: f64, index: usize) -> bool;
    fn worst_dist(&self) -> f64;
}

// Version 1: used vectors
#[derive(Debug)]
struct KNNResultSet {
    indices: Vec<usize>,
    dists: Vec<f64>,
    capacity: usize,
    count: usize,
}

impl KNNResultSet {

    // Since size is fixed, we can just offload all the allocation work in the initialization
    // Remove overhead of calling Vec::push()
    #[inline]
    fn new_with_capacity(capacity: usize) -> Self {
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
impl ResultSet for KNNResultSet {
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
struct RadiusResultSet {
    radius: f64,
    indices_dists: Vec<(usize, f64)>, // (Index, Distance)
}


impl RadiusResultSet {
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
impl ResultSet for RadiusResultSet {
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

struct SearchParams {
    checks: usize,
    eps: f64,
    sorted: bool,
}

impl SearchParams {
    #[inline]
    pub fn new() -> Self {
        Self {
            checks: 32,
            eps: 0.0,
            sorted: true,
        }
    }
}

struct KDTree {
    vind: Vec<usize>,
    leaf_size: usize,
}

#[derive(Clone, Debug)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

impl Index<usize> for Point {
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
struct DataSource {
    vec: Vec<Point>
}

impl DataSource {
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
    pub fn get_squared_distance(&self, point1: &Point, point2_idx: usize) -> f64 {
        let mut dist = 0.0;
        let point2 = &self.vec[point2_idx];
        for i in 0..3 {
            let diff = point1[i] - point2[i];
            dist += diff * diff;
        }
        dist
    }
}

struct KDTreeSingleIndexParams {
    leaf_max_size: usize,
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
pub struct Node {
    node_type: NodeType,         // Replaces the union
    child1: Option<Box<Node>>,   // Replaces Node* child1
    child2: Option<Box<Node>>,   // Replaces Node* child2
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
struct Interval {
    low: f64,
    high: f64,
}

#[derive(Clone, Debug)]
struct BoundingBox {
    bounds: Vec<Interval>,
}

impl BoundingBox {
    #[inline]
    pub fn new(dim: usize) -> Self {
        let mut bounds = Vec::with_capacity(dim);
        for _ in 0..dim {
            bounds.push(Interval { low: 0.0, high: 0.0 });
        }
        Self { bounds }
    }
}

#[derive(Debug)]
struct KDTreeSingleIndex {
    // Indices to points in the dataset
    vind: Vec<usize>,
    leaf_size: usize,
    dataset: DataSource,
    root: Option<Box<Node>>,
    size: usize,
    size_at_index_build: usize,
    dim: usize,
    root_bounding_box: BoundingBox,
}


impl KDTreeSingleIndex {
    #[inline]
    pub fn new(dataset: DataSource, params: KDTreeSingleIndexParams) -> Self {
        let dim = 3;
        let size = dataset.size();
        let mut vind = Vec::with_capacity(size);
        unsafe {
            vind.set_len(size);
            for i in 0..size {
                vind[i] = i;
            }
        }
        for i in 0..size {
            vind.push(i);
        }
        let root = None;
        let size_at_index_build = size;
        let mut tree = Self {
            vind,
            leaf_size: params.leaf_max_size,
            dataset,
            root,
            size,
            size_at_index_build,
            dim,
            root_bounding_box: BoundingBox::new(dim),
        };
        tree
    }

    #[inline]
    pub fn build_index(&mut self) {
        self.compute_bounding_box();
        let mut bounding_box = self.root_bounding_box.clone();
        self.root = Some(Box::new(self.divide_tree(0, self.size, &mut bounding_box)));
    }


    #[inline]
    pub fn compute_bounding_box(&mut self) {
        let size = self.size;
        for i in 0..self.dim {
            self.root_bounding_box.bounds[i].low = self.dataset.get_point(0, i);
            self.root_bounding_box.bounds[i].high = self.dataset.get_point(0, i);
        }
        for k in 1..size {
            for i in 0..self.dim {
                let val = self.dataset.get_point(k, i);
                if val < self.root_bounding_box.bounds[i].low {
                    self.root_bounding_box.bounds[i].low = val;
                }
                if val > self.root_bounding_box.bounds[i].high {
                    self.root_bounding_box.bounds[i].high = val;
                }
            }
        }
    }

    #[inline]
    pub fn divide_tree(&mut self, left: usize, right: usize, bounding_box: &mut BoundingBox) -> Node {
        let node;
        if (right - left) <= self.leaf_size {
            // Explicitly give the node a Leaf type
            node = Node {
                node_type: NodeType::Leaf { left, right },
                child1: None,
                child2: None,
            };
            for i in 0..self.dim {
                let val = self.dataset.get_point(self.vind[left], i);
                bounding_box.bounds[i].low = val;
                bounding_box.bounds[i].high = val;
            }
            for k in (left + 1)..right {
                for i in 0..self.dim {
                    let val = self.dataset.get_point(self.vind[k], i);
                    if val < bounding_box.bounds[i].low {
                        bounding_box.bounds[i].low = val;
                    }
                    if val > bounding_box.bounds[i].high {
                        bounding_box.bounds[i].high = val;
                    }
                }
            }
        } else {
            let mut mid = 0;
            let mut cut_feat: usize = 0;
            let mut cut_val = 0.0;
            self.middle_split(left, right - left, &mut mid, &mut cut_feat, &mut cut_val, bounding_box);

            let mut left_bounding_box = bounding_box.clone();
            left_bounding_box.bounds[cut_feat].high = cut_val;
            let child1 = Some(Box::new(self.divide_tree(left, left + mid, &mut left_bounding_box)));

            let mut right_bounding_box = bounding_box.clone();
            right_bounding_box.bounds[cut_feat].low = cut_val;
            let child2 = Some(Box::new(self.divide_tree(left + mid, right, &mut right_bounding_box)));

            for i in 0..self.dim {
                bounding_box.bounds[i].low = left_bounding_box.bounds[i].low.min(right_bounding_box.bounds[i].low);
                bounding_box.bounds[i].high = left_bounding_box.bounds[i].high.max(right_bounding_box.bounds[i].high);
            }
            node = Node {
                node_type: NodeType::NonLeaf {
                    div_feat: cut_feat as i32,
                    div_low: left_bounding_box.bounds[cut_feat].high,
                    div_high: right_bounding_box.bounds[cut_feat].low },
                child1,
                child2,
            };
        }
        node
    }

    #[inline]
    fn middle_split(&mut self, ind: usize, count: usize, index: &mut usize,
                    cut_feat: &mut usize, cut_val: &mut f64, bounding_box: &BoundingBox) {
        let eps = 1e-5;
        let mut max_span = bounding_box.bounds[0].high - bounding_box.bounds[0].low;
        for i in 1..self.dim {
            let span = bounding_box.bounds[i].high - bounding_box.bounds[i].low;
            if span > max_span {
                max_span = span;
            }
        }
        let mut max_spread = -1.0;
        *cut_feat = 0;
        let mut min_element = 0.0;
        let mut max_element = 0.0;
        for i in 0..self.dim {
            let span = bounding_box.bounds[i].high - bounding_box.bounds[i].low;
            if span > (1.0 - eps) * max_span {
                let mut min_element_: f64 = 0.0;
                let mut max_element_: f64 = 0.0;
                self.compute_min_max(ind, count, i, &mut min_element_, &mut max_element_);
                let spread = max_element_ - min_element_;
                if spread > max_spread {
                    max_spread = spread;
                    *cut_feat = i;
                    min_element = min_element_;
                    max_element = max_element_;
                }
            }
        }
        let split_val = 0.5 * (bounding_box.bounds[*cut_feat].low + bounding_box.bounds[*cut_feat].high);

        if split_val < min_element {
            *cut_val = min_element;
        } else if split_val > max_element {
            *cut_val = max_element;
        } else {
            *cut_val = split_val;
        }

        // lim1 is index of last points that are less than cut_val
        // lim2 is index of last points that are equal to cut_val
        let mut lim1: usize = 0;
        let mut lim2: usize = 0;
        self.plane_split(ind, count, *cut_feat, cut_val, &mut lim1, &mut lim2);
        // Balancing purpose
        if lim1 > count / 2 { *index = lim1; }
        else if lim2 < count / 2 { *index = lim2; }
        else { *index = count / 2; }
    }

    #[inline]
    fn compute_min_max(&self, ind: usize, count: usize, cut_feat: usize, min_element: &mut f64, max_element: &mut f64) {
        *min_element = self.dataset.get_point(self.vind[ind], cut_feat);
        *max_element = *min_element;
        for i in 1..count {
            let val = self.dataset.get_point(self.vind[ind + i], cut_feat);
            if val < *min_element {
                *min_element = val;
            }
            if val > *max_element {
                *max_element = val;
            }
        }
    }

    /**
    *  Subdivide the list of points by a plane perpendicular on axe corresponding
    *  to the 'cutfeat' dimension at 'cutval' position.
    *
    *  On return:
    *  dataset[ind[0..lim1-1]][cutfeat]<cutval
    *  dataset[ind[lim1..lim2-1]][cutfeat]==cutval
    *  dataset[ind[lim2..count]][cutfeat]>cutval
    */
    fn plane_split(&mut self, ind: usize, count: usize, cut_feat: usize, cut_val: &mut f64, lim1: &mut usize, lim2: &mut usize) {
        let mut left = 0;
        let mut right = count - 1;
        // This is a variation of the Dutch National Flag problem
        loop {
            while left <= right && self.dataset.get_point(self.vind[ind + left], cut_feat) < *cut_val {
                left += 1;
            }
            while right != 0 && left <= right && self.dataset.get_point(self.vind[ind + right], cut_feat) >= *cut_val {
                right -= 1;
            }
            if left > right || right == 0 {
                break;
            }
            let temp = self.vind[ind + left];
            self.vind[ind + left] = self.vind[ind + right];
            self.vind[ind + right] = temp;
            left += 1;
            right -= 1;
        }
        // By this point, all points less than cut_val are on the left side of
        *lim1 = left;
        right = count - 1;
        loop {
            while left <= right && self.dataset.get_point(self.vind[ind + left], cut_feat) <= *cut_val {
                left += 1;
            }
            while right != 0 && left <= right && self.dataset.get_point(self.vind[ind + right], cut_feat) > *cut_val {
                right -= 1;
            }
            if left > right || right == 0 {
                break;
            }
            let temp = self.vind[ind + left];
            self.vind[ind + left] = self.vind[ind + right];
            self.vind[ind + right] = temp;
            left += 1;
            right -= 1;
        }
        *lim2 = left;
    }

    // Squared Euclidean
    #[inline]
    fn accum_dist(a: f64, b: f64) -> f64 {
        let diff = a - b;
        diff * diff
    }

    // Compute how far the point is from the bounding box
    #[inline]
    fn compute_initial_distance(&self, point: &Point, dists: &mut Vec<f64>) -> f64 {
        let mut dist_square: f64 = 0.0;
        for i in 0..self.dim {
            if point[i] < self.root_bounding_box.bounds[i].low {
                dists[i] = Self::accum_dist(point[i], self.root_bounding_box.bounds[i].low);
                dist_square += dists[i];
            }
            if point[i] > self.root_bounding_box.bounds[i].high {
                dists[i] = Self::accum_dist(point[i], self.root_bounding_box.bounds[i].high);
                dist_square += dists[i];
            }
        }
        dist_square
    }

    #[inline]
    fn search_level(&self, result: &mut dyn ResultSet, point: &Point,
                    node: &Node, mut min_dists_square: f64, dists: &mut Vec<f64>, eps_error: f64) -> bool {
        if matches!(node.node_type, NodeType::Leaf { .. }) {
            let worst_dist = result.worst_dist();
            let NodeType::Leaf { left, right } = node.node_type else { return false; };
            for i in left..right {
                let index = self.vind[i];
                let dist_square = self.dataset.get_squared_distance(point, index);
                if dist_square < worst_dist {
                    if !result.add_point(dist_square, index) {
                        return false;
                    }
                }
            }
            return true;
        }

        let NodeType::NonLeaf { div_feat, div_low, div_high } = node.node_type else { return false; };
        let index = div_feat as usize;
        let val = point[index];
        let diff1 = val - div_low;
        let diff2 = val - div_high;

        let (best_child, other_child, cut_dists) =
            if diff1 + diff2 < 0.0 {
                (&node.child1, &node.child2, Self::accum_dist(val, div_high))
            } else {
                (&node.child2, &node.child1, Self::accum_dist(val, div_low))
            };

        if !self.search_level(result, point, best_child.as_ref().unwrap(), min_dists_square, dists, eps_error) {
            return false;
        }
        let dist = dists[index];
        min_dists_square += cut_dists - dist;
        dists[index] = cut_dists;
        if min_dists_square * eps_error <= result.worst_dist() {
            if !self.search_level(result, point, other_child.as_ref().unwrap(), min_dists_square, dists, eps_error) {
                return false;
            }
        }
        dists[index] = dist;
        true
    }
    fn knn_search(&self, point: &Point, num_closest: usize, result: &mut dyn ResultSet)  {
        let mut dists = vec![0.0; self.dim];
        let eps_error = 1.0 + 1e-5;
        let min_dists_square = self.compute_initial_distance(point, &mut dists);
        self.search_level(result, point, self.root.as_ref().unwrap(), min_dists_square, &mut dists, eps_error);
    }
    fn dataset_get(&self, index: usize, dim: usize) -> f64 {
        self.dataset.get_point(index, dim)
    }
}

fn generate_random_point_clouds_of_size(size: usize) -> DataSource {
    let mut dataset = DataSource::with_capacity(size);
    for _ in 0..size {
        dataset.add_point(rand::random(), rand::random(), rand::random());
    }
    dataset
}

fn generate_point_clouds_of_size(size: usize) -> DataSource {
    let mut dataset = DataSource::with_capacity(size);
    for i in 0..(size / 2) {
        dataset.add_point(i as f64, i as f64, i as f64);
        dataset.add_point((i + size / 2) as f64, (i + size / 2) as f64, (i + size / 2) as f64);
    }
    dataset
}

fn generate_random_points_to_test(size: usize) -> Vec<Point> {
    let mut points = Vec::with_capacity(size);
    for _ in 0..size {
        points.push(Point { x: rand::random(), y: rand::random(), z: rand::random() });
    }
    points
}

fn generate_random_vector_to_test(size: usize) -> Vec<[f64; 3]> {
    let mut points = Vec::with_capacity(size);
    for _ in 0..size {
        points.push([rand::random(), rand::random(), rand::random()]);
    }
    points
}

fn main() {
    let size = 800000;

    let points_to_test: Vec<Point> = generate_random_points_to_test(size);
    let dataset = generate_random_point_clouds_of_size(size);
    let entries: Vec<[f64; 3]> = dataset.vec.iter().map(|point| [point.x, point.y, point.z]).collect();

    // Warm-up
    let mut kdtree = KDTreeSingleIndex::new(dataset.clone(), KDTreeSingleIndexParams::new());
    kdtree.build_index();
    for point in &points_to_test {
        let mut result = KNNResultSet::new_with_capacity(10);
        kdtree.knn_search(point, 10, &mut result);
    }

    let mut kiddo: kiddo::ImmutableKdTree<_, 3> = (&*entries).into();
    for point in &points_to_test {
        kiddo.nearest_n::<SquaredEuclidean>(&[point.x, point.y, point.z], 10);
    }

    let kd_tree = kd_tree::KdTree::build_by_ordered_float(entries.clone());
    for point in &points_to_test {
        let _result = kd_tree.nearests(&[point.x, point.y, point.z], 10);
    }

    // Test KDTreeSingleIndex
    let mut params = KDTreeSingleIndexParams::new();
    params.leaf_max_size = 10;
    let mut kdtree = KDTreeSingleIndex::new(dataset.clone(), params);
    kdtree.build_index();
    let mut result = KNNResultSet::new_with_capacity(10);
    println!("Querying {} points for 10 nearest neighbours using nanofrann", size);
    let start = std::time::Instant::now();
    for point in points_to_test.iter() {
        kdtree.knn_search(point, 10, &mut result);
    }
    let duration = start.elapsed();
    println!("Time taken: {:?}", duration);

    // Test Kiddo
    let kiddo: kiddo::ImmutableKdTree<_, 3> = (&*entries).into();
    println!("Querying {} points for 10 nearest neighbours using kiddo", size);
    let start_time = std::time::Instant::now();
    for point in points_to_test.iter() {
        kiddo.nearest_n::<SquaredEuclidean>(&[point.x, point.y, point.z], 10);
    }
    let elapsed_time = start_time.elapsed();
    println!("Time taken: {:?}", elapsed_time);

    // Test kd_tree
    let kd_tree = kd_tree::KdTree::build_by_ordered_float(entries);
    println!("Querying {} points for 10 nearest neighbours using kd_tree", size);
    let start_time = std::time::Instant::now();
    for point in points_to_test.iter() {
        let result = kd_tree.nearests(&[point.x, point.y, point.z], 10);
    }
    let elapsed_time = start_time.elapsed();
    println!("Time taken: {:?}", elapsed_time);
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

#[cfg(test)]
mod kd_tree_test {
    use approx::{assert_relative_eq};
    use super::*;

    #[test]
    fn test_build_index() {
        let mut dataset = DataSource::with_capacity(10);
        dataset.add_point(1.0, 2.0, 3.0);
        dataset.add_point(4.0, 5.0, 6.0);
        dataset.add_point(7.0, 8.0, 9.0);
        dataset.add_point(2.0, 3.0, 4.0);
        dataset.add_point(5.0, 6.0, 7.0);
        let mut params = KDTreeSingleIndexParams::new();
        params.leaf_max_size = 2;
        // Initialize KDTreeSingleIndex
        let mut kdtree = KDTreeSingleIndex::new(dataset, params);

        // Build the KD-tree index
        kdtree.build_index();

        // Check that the root node exists
        assert!(kdtree.root.is_some(), "Root node should be initialized");

        // Check that the KD-tree structure is built correctly
        if let Some(root) = &kdtree.root {
            // Check if the root is a non-leaf node (since the dataset is larger than leaf size)
            match &root.node_type {
                NodeType::NonLeaf { div_feat, div_low, div_high } => {
                    // Verify the division feature and division values
                    assert!(*div_feat >= 0 && *div_feat < 3, "Division feature should be within valid range");
                    assert!(div_low < div_high, "Division low should be less than division high");
                }
                NodeType::Leaf { .. } => panic!("Root should not be a leaf node"),
            }

            // Check that the children nodes are initialized correctly
            assert!(root.child1.is_some(), "Child 1 should be initialized");
            assert!(root.child2.is_some(), "Child 2 should be initialized");
        }

        // Verify bounding box has been computed correctly
        let expected_bounding_box = BoundingBox {
            bounds: vec![
                Interval { low: 1.0, high: 7.0 }, // For x-dimension
                Interval { low: 2.0, high: 8.0 }, // For y-dimension
                Interval { low: 3.0, high: 9.0 }, // For z-dimension
            ],
        };
        for i in 0..3 {
            assert_relative_eq!(kdtree.root_bounding_box.bounds[i].low, expected_bounding_box.bounds[i].low);
            assert_relative_eq!(kdtree.root_bounding_box.bounds[i].high, expected_bounding_box.bounds[i].high);
        }

        println!("{:?}", kdtree);
    }

    #[test]
    fn test_knn_search() {
        let mut dataset = DataSource::with_capacity(10);
        dataset.add_point(0.0, 0.0, 0.0);
        dataset.add_point(1.0, 1.0, 1.0);
        dataset.add_point(2.0, 0.0, 0.0);
        dataset.add_point(0.0, 2.0, 0.0);
        dataset.add_point(0.0, 0.0, 2.0);
        dataset.add_point(2.0, 2.0, 2.0);
        dataset.add_point(0.0, 0.0, 2.0);
        dataset.add_point(0.0, 2.0, 2.0);
        dataset.add_point(0.1, 0.1, 0.1);
        let mut params = KDTreeSingleIndexParams::new();
        params.leaf_max_size = 2;
        // Initialize KDTreeSingleIndex
        let mut kdtree = KDTreeSingleIndex::new(dataset, params);

        // Build the KD-tree index
        kdtree.build_index();

        // Check that the root node exists
        assert!(kdtree.root.is_some(), "Root node should be initialized");
        let point_to_test = Point { x: 0.0, y: 0.0, z: 0.0 };
        let mut result = KNNResultSet::new_with_capacity(3);
        let search_params = SearchParams::new();
        kdtree.knn_search(&point_to_test, 3, &mut result);
        println!("{:?}", result);
    }
}
