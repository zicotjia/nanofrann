mod common;
mod tree;

use kiddo::SquaredEuclidean;
use num_traits::{FromPrimitive, Num};
use rand::distr::{Distribution, Standard};
use tree::nanofrann_float::*;
use common::common::*;

use rand::random;
use crate::common::min_max::MinMax;
use crate::tree::nanofrann_flat::KDTreeSingleIndexFlat;

fn generate_random_point_clouds_of_size<T>(size: usize) -> DataSource<T>
where
    T: Num + Copy + MinMax + PartialOrd + FromPrimitive + std::ops::AddAssign, Standard: Distribution<T>
{
    let mut dataset = DataSource::<T>::with_capacity(size);
    for _ in 0..size {
        dataset.add_point(random::<T>(), random::<T>(), random::<T>());
    }
    dataset
}

fn generate_point_clouds_of_size<T>(size: usize) -> DataSource<T>
where
    T: Num + Copy + MinMax + PartialOrd + FromPrimitive + std::ops::AddAssign, Standard: Distribution<T>
{
    let mut dataset = DataSource::<T>::with_capacity(size);
    for i in 0..(size / 2) {
        let point1 = T::from_usize(i).unwrap_or(T::ZERO);
        let point2 = T::from_usize(i + size / 2).unwrap_or(T::ZERO);
        dataset.add_point(point1, point1, point1);
        dataset.add_point(point2, point2, point2);
    }
    dataset
}

fn generate_sorted_point_clouds_of_size<T>(size: usize) -> DataSource<T>
where
    T: Num + Copy + MinMax + PartialOrd + FromPrimitive + std::ops::AddAssign, Standard: Distribution<T>
{
    let mut dataset = DataSource::<T>::with_capacity(size);
    for i in 0..size {
        let point1 = T::from_usize(i).unwrap_or(T::ZERO);
        dataset.add_point(point1, point1, point1);
    }
    dataset
}

fn generate_random_points_to_test<T>(size: usize) -> Vec<Point<T>>
where
    T: Num + Copy + MinMax + PartialOrd + FromPrimitive + std::ops::AddAssign, Standard: Distribution<T>
{
    let mut points = Vec::with_capacity(size);
    for _ in 0..size {
        points.push(Point {
            x: random::<T>(),
            y: random::<T>(),
            z: random::<T>(),
        });
    }
    points
}

fn generate_random_vector_to_test<T>(size: usize) -> Vec<[T; 3]>
where
    T: Num + Copy + MinMax + PartialOrd + FromPrimitive + std::ops::AddAssign, Standard: Distribution<T>
{
    let mut points = Vec::with_capacity(size);
    for _ in 0..size {
        points.push([random::<T>(), random::<T>(), random::<T>()]);
    }
    points
}


fn main() {
    let size = 800000;
    let test_size = 1;

    let dataset = generate_random_point_clouds_of_size::<f32>(size);
    let entries: Vec<[f64; 3]> = dataset.vec.iter().map(|point| [point.x as f64, point.y as f64, point.z as f64]).collect();
    let points_to_test: Vec<Point<f32>> = generate_random_points_to_test(test_size);
    let points_to_test_vec: Vec<[f64; 3]> = points_to_test.iter().map(|point| [point.x as f64, point.y as f64, point.z as f64]).collect();

    // Build
    let mut kdtree = KDTreeSingleIndex::new(&dataset, KDTreeSingleIndexParams::new());
    let start = std::time::Instant::now();
    kdtree.build_index();
    let elapsed = start.elapsed();
    println!("Time taken to build for nanofrann: {:?}", elapsed);

    let mut kdtree_flat = KDTreeSingleIndexFlat::new(&dataset, KDTreeSingleIndexParams::new());
    let start = std::time::Instant::now();
    kdtree_flat.build_index();
    let elapsed = start.elapsed();
    println!("Time taken to build for nanofrann_flat: {:?}", elapsed);

    // println!("flat vind: {:?}", kdtree_flat.vind);
    // println!("normal vind: {:?}", kdtree.vind);

    let start = std::time::Instant::now();
    let mut kiddo: kiddo::ImmutableKdTree<_, 3> = (&*entries).into();
    let elapsed = start.elapsed();
    println!("Time taken to build for kiddo: {:?}", elapsed);

    let start = std::time::Instant::now();
    let kd_tree = kd_tree::KdTree::build_by_ordered_float(entries.clone());
    let elapsed = start.elapsed();
    println!("Time taken to build for kd_tree: {:?}", elapsed);

    // Warm-up
    for point in &points_to_test {
        let mut result_0 = KNNResultSet::new_with_capacity(10);
        kdtree.knn_search(point, 10, &mut result_0);
    }

    for point in &points_to_test {
        let mut result_0 = KNNResultSet::new_with_capacity(10);
        kdtree_flat.knn_search(point, 10, &mut result_0);
    }

    for point in &points_to_test_vec {
        kiddo.nearest_n::<SquaredEuclidean>(point, 10);
    }

    for point in &points_to_test_vec {
        kd_tree.nearests(point, 10);
    }

    // Test KDTreeSingleIndex
    kdtree.build_index();
    println!("Querying {} points for 10 nearest neighbours using nanofrann", test_size);
    let start = std::time::Instant::now();
    for point in points_to_test.iter() {
        let mut result = KNNResultSet::new_with_capacity(10);
        kdtree.knn_search(point, 10, &mut result);
    }
    let duration = start.elapsed();
    println!("Time taken: {:?}", duration);

    // Test KDTreeSingleIndexFlat
    kdtree_flat.build_index();
    println!("Querying {} points for 10 nearest neighbours using nanofrann_flat", test_size);
    let start = std::time::Instant::now();
    for point in points_to_test.iter() {
        let mut result = KNNResultSet::new_with_capacity(10);
        kdtree_flat.knn_search(point, 10, &mut result);
    }
    let duration = start.elapsed();
    println!("Time taken: {:?}", duration);


    // Test Kiddo
    let kiddo: kiddo::ImmutableKdTree<_, 3> = (&*entries).into();
    println!("Querying {} points for 10 nearest neighbours using kiddo", test_size);
    let start_time = std::time::Instant::now();
    for point in points_to_test_vec.iter() {
        kiddo.nearest_n::<SquaredEuclidean>(point, 10);
    }
    let elapsed_time = start_time.elapsed();
    println!("Time taken: {:?}", elapsed_time);

    // Test kd_tree
    let kd_tree = kd_tree::KdTree::build_by_ordered_float(entries);
    println!("Querying {} points for 10 nearest neighbours using kd_tree", test_size);
    let start_time = std::time::Instant::now();
    for point in points_to_test_vec.iter() {
        kd_tree.nearests(point, 10);
    }
    let elapsed_time = start_time.elapsed();
    println!("Time taken: {:?}", elapsed_time);

    // Test correctness
    let mut result_nano = KNNResultSet::new_with_capacity(10);
    kdtree.knn_search(&points_to_test[0], 10, &mut result_nano);
    let mut result_nano_flat = KNNResultSet::new_with_capacity(10);
    kdtree_flat.knn_search(&points_to_test[0], 10, &mut result_nano_flat);
    let result_kiddo = kiddo.nearest_n::<SquaredEuclidean>(&points_to_test_vec[0], 10);
    let result_kiddo_indices: Vec<u64> = result_kiddo.iter().map(|neigbour| neigbour.item).collect();

    println!("Result from KDTreeSingleIndexFlat: {:?}", result_nano_flat.indices);
    println!("Result from KDTreeSingleIndex: {:?}", result_nano.indices);
    println!("Result from kiddo: {:?}", result_kiddo_indices);
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
        let mut kdtree = KDTreeSingleIndex::new(&dataset, params);

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
        let mut kdtree = KDTreeSingleIndex::new(&dataset, params);

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
