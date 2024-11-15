use std::hint::black_box;
use std::time::Duration;
use nanofrann::common::common::{DataSource, KDTreeSingleIndexParams, KNNResultSet, Point};
extern crate ply_rs;
use ply_rs::parser;
use criterion::{criterion_group, criterion_main, Criterion};
use ply_rs::ply::{Ply, Property};
use nanofrann::tree::nanofrann_float::KDTreeSingleIndex;

#[derive(Debug, Clone)] // not necessary for parsing, only for println at end of example.
struct Vertex {
    x: f64,
    y: f64,
    z: f64,
}

fn get_f64_property(prop: Option<&Property>) -> Option<f64> {
    match prop {
        Some(Property::Float(v)) => Some(*v as f64),
        Some(Property::Double(v)) => Some(*v),
        _ => None,
    }
}
fn read_ply(file_path: &str) -> Vec<Vertex> {
    let f = std::fs::File::open(file_path).unwrap();
    let mut f = std::io::BufReader::new(f);
    let mut parser = parser::Parser::<ply_rs::ply::DefaultElement>::new();
    let ply: Ply<ply_rs::ply::DefaultElement> = parser.read_ply(&mut f).unwrap();
    let vertices = Vec::new();
    if let Some(ply_vertices) = ply.payload.get("vertex") {
        let vertices = ply_vertices.clone();
        let vertices = vertices.into_iter().map(|vertex| {
            let x = get_f64_property(vertex.get("x")).unwrap();
            let y = get_f64_property(vertex.get("y")).unwrap();
            let z = get_f64_property(vertex.get("z")).unwrap();
            Vertex { x, y, z }
        }).collect();
        vertices
    } else {
        vertices
    }
}

fn setup_dataset_for_nanofrann(vertices: Vec<Vertex>) -> DataSource<f64> {
    let mut data = DataSource::with_capacity(vertices.len());
    for vertex in vertices.iter() {
        data.add_point(vertex.x, vertex.y, vertex.z);
    }
    data
}

fn setup_dataset_for_kd_tree(vertices: Vec<Vertex>) -> Vec<[f64; 3]> {
    let mut data = Vec::new();
    for vertex in vertices.iter() {
        data.push([vertex.x, vertex.y, vertex.z]);
    }
    data
}

fn setup_dataset_for_kiddo(vertices: Vec<Vertex>) -> Vec<[f64; 3]> {
    let mut data = Vec::new();
    for vertex in vertices.iter() {
        data.push([vertex.x, vertex.y, vertex.z]);
    }
    data
}
fn benchmark_kd_trees(c: &mut Criterion) {
    let mut group = c.benchmark_group("kd_tree_construction");
    group.sample_size(10);

    // Reading PLY file
    let path = "data/longdress/ply/longdress_vox10_1051.ply";
    let vertices = read_ply(path);

    // Nanofrann setup
    let nanofrann_data = setup_dataset_for_nanofrann(vertices.clone());
    let mut nanofrann_tree = KDTreeSingleIndex::new(&nanofrann_data, KDTreeSingleIndexParams::new());
    nanofrann_tree.build_index();

    // kd_tree setup
    let kd_tree_data = setup_dataset_for_kd_tree(vertices.clone());
    let mut kd_tree = kd_tree::KdTree::build_by_ordered_float(kd_tree_data.clone());

    let points_to_test_nanofrann : Vec<Point<f64>> = vertices.iter().map(|vertex| {
        Point::<f64> {
            x: vertex.x,
            y: vertex.y,
            z: vertex.z,
        }
    }).collect();
    group.bench_function("nanofrann_query_resample", |b| {
        b.iter(|| {
            for point in points_to_test_nanofrann.iter() {
                let mut result = KNNResultSet::<f64>::new_with_capacity(10);
                black_box(nanofrann_tree.knn_search(&point,  &mut result));
            }
        });
    });
    let points_to_test_kd_tree : Vec<[f64; 3]> = vertices.iter().map(|vertex| {
        [vertex.x as f64, vertex.y as f64, vertex.z as f64]
    }).collect();
    group.bench_function("kd_tree_query_resample", |b| {
        b.iter(|| {
            for point in &points_to_test_kd_tree {
                let res = kd_tree.nearests(point, 10);
                black_box(res);
            }
        });
    });
}

criterion_group!{
  name = benches;
  config = Criterion::default().measurement_time(Duration::from_secs(10));
  targets = benchmark_kd_trees
}
criterion_main!(benches);
