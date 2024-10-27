pub trait MinMax {
    const MIN: Self;
    const MAX: Self;
    const ZERO: Self;
}

impl MinMax for u8 {
    const MIN: u8 = u8::MIN;
    const MAX: u8 = u8::MAX;
    const ZERO: u8 = 0;
}

impl MinMax for u16 {
    const MIN: u16 = u16::MIN;
    const MAX: u16 = u16::MAX;
    const ZERO: u16 = 0;
}

impl MinMax for u32 {
    const MIN: u32 = u32::MIN;
    const MAX: u32 = u32::MAX;
    const ZERO: u32 = 0;
}

impl MinMax for u64 {
    const MIN: u64 = u64::MIN;
    const MAX: u64 = u64::MAX;
    const ZERO: u64 = 0;
}

impl MinMax for u128 {
    const MIN: u128 = u128::MIN;
    const MAX: u128 = u128::MAX;
    const ZERO: u128 = 0;
}

impl MinMax for usize {
    const MIN: usize = usize::MIN;
    const MAX: usize = usize::MAX;
    const ZERO: usize = 0;
}

impl MinMax for i8 {
    const MIN: i8 = i8::MIN;
    const MAX: i8 = i8::MAX;
    const ZERO: i8 = 0;
}

impl MinMax for i16 {
    const MIN: i16 = i16::MIN;
    const MAX: i16 = i16::MAX;
    const ZERO: i16 = 0;
}

impl MinMax for i32 {
    const MIN: i32 = i32::MIN;
    const MAX: i32 = i32::MAX;
    const ZERO: i32 = 0;
}

impl MinMax for i64 {
    const MIN: i64 = i64::MIN;
    const MAX: i64 = i64::MAX;
    const ZERO: i64 = 0;
}

impl MinMax for i128 {
    const MIN: i128 = i128::MIN;
    const MAX: i128 = i128::MAX;
    const ZERO: i128 = 0;
}

impl MinMax for isize {
    const MIN: isize = isize::MIN;
    const MAX: isize = isize::MAX;
    const ZERO: isize = 0;
}

impl MinMax for f32 {
    const MIN: f32 = f32::MIN;
    const MAX: f32 = f32::MAX;
    const ZERO: f32 = 0.0;
}

impl MinMax for f64 {
    const MIN: f64 = f64::MIN;
    const MAX: f64 = f64::MAX;
    const ZERO: f64 = 0.0;
}
