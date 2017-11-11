/// The future place for code that works with data.
/// TODO: write functions that loads csv data.
/// TODO: write functions that write existing NN to file.


struct DataSet<'a>{
    x: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>
}


trait Extractable {
    fn rand_element<'a>() -> (&'a Vec<f64>, &'a Vec<f64>);
    fn get<'a>(i: i32) -> (&'a Vec<f64>, &'a Vec<f64>);
}