


/// # Widrow's rule of thumb
/// This is an empirical rule that shows the size of training sample
/// in order to get good generalization.
/// ## Example
/// For network architecture [2, 1] and allowed error 0.1 (10%)
/// the size of training sample must exceed the amount of free
/// network parameters in 10 times
pub fn widrows(architecture: &[i32], allowed_error: f64) -> f64 {
    let mut s = architecture[0]*(architecture[0] + 1);

    for i in 1..architecture.len(){
        s += architecture[i] * architecture[i - 1] + architecture[i];
    }

    (s as f64) / allowed_error
}

/// # Optimal Brain Surgery (OBS)
/// OBS is procedure that helps get rid of redundant free network
/// parameters (e.g. weights).
/// After applying of OBS you need to train your network again.
#[allow(non_snake_case)]
pub fn OBS(nn: &super::FeedForward) {
    unimplemented!();
}